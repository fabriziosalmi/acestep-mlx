import Foundation
import Testing

@testable import AceStepKit

/// Runs EngineProcess against a shell script that speaks the protocol and then exits 3.
@Suite struct EngineProcessTests {
    @Test func deliversEveryEventBeforeTheExit() async throws {
        let root = try temporaryDirectory()
        let script = root.appendingPathComponent("fake-engine")
        try Data("""
        #!/bin/sh
        echo 'loading models' >&2
        printf '{"event": "ready", "protocol": 1, "load_s": 0.1}\\n'
        read request
        printf '{"event": "phase", "id": "r1", "phase": "lyrics"}\\n{"event": "phase", '
        printf '"id": "r1", "phase": "rendering"}\\n'
        echo 'not json'
        printf '{"event": "error", "id": "r1", "message": "synthetic"}\\n'
        echo 'Traceback: last words' >&2
        exit 3

        """.utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let config = EngineConfig(python: script, engineDir: root, project: root,
                                  songsDir: root.appendingPathComponent("songs"),
                                  logFile: root.appendingPathComponent("logs/engine.log"))
        let engine = EngineProcess(config: config)
        let stream = try engine.start()
        try engine.send(GenerateRequest(id: "r1", prompt: "x").line())

        var outputs: [EngineOutput] = []
        for await output in stream { outputs.append(output) }

        #expect(Array(outputs.prefix(4)) == [
            .event(.ready(loadSeconds: 0.1)),
            .event(.phase(id: "r1", phase: .lyrics)),
            .event(.phase(id: "r1", phase: .rendering)),
            .event(.error(id: nil, message: "bad engine line: not json")),
        ])
        #expect(outputs.count == 6)
        #expect(outputs[4] == .event(.error(id: "r1", message: "synthetic")))
        guard case let .exited(reason) = outputs.last else {
            Issue.record("last output is not the exit")
            return
        }
        #expect(reason.hasPrefix("exit 3"))
        #expect(reason.contains("Traceback: last words"))
        let log = try String(contentsOf: config.logFile, encoding: .utf8)
        #expect(log.contains("loading models"))
    }

    /// A child keeps stderr open past the engine's exit: its last line still reaches the tail,
    /// and a child that keeps it open for long does not hold the exit back.
    @Test func theExitWaitsBrieflyForStderr() async throws {
        for (holdSeconds, expectLine) in [("0.5", true), ("8", false)] {
            let root = try temporaryDirectory()
            let script = root.appendingPathComponent("fake-engine")
            try Data("""
            #!/bin/sh
            printf '{"event": "ready", "protocol": 2, "load_s": 0.1}\\n'
            exec 1>&-
            (sleep \(holdSeconds); echo 'late words' >&2) &
            exit 3

            """.utf8).write(to: script)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
            let config = EngineConfig(python: script, engineDir: root, project: root,
                                      songsDir: root.appendingPathComponent("songs"),
                                      logFile: root.appendingPathComponent("logs/engine.log"))
            let started = Date()
            var reason: String?
            for await output in try EngineProcess(config: config, stderrGrace: 2).start() {
                if case let .exited(text) = output { reason = text }
            }
            let waited = Date().timeIntervalSince(started)
            #expect(reason?.hasPrefix("exit 3") == true)
            #expect(reason?.contains("late words") == expectLine, "hold \(holdSeconds) s: \(reason ?? "-")")
            #expect(waited < 4, "hold \(holdSeconds) s: exit after \(waited) s")
        }
    }

    @Test func startsAgainAfterAnExit() async throws {
        let root = try temporaryDirectory()
        let script = root.appendingPathComponent("fake-engine")
        try Data("""
        #!/bin/sh
        echo "launch $$" >&2
        printf '{"event": "ready", "protocol": 1, "load_s": 0.1}\\n'
        read request
        exit 0

        """.utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let config = EngineConfig(python: script, engineDir: root, project: root,
                                  songsDir: root.appendingPathComponent("songs"),
                                  logFile: root.appendingPathComponent("logs/engine.log"))
        let engine = EngineProcess(config: config)
        for _ in 0..<2 {
            let stream = try engine.start()
            try engine.send(GenerateRequest(id: "r1", prompt: "x").line())
            var outputs: [EngineOutput] = []
            for await output in stream { outputs.append(output) }
            #expect(outputs.first == .event(.ready(loadSeconds: 0.1)))
            #expect(outputs.count == 2)
            guard case let .exited(reason) = outputs.last else {
                Issue.record("last output is not the exit")
                return
            }
            #expect(reason.hasPrefix("exit 0: launch "))
        }
        let log = try String(contentsOf: config.logFile, encoding: .utf8)
        #expect(log.components(separatedBy: "=== engine start").count == 3)
        #expect(log.components(separatedBy: "launch ").count == 3)
    }

    /// Regression: the exit used to wait in waitUntilExit on the pipe queue and hung in
    /// about 1 launch in 10 (30 of 300 measured); the stream must always end.
    @Test func everyLaunchEndsWithItsExit() async throws {
        let root = try temporaryDirectory()
        let script = root.appendingPathComponent("fake-engine")
        try Data("""
        #!/bin/sh
        echo "launch $$" >&2
        printf '{"event": "ready", "protocol": 1, "load_s": 0.1}\\n'
        read request
        echo 'bye' >&2
        exit 0

        """.utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let config = EngineConfig(python: script, engineDir: root, project: root,
                                  songsDir: root.appendingPathComponent("songs"),
                                  logFile: root.appendingPathComponent("logs/engine.log"))
        var hangs = 0, bad = 0
        for _ in 0..<100 {
            let engine = EngineProcess(config: config)
            let stream = try engine.start()
            try engine.send(GenerateRequest(id: "r1", prompt: "x").line())
            let outcome = await withTaskGroup(of: [EngineOutput]?.self) { group in
                group.addTask { var o: [EngineOutput] = []; for await x in stream { o.append(x) }; return o }
                group.addTask { try? await Task.sleep(for: .seconds(5)); return nil }
                let first = await group.next()!
                group.cancelAll()
                return first
            }
            guard let outputs = outcome else { hangs += 1; continue }
            if outputs.count != 2 || outputs.first != .event(.ready(loadSeconds: 0.1)) { bad += 1 }
        }
        #expect(hangs == 0 && bad == 0)
    }

    /// The exit must wait for stdout EOF: lines still in the pipe when the process ends
    /// are delivered first.
    @Test func linesWrittenJustBeforeTheExitAreKept() async throws {
        let root = try temporaryDirectory()
        let script = root.appendingPathComponent("fake-engine")
        try Data("""
        #!/bin/sh
        printf '{"event": "ready", "protocol": 1, "load_s": 0.1}\\n'
        i=0
        while [ $i -lt 3000 ]; do
          printf '{"event": "phase", "id": "r1", "phase": "lyrics"}\\n'
          i=$((i+1))
        done
        exit 0

        """.utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let config = EngineConfig(python: script, engineDir: root, project: root,
                                  songsDir: root.appendingPathComponent("songs"),
                                  logFile: root.appendingPathComponent("logs/engine.log"))
        for _ in 0..<20 {
            var outputs: [EngineOutput] = []
            for await output in try EngineProcess(config: config).start() { outputs.append(output) }
            #expect(outputs.count == 3002)
            #expect(outputs.last == .exited("exit 0"))
        }
    }
}
