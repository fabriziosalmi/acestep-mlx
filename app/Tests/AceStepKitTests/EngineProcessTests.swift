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
}
