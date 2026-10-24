import Foundation

/// What the app receives from an engine: protocol events, then one `exited` at the end.
public enum EngineOutput: Sendable, Equatable {
    case event(EngineEvent)
    case exited(String)
}

/// A running engine the app can talk to; `EngineProcess` in the app, a fake in tests.
public protocol EngineTransport: Sendable {
    /// Starts the engine; the stream ends after `.exited`.
    func start() throws -> AsyncStream<EngineOutput>
    /// Writes one request line.
    func send(_ line: Data) throws
    /// Terminates the engine.
    func stop()
}

/// The Python engine as a child process speaking the JSON-lines protocol.
///
/// Stdout carries protocol lines only. Stderr (logs, tracebacks) goes to the log file,
/// and its last lines explain an unexpected exit. Each `start` launches a new process,
/// so the engine can be stopped to free its memory and started again.
public final class EngineProcess: EngineTransport, @unchecked Sendable {
    private let config: EngineConfig
    private let lock = NSLock()
    private var run: Run?
    private static let tailLines = 12
    private static let logLimit = 20_000_000

    /// One launched process with its pipes and output state.
    private final class Run: @unchecked Sendable {
        let process = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        private let lock = NSLock()
        private var splitter = LineSplitter()
        private var stderrTail: [String] = []
        private var stdoutEOF = false
        private var stderrEOF = false
        private var status: Int32?
        private var finished = false

        /// Records an EOF or the exit status; returns the status exactly once, when stdout
        /// EOF, stderr EOF and the status are all in, or with `giveUpOnStderr` when only
        /// stderr EOF is missing.
        func complete(stdoutEOF: Bool = false, stderrEOF: Bool = false, status: Int32? = nil,
                      giveUpOnStderr: Bool = false) -> Int32? {
            lock.lock()
            defer { lock.unlock() }
            if stdoutEOF { self.stdoutEOF = true }
            if stderrEOF { self.stderrEOF = true }
            if let status { self.status = status }
            guard self.stdoutEOF, self.stderrEOF || giveUpOnStderr, let done = self.status, !finished else {
                return nil
            }
            finished = true
            return done
        }

        /// The process is gone and stdout is done, but stderr is still open.
        var onlyStderrOpen: Bool {
            lock.lock()
            defer { lock.unlock() }
            return stdoutEOF && status != nil && !stderrEOF && !finished
        }

        func lines(from chunk: Data) -> [Data] {
            lock.lock()
            defer { lock.unlock() }
            return splitter.feed(chunk)
        }

        func remember(stderr chunk: Data) {
            let text = String(decoding: chunk, as: UTF8.self)
            lock.lock()
            defer { lock.unlock() }
            stderrTail.append(contentsOf: text.split(separator: "\n").map(String.init))
            stderrTail = Array(stderrTail.suffix(EngineProcess.tailLines))
        }

        func tail() -> String {
            lock.lock()
            defer { lock.unlock() }
            return stderrTail.suffix(3).joined(separator: " | ")
        }
    }

    /// How long the exit waits for stderr EOF once the process is gone and stdout is done:
    /// a child process that inherited stderr can keep it open after the engine exits.
    private let stderrGrace: Double

    public init(config: EngineConfig, stderrGrace: Double = 2) {
        self.config = config
        self.stderrGrace = stderrGrace
    }

    public func start() throws -> AsyncStream<EngineOutput> {
        let (stream, continuation) = AsyncStream.makeStream(of: EngineOutput.self)
        let log = try openLog()
        try FileManager.default.createDirectory(at: config.songsDir, withIntermediateDirectories: true)
        // The working folder; on the first start it is empty and the engine fills checkpoints/.
        try FileManager.default.createDirectory(at: config.project, withIntermediateDirectories: true)
        let run = Run()
        let process = run.process
        process.executableURL = config.python
        process.arguments = config.arguments
        process.environment = config.environment(base: ProcessInfo.processInfo.environment)
        process.currentDirectoryURL = config.project
        process.standardInput = run.stdin
        process.standardOutput = run.stdout
        process.standardError = run.stderr

        // The exit is reported after stdout EOF (every protocol line delivered), stderr EOF
        // (the last log lines in the tail) and the termination, whichever comes last;
        // nothing blocks waiting for the others.
        let finish: @Sendable (Int32) -> Void = { status in
            try? log.close()
            let tail = run.tail()
            continuation.yield(.exited("exit \(status)\(tail.isEmpty ? "" : ": \(tail)")"))
            continuation.finish()
        }
        let grace = stderrGrace
        // Once the process is gone and stdout is done, stderr gets `grace` seconds more.
        let waitForStderr: @Sendable () -> Void = {
            guard run.onlyStderrOpen else { return }
            DispatchQueue.global().asyncAfter(deadline: .now() + grace) {
                if let status = run.complete(giveUpOnStderr: true) { finish(status) }
            }
        }
        process.terminationHandler = { process in
            if let status = run.complete(status: process.terminationStatus) {
                finish(status)
            } else {
                waitForStderr()
            }
        }
        run.stdout.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            for line in run.lines(from: chunk) {
                do {
                    continuation.yield(.event(try EngineEvent.decode(line)))
                } catch {
                    let text = String(decoding: line, as: UTF8.self)
                    continuation.yield(.event(.error(id: nil, message: "bad engine line: \(text)")))
                }
            }
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                if let status = run.complete(stdoutEOF: true) {
                    finish(status)
                } else {
                    waitForStderr()
                }
            }
        }
        run.stderr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                // EOF; without this the handler keeps firing on an empty pipe.
                handle.readabilityHandler = nil
                if let status = run.complete(stderrEOF: true) { finish(status) }
                return
            }
            try? log.write(contentsOf: data)
            run.remember(stderr: data)
        }
        try process.run()
        lock.lock()
        self.run = run
        lock.unlock()
        return stream
    }

    public func send(_ line: Data) throws {
        guard let run = current() else { throw CocoaError(.fileWriteNoPermission) }
        try run.stdin.fileHandleForWriting.write(contentsOf: line)
    }

    public func stop() {
        if let process = current()?.process, process.isRunning {
            process.terminate()
        }
    }

    private func current() -> Run? {
        lock.lock()
        defer { lock.unlock() }
        return run
    }

    /// The engine log, appended to by every launch; started over once it passes `logLimit`.
    private func openLog() throws -> FileHandle {
        let fm = FileManager.default
        try fm.createDirectory(at: config.logFile.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        let size = (try? fm.attributesOfItem(atPath: config.logFile.path)[.size] as? Int) ?? nil
        if size == nil || size! > Self.logLimit {
            fm.createFile(atPath: config.logFile.path, contents: nil)
        }
        let log = try FileHandle(forWritingTo: config.logFile)
        try log.seekToEnd()
        try log.write(contentsOf: Data("=== engine start \(Date())\n".utf8))
        return log
    }
}
