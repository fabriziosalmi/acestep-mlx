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
/// and its last lines explain an unexpected exit.
public final class EngineProcess: EngineTransport, @unchecked Sendable {
    private let config: EngineConfig
    private let process = Process()
    private let stdin = Pipe()
    private let stdout = Pipe()
    private let stderr = Pipe()
    private let lock = NSLock()
    private var splitter = LineSplitter()
    private var stderrTail: [String] = []
    private static let tailLines = 12

    public init(config: EngineConfig) {
        self.config = config
    }

    public func start() throws -> AsyncStream<EngineOutput> {
        let (stream, continuation) = AsyncStream.makeStream(of: EngineOutput.self)
        let log = try openLog()
        try FileManager.default.createDirectory(at: config.songsDir, withIntermediateDirectories: true)
        process.executableURL = config.python
        process.arguments = config.arguments
        process.environment = config.environment(base: ProcessInfo.processInfo.environment)
        process.currentDirectoryURL = config.project
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr

        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self else { return }
            let chunk = handle.availableData
            for line in self.lines(from: chunk) {
                do {
                    continuation.yield(.event(try EngineEvent.decode(line)))
                } catch {
                    let text = String(decoding: line, as: UTF8.self)
                    continuation.yield(.event(.error(id: nil, message: "bad engine line: \(text)")))
                }
            }
            if chunk.isEmpty {
                // EOF: every protocol line has been delivered, so the exit comes last.
                handle.readabilityHandler = nil
                self.process.waitUntilExit()
                self.stderr.fileHandleForReading.readabilityHandler = nil
                let tail = self.tail()
                let status = self.process.terminationStatus
                continuation.yield(.exited("exit \(status)\(tail.isEmpty ? "" : ": \(tail)")"))
                continuation.finish()
            }
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            try? log.write(contentsOf: data)
            self?.remember(stderr: data)
        }
        try process.run()
        return stream
    }

    public func send(_ line: Data) throws {
        try stdin.fileHandleForWriting.write(contentsOf: line)
    }

    public func stop() {
        if process.isRunning {
            process.terminate()
        }
    }

    private func openLog() throws -> FileHandle {
        let fm = FileManager.default
        try fm.createDirectory(at: config.logFile.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        fm.createFile(atPath: config.logFile.path, contents: nil)
        return try FileHandle(forWritingTo: config.logFile)
    }

    private func lines(from chunk: Data) -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        return splitter.feed(chunk)
    }

    private func remember(stderr chunk: Data) {
        let text = String(decoding: chunk, as: UTF8.self)
        lock.lock()
        defer { lock.unlock() }
        stderrTail.append(contentsOf: text.split(separator: "\n").map(String.init))
        stderrTail = Array(stderrTail.suffix(Self.tailLines))
    }

    private func tail() -> String {
        lock.lock()
        defer { lock.unlock() }
        return stderrTail.suffix(3).joined(separator: " | ")
    }
}
