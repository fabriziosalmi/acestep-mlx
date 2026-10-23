import Foundation
import Observation

/// What the engine is doing, as the window shows it.
public enum EngineState: Equatable, Sendable {
    /// The development install is incomplete; the paths are missing.
    case notInstalled([String])
    case starting(since: Date)
    case ready
    case working(Phase, since: Date)
    case failed(String)
}

/// The app's single source of truth: engine state, the prompt, and the songs made so far.
@MainActor
@Observable
public final class AppModel {
    public private(set) var state: EngineState
    public private(set) var songs: [Song]
    /// The last failed request's message; cleared by the next request.
    public private(set) var lastError: String?
    public var prompt = ""

    @ObservationIgnored private let transport: EngineTransport?
    @ObservationIgnored private let now: () -> Date
    @ObservationIgnored private let makeID: () -> String
    @ObservationIgnored private var currentRequest: String?
    @ObservationIgnored private var listener: Task<Void, Never>?

    /// - Parameters:
    ///   - transport: The engine, or nil when `missing` lists what is not installed.
    ///   - songs: Songs already on disk, newest first.
    public init(
        transport: EngineTransport?,
        missing: [String] = [],
        songs: [Song] = [],
        now: @escaping () -> Date = Date.init,
        makeID: @escaping () -> String = { UUID().uuidString.lowercased() }
    ) {
        self.transport = transport
        self.songs = songs
        self.now = now
        self.makeID = makeID
        state = transport == nil ? .notInstalled(missing) : .starting(since: now())
    }

    public var canMake: Bool {
        state == .ready && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Launches the engine and follows its output until it exits.
    public func start() {
        guard let transport, listener == nil else { return }
        do {
            let stream = try transport.start()
            listener = Task { [weak self] in
                for await output in stream {
                    self?.handle(output)
                }
            }
        } catch {
            state = .failed("could not launch the engine: \(error.localizedDescription)")
        }
    }

    /// Asks the engine for a song from the current prompt.
    public func make() {
        guard canMake, let transport else { return }
        let request = GenerateRequest(
            id: makeID(), prompt: prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        )
        do {
            try transport.send(try request.line())
        } catch {
            lastError = "could not reach the engine: \(error.localizedDescription)"
            return
        }
        currentRequest = request.id
        lastError = nil
        state = .working(.lyrics, since: now())
    }

    public func stop() {
        transport?.stop()
    }

    /// Applies one engine output to the state; internal so tests can drive it directly.
    func handle(_ output: EngineOutput) {
        switch output {
        case .exited(let reason):
            currentRequest = nil
            state = .failed("The engine stopped (\(reason)).")
        case .event(.ready):
            if case .starting = state { state = .ready }
        case let .event(.phase(id, phase)):
            if id == currentRequest, case let .working(_, since) = state {
                state = .working(phase, since: since)
            }
        case let .event(.done(id, song)):
            guard id == currentRequest else { return }
            songs.insert(song, at: 0)
            finishRequest()
        case let .event(.error(id, message)):
            if case .starting = state {
                state = .failed(message)
            } else if id == nil || id == currentRequest {
                lastError = message
                if id != nil { finishRequest() }
            }
        }
    }

    private func finishRequest() {
        currentRequest = nil
        state = .ready
    }
}
