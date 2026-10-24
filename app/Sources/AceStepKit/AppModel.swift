import Foundation
import Observation

/// What the engine is doing, as the window shows it.
public enum EngineState: Equatable, Sendable {
    /// The engine's files are incomplete; the paths are missing.
    case notInstalled([String])
    case starting(since: Date)
    /// First start: the engine is fetching model files; bytes present of bytes needed.
    case downloading(done: Int64, total: Int64)
    case ready
    /// The engine was stopped after an idle period to give its memory back; the next
    /// request starts it again.
    case asleep
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
    /// The user's own lyrics; blank means the LM writes them.
    public var lyrics = ""
    /// The engine's process has ended on its own (not for idleness); observed, so the window
    /// shows "Try again" as soon as the exit arrives.
    public private(set) var engineExited = false

    @ObservationIgnored private let transport: EngineTransport?
    @ObservationIgnored private let now: () -> Date
    @ObservationIgnored private let makeID: () -> String
    @ObservationIgnored private var currentRequest: String?
    @ObservationIgnored private var listener: Task<Void, Never>?
    @ObservationIgnored private let idleSeconds: Double?
    @ObservationIgnored private let sleeper: @Sendable (Double) async -> Void
    /// Internal so tests can await the timer they released.
    @ObservationIgnored var idleTimer: Task<Void, Never>?
    /// Bumped by every armed timer; a timer only acts if no newer one was armed after it.
    @ObservationIgnored private(set) var activity = 0
    /// The engine was told to stop for idleness and its exit has not arrived yet.
    @ObservationIgnored private var stopping = false
    /// A request waiting for the engine to be ready again.
    @ObservationIgnored private var pending: GenerateRequest?

    /// - Parameters:
    ///   - transport: The engine, or nil when `missing` lists what is not installed.
    ///   - songs: Songs already on disk, newest first.
    ///   - idleSeconds: Stop the engine after this long ready with nothing to do; nil never stops it.
    ///   - sleeper: Waits that many seconds; tests replace it.
    public init(
        transport: EngineTransport?,
        missing: [String] = [],
        songs: [Song] = [],
        idleSeconds: Double? = nil,
        now: @escaping () -> Date = Date.init,
        makeID: @escaping () -> String = { UUID().uuidString.lowercased() },
        sleeper: @escaping @Sendable (Double) async -> Void = { try? await Task.sleep(for: .seconds($0)) }
    ) {
        self.transport = transport
        self.songs = songs
        self.idleSeconds = idleSeconds
        self.now = now
        self.makeID = makeID
        self.sleeper = sleeper
        state = transport == nil ? .notInstalled(missing) : .starting(since: now())
    }

    /// Ready for a request, possibly after starting the engine again.
    public var canRequest: Bool {
        state == .ready || state == .asleep
    }

    /// The engine failed and its process is gone, so it can be launched again.
    public var canRetry: Bool {
        if case .failed = state { return engineExited }
        return false
    }

    public var canMake: Bool {
        canRequest && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Launches the engine and follows its output until it exits.
    public func start() {
        guard let transport, listener == nil else { return }
        engineExited = false
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

    /// Launches the engine again after a failure (a download cut short, a crash).
    public func retry() {
        guard canRetry else { return }
        lastError = nil
        state = .starting(since: now())
        start()
    }

    /// Takes a song out of the history; the caller moves its folder to the Trash.
    public func remove(_ song: Song) {
        songs.removeAll { $0.id == song.id }
    }

    /// Asks the engine for a song from the current prompt.
    public func make() {
        guard canMake else { return }
        let own = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        request(GenerateRequest(id: makeID(), prompt: prompt.trimmingCharacters(in: .whitespacesAndNewlines),
                                options: GenerateOptions(lyrics: own.isEmpty ? nil : own)))
    }

    /// Asks for another version of `song`: same text and metadata, new seed.
    public func makeAnother(like song: Song) {
        guard canRequest, let name = song.name else { return }
        request(GenerateRequest(id: makeID(), prompt: song.prompt, options: GenerateOptions(like: name)))
    }

    private func request(_ request: GenerateRequest) {
        guard state == .asleep else { return send(request) }
        pending = request
        lastError = nil
        state = .starting(since: now())
        // While the old process is still exiting, the exit starts the new one.
        if !stopping { start() }
    }

    private func send(_ request: GenerateRequest) {
        guard let transport else { return }
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
        idleTimer?.cancel()
        transport?.stop()
    }

    private func armIdleTimer() {
        guard let idleSeconds else { return }
        activity += 1
        let mark = activity
        idleTimer?.cancel()
        idleTimer = Task { [weak self, sleeper] in
            await sleeper(idleSeconds)
            self?.idleElapsed(mark)
        }
    }

    /// Internal so tests can fire a stale timer on purpose.
    func idleElapsed(_ mark: Int) {
        guard mark == activity, state == .ready, currentRequest == nil else { return }
        stopping = true
        state = .asleep
        transport?.stop()
    }

    /// Applies one engine output to the state; internal so tests can drive it directly.
    func handle(_ output: EngineOutput) {
        switch output {
        case .exited(let reason):
            listener = nil
            if stopping {
                stopping = false
                if pending != nil { start() }
                return
            }
            currentRequest = nil
            pending = nil
            engineExited = true
            // An engine that failed to start says why before it exits; keep that message.
            if case .failed = state { return }
            state = .failed("The engine stopped (\(reason)).")
        case let .event(.download(done, total, complete)):
            guard isStarting else { return }
            state = complete ? .starting(since: now()) : .downloading(done: done, total: total)
        case .event(.ready):
            guard isStarting else { return }
            state = .ready
            if let request = pending {
                pending = nil
                send(request)
            } else {
                armIdleTimer()
            }
        case let .event(.phase(id, phase)):
            if id == currentRequest, case let .working(_, since) = state {
                state = .working(phase, since: since)
            }
        case let .event(.done(id, song)):
            guard id == currentRequest else { return }
            songs.insert(song, at: 0)
            finishRequest()
        case let .event(.error(id, message)):
            if isStarting {
                state = .failed(message)
            } else if id == nil || id == currentRequest {
                lastError = message
                if id != nil { finishRequest() }
            }
        }
    }

    /// Launched and not ready yet: loading models or fetching them first.
    private var isStarting: Bool {
        switch state {
        case .starting, .downloading: true
        default: false
        }
    }

    private func finishRequest() {
        currentRequest = nil
        state = .ready
        armIdleTimer()
    }
}
