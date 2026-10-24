import Foundation
import Testing

@testable import AceStepKit

/// Records sent lines, starts and stops; can be told to fail sends. Every start after the
/// first hands out a new stream, like a new engine process.
final class FakeTransport: EngineTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var _sent: [Data] = []
    private(set) var starts = 0
    private(set) var stops = 0
    var failSends = false
    private(set) var stream: AsyncStream<EngineOutput>
    private(set) var continuation: AsyncStream<EngineOutput>.Continuation

    init() {
        (stream, continuation) = AsyncStream.makeStream(of: EngineOutput.self)
    }

    /// Sent requests flattened to strings: top-level fields plus "options.<key>".
    var sent: [[String: String]] {
        lock.lock()
        defer { lock.unlock() }
        return _sent.compactMap { line in
            guard let object = try? JSONSerialization.jsonObject(with: line.dropLast()) as? [String: Any]
            else { return nil }
            var flat: [String: String] = [:]
            for (key, value) in object {
                if let options = value as? [String: Any] {
                    for (name, option) in options { flat["options.\(name)"] = "\(option)" }
                } else {
                    flat[key] = "\(value)"
                }
            }
            return flat
        }
    }

    func start() throws -> AsyncStream<EngineOutput> {
        if starts > 0 { (stream, continuation) = AsyncStream.makeStream(of: EngineOutput.self) }
        starts += 1
        return stream
    }

    func send(_ line: Data) throws {
        if failSends { throw CocoaError(.fileWriteUnknown) }
        lock.lock()
        _sent.append(line)
        lock.unlock()
    }

    /// Counts only: tests deliver the exit themselves.
    func stop() { stops += 1 }
}

/// Idle timers that wait until the test opens the gate; once open, every timer fires at once.
final class IdleGate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    var sleeper: @Sendable (Double) async -> Void {
        { [self] _ in
            await withCheckedContinuation { c in
                let now = lock.withLock { () -> Bool in
                    if !opened { waiting.append(c) }
                    return opened
                }
                if now { c.resume() }
            }
        }
    }

    /// Releases every timer, then waits until the model's current timer has run.
    @MainActor
    func open(_ model: AppModel) async {
        let released = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
            opened = true
            defer { waiting = [] }
            return waiting
        }
        released.forEach { $0.resume() }
        await model.idleTimer?.value
    }
}

let t0 = Date(timeIntervalSince1970: 1_000)

func song(_ prompt: String) -> Song {
    Song(prompt: prompt, caption: "", lyrics: "", audio: "a.flac", created: "2026-10-23", dir: "/s/\(prompt)")
}

@MainActor
func readyModel(_ transport: FakeTransport) -> AppModel {
    let model = AppModel(transport: transport, now: { t0 }, makeID: { "req-1" })
    model.handle(.event(.ready(loadSeconds: 30)))
    return model
}

@MainActor
func sleepyModel(_ transport: FakeTransport, _ gate: IdleGate) -> AppModel {
    var ids = 0
    let model = AppModel(transport: transport, idleSeconds: 60, now: { t0 },
                         makeID: { ids += 1; return "req-\(ids)" }, sleeper: gate.sleeper)
    model.start()
    model.handle(.event(.ready(loadSeconds: 30)))
    return model
}

@MainActor
@Suite struct AppModelTests {
    @Test func startsLoadingAndBecomesReady() {
        let model = AppModel(transport: FakeTransport(), now: { t0 })
        #expect(model.state == .starting(since: t0))
        model.handle(.event(.ready(loadSeconds: 30)))
        #expect(model.state == .ready)
    }

    @Test func aFirstStartDownloadsThenLoadsThenIsReady() {
        var clock = t0
        let model = AppModel(transport: FakeTransport(), now: { clock })
        model.prompt = "song"
        model.handle(.event(.download(done: 0, total: 100, complete: false)))
        #expect(model.state == .downloading(done: 0, total: 100))
        model.handle(.event(.download(done: 60, total: 100, complete: false)))
        #expect(model.state == .downloading(done: 60, total: 100))
        #expect(!model.canMake)
        clock = t0.addingTimeInterval(500)
        model.handle(.event(.download(done: 100, total: 100, complete: true)))
        #expect(model.state == .starting(since: t0.addingTimeInterval(500)))
        model.handle(.event(.ready(loadSeconds: 30)))
        #expect(model.state == .ready)
        // A late download line once ready changes nothing.
        model.handle(.event(.download(done: 1, total: 100, complete: false)))
        #expect(model.state == .ready)
    }

    @Test func aDownloadFailureIsFatal() {
        let model = AppModel(transport: FakeTransport(), now: { t0 })
        model.handle(.event(.download(done: 10, total: 100, complete: false)))
        model.handle(.event(.error(id: nil, message: "models could not be downloaded: x")))
        #expect(model.state == .failed("models could not be downloaded: x"))
    }

    @Test func readyDuringADownloadIsAccepted() {
        let model = AppModel(transport: FakeTransport(), now: { t0 })
        model.handle(.event(.download(done: 10, total: 100, complete: false)))
        model.handle(.event(.ready(loadSeconds: 30)))
        #expect(model.state == .ready)
    }

    @Test func withoutEngineShowsWhatIsMissing() {
        let model = AppModel(transport: nil, missing: ["/x/python"])
        #expect(model.state == .notInstalled(["/x/python"]))
        model.prompt = "song"
        #expect(!model.canMake)
    }

    @Test func makeSendsTrimmedPromptOnlyWhenReady() {
        let transport = FakeTransport()
        let model = AppModel(transport: transport, now: { t0 }, makeID: { "req-1" })
        model.prompt = "  folk ballad \n"
        model.make()
        #expect(transport.sent.isEmpty)
        model.handle(.event(.ready(loadSeconds: 1)))
        model.make()
        #expect(transport.sent == [["id": "req-1", "cmd": "generate", "prompt": "folk ballad"]])
        #expect(model.state == .working(.lyrics, since: t0))
        #expect(!model.canMake)
    }

    @Test func anotherLikeThisSendsTheSongFolder() {
        let transport = FakeTransport()
        let model = readyModel(transport)
        let base = Song(prompt: "dark techno", caption: "", lyrics: "", audio: "a.flac", created: "c",
                        dir: "/Users/x/Music/ACE-Step/20261023-194021-11111111")
        model.makeAnother(like: base)
        #expect(transport.sent == [["id": "req-1", "cmd": "generate", "prompt": "dark techno",
                                    "options.like": "20261023-194021-11111111"]])
        #expect(model.state == .working(.lyrics, since: t0))
        model.makeAnother(like: base)
        #expect(transport.sent.count == 1)
    }

    @Test func blankPromptCannotBeMade() {
        let model = readyModel(FakeTransport())
        model.prompt = " \n "
        #expect(!model.canMake)
    }

    @Test func phasesThenDoneAddTheSong() {
        let model = readyModel(FakeTransport())
        #expect(model.songs.isEmpty)
        model.prompt = "a"
        model.make()
        model.handle(.event(.phase(id: "other", phase: .rendering)))
        #expect(model.state == .working(.lyrics, since: t0))
        model.handle(.event(.phase(id: "req-1", phase: .composing)))
        #expect(model.state == .working(.composing, since: t0))
        model.handle(.event(.done(id: "other", song: song("x"))))
        #expect(model.songs.isEmpty)
        model.handle(.event(.done(id: "req-1", song: song("a"))))
        #expect(model.songs.map(\.prompt) == ["a"])
        #expect(model.state == .ready)
    }

    @Test func failedRequestKeepsEngineReady() {
        let model = readyModel(FakeTransport())
        model.prompt = "a"
        model.make()
        model.handle(.event(.error(id: "req-1", message: "RuntimeError: no")))
        #expect(model.state == .ready)
        #expect(model.lastError == "RuntimeError: no")
        model.make()
        #expect(model.lastError == nil)
    }

    @Test func loadFailureAndExitAreFatal() {
        let model = AppModel(transport: FakeTransport(), now: { t0 })
        model.handle(.event(.error(id: nil, message: "engine failed to load: x")))
        #expect(model.state == .failed("engine failed to load: x"))
        let other = readyModel(FakeTransport())
        other.handle(.exited("exit 1: Traceback"))
        #expect(other.state == .failed("The engine stopped (exit 1: Traceback)."))
    }

    @Test func sendFailureIsReportedWithoutStartingWork() {
        let transport = FakeTransport()
        transport.failSends = true
        let model = readyModel(transport)
        model.prompt = "a"
        model.make()
        #expect(model.state == .ready)
        #expect(model.lastError?.hasPrefix("could not reach the engine") == true)
    }

    @Test func startFollowsTheStream() async {
        let transport = FakeTransport()
        let model = AppModel(transport: transport, now: { t0 })
        model.start()
        transport.continuation.yield(.event(.ready(loadSeconds: 2)))
        transport.continuation.yield(.exited("exit 0"))
        for _ in 0..<100 where model.state != .failed("The engine stopped (exit 0).") {
            try? await Task.sleep(for: .milliseconds(10))
        }
        #expect(model.state == .failed("The engine stopped (exit 0)."))
    }

    @Test func idleEngineIsStoppedAndItsExitIsNotAFailure() async {
        let transport = FakeTransport()
        let gate = IdleGate()
        let model = sleepyModel(transport, gate)
        await gate.open(model)
        #expect(transport.stops == 1)
        #expect(model.state == .asleep)
        model.handle(.exited("exit 15"))
        #expect(model.state == .asleep)
        model.prompt = "b"
        #expect(model.canMake)
    }

    @Test func aRequestKeepsTheEngineAwake() async {
        let transport = FakeTransport()
        let gate = IdleGate()
        let model = sleepyModel(transport, gate)
        model.prompt = "a"
        model.make()
        await gate.open(model)
        #expect(transport.stops == 0)
        #expect(model.state == .working(.lyrics, since: t0))
        model.handle(.event(.done(id: "req-1", song: song("a"))))
        await gate.open(model)
        #expect(transport.stops == 1)
        #expect(model.state == .asleep)
    }

    @Test func aRequestWakesTheEngineAndIsSentWhenReady() async {
        let transport = FakeTransport()
        let gate = IdleGate()
        let model = sleepyModel(transport, gate)
        await gate.open(model)
        model.handle(.exited("exit 15"))
        model.prompt = "b"
        model.make()
        #expect(transport.starts == 2)
        #expect(model.state == .starting(since: t0))
        #expect(transport.sent.isEmpty)
        model.handle(.event(.ready(loadSeconds: 30)))
        #expect(transport.sent == [["id": "req-1", "cmd": "generate", "prompt": "b"]])
        #expect(model.state == .working(.lyrics, since: t0))
    }

    @Test func aRequestWhileStoppingWaitsForTheExit() async {
        let transport = FakeTransport()
        let gate = IdleGate()
        let model = sleepyModel(transport, gate)
        await gate.open(model)
        model.prompt = "b"
        model.make()
        #expect(transport.starts == 1)
        #expect(model.state == .starting(since: t0))
        model.handle(.exited("exit 15"))
        #expect(transport.starts == 2)
        model.handle(.event(.ready(loadSeconds: 30)))
        #expect(transport.sent == [["id": "req-1", "cmd": "generate", "prompt": "b"]])
    }

    @Test func withoutIdlePeriodTheEngineStaysUp() async {
        let transport = FakeTransport()
        let gate = IdleGate()
        let model = AppModel(transport: transport, now: { t0 }, sleeper: gate.sleeper)
        model.handle(.event(.ready(loadSeconds: 30)))
        await gate.open(model)
        #expect(transport.stops == 0)
        #expect(model.state == .ready)
    }

    @Test func aTimerFromBeforeARequestDoesNotStopTheEngineAfterIt() {
        let transport = FakeTransport()
        let model = sleepyModel(transport, IdleGate())
        let before = model.activity
        model.prompt = "a"
        model.make()
        model.handle(.event(.done(id: "req-1", song: song("a"))))
        model.idleElapsed(before)
        #expect(transport.stops == 0)
        #expect(model.state == .ready)
        model.idleElapsed(model.activity)
        #expect(transport.stops == 1)
    }
}
