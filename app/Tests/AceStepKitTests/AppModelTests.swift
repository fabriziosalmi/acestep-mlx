import Foundation
import Testing

@testable import AceStepKit

/// Records sent lines; can be told to fail sends.
final class FakeTransport: EngineTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var _sent: [Data] = []
    var failSends = false
    let stream: AsyncStream<EngineOutput>
    let continuation: AsyncStream<EngineOutput>.Continuation

    init() {
        (stream, continuation) = AsyncStream.makeStream(of: EngineOutput.self)
    }

    var sent: [[String: String]] {
        lock.lock()
        defer { lock.unlock() }
        return _sent.compactMap { try? JSONSerialization.jsonObject(with: $0.dropLast()) as? [String: String] }
    }

    func start() throws -> AsyncStream<EngineOutput> { stream }

    func send(_ line: Data) throws {
        if failSends { throw CocoaError(.fileWriteUnknown) }
        lock.lock()
        _sent.append(line)
        lock.unlock()
    }

    func stop() { continuation.finish() }
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
@Suite struct AppModelTests {
    @Test func startsLoadingAndBecomesReady() {
        let model = AppModel(transport: FakeTransport(), now: { t0 })
        #expect(model.state == .starting(since: t0))
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
}
