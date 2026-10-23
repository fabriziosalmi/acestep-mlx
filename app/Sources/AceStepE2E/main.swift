// Drives the real engine through AppModel, as the app does, and checks the result.
// Usage on a machine with the development install: swift run AceStepE2E [prompt]
// Exit status 0 only if a song was made, written to disk and found by SongStore.
import AceStepKit
import Foundation

@MainActor
func waitFor(_ seconds: Double, _ done: () -> Bool) async {
    let deadline = Date().addingTimeInterval(seconds)
    while !done(), Date() < deadline {
        try? await Task.sleep(for: .milliseconds(200))
    }
}

@MainActor
func check(_ ok: Bool, _ what: String) -> Bool {
    print("\(ok ? "PASS" : "FAIL") \(what)")
    return ok
}

@MainActor
func run() async -> Bool {
    var config = EngineConfig.resolve()
    guard check(config.missingPaths().isEmpty, "development install present \(config.missingPaths())") else {
        return false
    }
    config.songsDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("acestep-e2e-\(UUID().uuidString)")
    let model = AppModel(transport: EngineProcess(config: config))
    let started = Date()
    model.start()
    await waitFor(180) { model.state == .ready }
    let loaded = Date()
    guard check(model.state == .ready, "engine ready in \(Int(loaded.timeIntervalSince(started))) s: \(model.state)") else {
        return false
    }
    let args = CommandLine.arguments.dropFirst()
    model.prompt = args.isEmpty ? "a short instrumental lo-fi beat with soft piano" : args.joined(separator: " ")
    model.make()
    var phases: [Phase] = []
    await waitFor(300) {
        if case let .working(phase, _) = model.state, phases.last != phase { phases.append(phase) }
        return model.state == .ready
    }
    let finished = Date()
    model.stop()
    var ok = check(model.lastError == nil, "no request error \(model.lastError ?? "")")
    ok = check(phases.first == .lyrics && phases.contains(.rendering), "phases seen \(phases.map(\.rawValue))") && ok
    guard let song = model.songs.first, let audio = song.audioURL else {
        return check(false, "a song arrived")
    }
    let size = (try? FileManager.default.attributesOfItem(atPath: audio.path)[.size] as? Int) ?? 0
    ok = check(size > 100_000, "audio on disk, \(size) bytes, made in \(Int(finished.timeIntervalSince(loaded))) s") && ok
    ok = check(SongStore.load(from: config.songsDir).map(\.audio) == [song.audio], "history lists it") && ok
    print("song: \(song.bpm.map(String.init) ?? "?") BPM, \(song.keyscale ?? "?"), \(song.language ?? "?"), \(audio.path)")
    return ok
}

exit(await run() ? 0 : 1)
