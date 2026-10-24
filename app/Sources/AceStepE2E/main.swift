// Drives the real engine through AppModel, as the app does, and checks the result.
// Usage on a machine with the development install: swift run AceStepE2E [prompt]
// Exit status 0 only if a song was made, written to disk and found by SongStore.
// With ACESTEP_E2E_CYCLES=N it then waits for the engine to sleep (ACESTEP_IDLE_SECONDS)
// and wakes it with a new song N times; the second wake is requested while it stops.
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
    let model = AppModel(transport: EngineProcess(config: config), idleSeconds: config.idleSeconds)
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
    var ok = check(model.lastError == nil, "no request error \(model.lastError ?? "")")
    ok = check(phases.first == .lyrics && phases.contains(.rendering), "phases seen \(phases.map(\.rawValue))") && ok
    guard let song = model.songs.first, let audio = song.audioURL else {
        return check(false, "a song arrived")
    }
    let size = (try? FileManager.default.attributesOfItem(atPath: audio.path)[.size] as? Int) ?? 0
    ok = check(size > 100_000, "audio on disk, \(size) bytes, made in \(Int(finished.timeIntervalSince(loaded))) s") && ok
    ok = check(SongStore.load(from: config.songsDir).map(\.audio) == [song.audio], "history lists it") && ok
    print("song: \(song.bpm.map(String.init) ?? "?") BPM, \(song.keyscale ?? "?"), \(song.language ?? "?"), \(audio.path)")
    let cycles = Int(ProcessInfo.processInfo.environment["ACESTEP_E2E_CYCLES"] ?? "") ?? 0
    for cycle in 1...max(cycles, 1) where cycles > 0 {
        ok = await sleepAndWake(model, cycle: cycle, idle: config.idleSeconds) && ok
    }
    model.stop()
    return ok
}

/// Waits for the idle stop, then asks for a song and times the wake.
@MainActor
func sleepAndWake(_ model: AppModel, cycle: Int, idle: Double) async -> Bool {
    let readyAt = Date()
    await waitFor(idle + 60) { model.state == .asleep }
    let asleepAt = Date()
    guard check(model.state == .asleep, "cycle \(cycle): asleep after \(Int(asleepAt.timeIntervalSince(readyAt))) s idle") else {
        return false
    }
    print("E2E \(Date().timeIntervalSince1970) asleep cycle \(cycle)")
    // Cycle 2 asks while the old process may still be exiting; the others after it is gone.
    if cycle != 2 { try? await Task.sleep(for: .seconds(15)) }
    let count = model.songs.count
    let asked = Date()
    print("E2E \(asked.timeIntervalSince1970) wake cycle \(cycle)")
    model.prompt = "a short instrumental lo-fi beat with soft piano, take \(cycle)"
    model.make()
    var loadedAt: Date?
    await waitFor(400) {
        if loadedAt == nil, case .working = model.state { loadedAt = Date() }
        return model.state == .ready && model.songs.count == count + 1
    }
    let done = Date()
    let load = loadedAt.map { Int($0.timeIntervalSince(asked)) } ?? -1
    return check(model.songs.count == count + 1 && model.lastError == nil,
                 "cycle \(cycle): woke, engine ready in \(load) s, song done in \(Int(done.timeIntervalSince(asked))) s")
}

exit(await run() ? 0 : 1)
