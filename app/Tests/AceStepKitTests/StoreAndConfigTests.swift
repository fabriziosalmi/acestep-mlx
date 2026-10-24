import Foundation
import Testing

@testable import AceStepKit

func temporaryDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

@Suite struct SongStoreTests {
    func writeSong(_ root: URL, folder: String, created: String, withAudio: Bool, json: String? = nil) throws {
        let dir = root.appendingPathComponent(folder)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let record = json ?? """
        {"prompt": "\(folder)", "caption": "c", "lyrics": "l", "bpm": null, "audio": "a.flac", "created": "\(created)"}
        """
        try Data(record.utf8).write(to: dir.appendingPathComponent("song.json"))
        if withAudio { try Data([0]).write(to: dir.appendingPathComponent("a.flac")) }
    }

    @Test func loadsValidSongsNewestFirst() throws {
        let root = try temporaryDirectory()
        try writeSong(root, folder: "old", created: "2026-10-22T10:00:00+00:00", withAudio: true)
        try writeSong(root, folder: "new", created: "2026-10-23T10:00:00+00:00", withAudio: true)
        try writeSong(root, folder: "no-audio", created: "2026-10-24T10:00:00+00:00", withAudio: false)
        try writeSong(root, folder: "broken", created: "", withAudio: true, json: "{")
        let songs = SongStore.load(from: root)
        #expect(songs.map(\.prompt) == ["new", "old"])
        let expected = root.appendingPathComponent("new").resolvingSymlinksInPath().path
        #expect(songs.first?.dir.map { URL(fileURLWithPath: $0).resolvingSymlinksInPath().path } == expected)
        #expect(songs.first?.bpm == nil)
    }

    @Test func missingDirectoryIsEmpty() {
        #expect(SongStore.load(from: URL(fileURLWithPath: "/nonexistent/acestep")).isEmpty)
    }
}

@Suite struct EngineConfigTests {
    @Test func defaultsFollowHomeAndEnvironmentOverrides() {
        let home = URL(fileURLWithPath: "/Users/someone")
        let plain = EngineConfig.resolve(environment: [:], home: home, hasBattery: false, runtime: nil)
        #expect(plain.preset == "mini")
        #expect(EngineConfig.resolve(environment: [:], home: home, hasBattery: true).preset == "laptop")
        #expect(plain.idleSeconds == 1200)
        #expect(EngineConfig.resolve(environment: [:], home: home, hasBattery: true).idleSeconds == 120)
        #expect(EngineConfig.resolve(environment: ["ACESTEP_IDLE_SECONDS": "20"], home: home,
                                     hasBattery: true).idleSeconds == 20)
        for bad in ["0", "-5", "soon"] {
            #expect(EngineConfig.resolve(environment: ["ACESTEP_IDLE_SECONDS": bad], home: home,
                                         hasBattery: false).idleSeconds == 1200)
        }
        #expect(EngineConfig.resolve(environment: ["ACESTEP_PRESET": "mini"], home: home,
                                     hasBattery: true).preset == "mini")
        #expect(plain.python.path == "/Users/someone/acestep-f0/ACE-Step-1.5/.venv/bin/python")
        #expect(plain.songsDir.path == "/Users/someone/Music/ACE-Step")
        let custom = EngineConfig.resolve(environment: ["ACESTEP_PROJECT": "/opt/ace"], home: home,
                                          hasBattery: false, runtime: nil)
        #expect(custom.project.path == "/opt/ace")
        #expect(custom.arguments == ["-m", "acestep_mlx.engine_service", "--project", "/opt/ace",
                                     "--out", "/Users/someone/Music/ACE-Step", "--preset", "mini"])
        let env = custom.environment(base: ["HOME": "/h", "PYTHONHOME": "/elsewhere"])
        #expect(env["PYTHONPATH"] == "/Users/someone/acestep-mlx/engine:/opt/ace")
        #expect(env["PYTHONHOME"] == nil)
        #expect(env["PYTHONNOUSERSITE"] == "1")
        #expect(env["PYTHONPYCACHEPREFIX"] == nil)
        #expect(env["HOME"] == "/h")
    }

    @Test func anAppRuntimeIsUsedWhenItHoldsAPython() throws {
        let home = URL(fileURLWithPath: "/Users/someone")
        let runtime = try temporaryDirectory()
        let empty = EngineConfig.resolve(environment: [:], home: home, hasBattery: false, runtime: runtime)
        #expect(empty.python.path == "/Users/someone/acestep-f0/ACE-Step-1.5/.venv/bin/python")
        let bin = runtime.appendingPathComponent("python/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: bin.appendingPathComponent("python3").path, contents: nil)
        let app = EngineConfig.resolve(environment: [:], home: home, hasBattery: false, runtime: runtime)
        #expect(app.python == runtime.appendingPathComponent("python/bin/python3"))
        #expect(app.engineDir == runtime.appendingPathComponent("engine"))
        #expect(app.upstreamDir == runtime.appendingPathComponent("upstream"))
        #expect(app.project.path == "/Users/someone/Library/Application Support/ACE-Step")
        let env = app.environment(base: [:])
        #expect(env["PYTHONPATH"] == "\(runtime.path)/engine:\(runtime.path)/upstream")
        #expect(env["PYTHONPYCACHEPREFIX"] == "/Users/someone/Library/Caches/ACE-Step/pycache")
    }

    @Test func aRuntimeNamedInTheEnvironmentIsUsedEvenIfMissing() {
        let home = URL(fileURLWithPath: "/Users/someone")
        let config = EngineConfig.resolve(environment: ["ACESTEP_RUNTIME": "/nowhere/runtime"], home: home,
                                          hasBattery: false, runtime: nil)
        #expect(config.python.path == "/nowhere/runtime/python/bin/python3")
        #expect(config.missingPaths() == ["/nowhere/runtime/python/bin/python3",
                                          "/nowhere/runtime/engine/acestep_mlx/engine_service.py",
                                          "/nowhere/runtime/upstream/acestep/__init__.py"])
    }

    @Test func reportsMissingPathsButNotModelWeights() throws {
        let root = try temporaryDirectory()
        let config = EngineConfig.resolve(environment: [:], home: root, hasBattery: false, runtime: nil)
        #expect(config.missingPaths().count == 3)
        let upstream = root.appendingPathComponent("acestep-f0/ACE-Step-1.5/acestep")
        try FileManager.default.createDirectory(at: upstream, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: upstream.appendingPathComponent("__init__.py").path, contents: nil)
        #expect(config.missingPaths().count == 2)
        #expect(!config.missingPaths().contains { $0.hasSuffix("checkpoints") })
    }
}
