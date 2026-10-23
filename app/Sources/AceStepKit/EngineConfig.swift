import Foundation

/// Where the engine lives and where songs go.
///
/// Until the engine is bundled inside the app, it runs from a development install:
/// the upstream venv's Python, this repo's `engine/`, and the upstream checkout with
/// its `checkpoints/`. Defaults match the Mac mini layout; `ACESTEP_PYTHON`,
/// `ACESTEP_ENGINE` and `ACESTEP_PROJECT` override them.
public struct EngineConfig: Sendable, Equatable {
    public var python: URL
    public var engineDir: URL
    public var project: URL
    public var songsDir: URL
    public var logFile: URL
    /// Engine preset: "mini" or "laptop" (engine/acestep_mlx/recipe.py).
    public var preset: String

    public init(python: URL, engineDir: URL, project: URL, songsDir: URL, logFile: URL,
                preset: String = "mini") {
        self.preset = preset
        self.python = python
        self.engineDir = engineDir
        self.project = project
        self.songsDir = songsDir
        self.logFile = logFile
    }

    /// Defaults under `home`, overridden by the environment. The preset is "laptop" on a
    /// machine with an internal battery and "mini" otherwise, unless ACESTEP_PRESET is set.
    public static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        hasBattery: Bool = PowerSource.hasInternalBattery()
    ) -> EngineConfig {
        func path(_ key: String, _ fallback: String) -> URL {
            URL(fileURLWithPath: environment[key] ?? home.appendingPathComponent(fallback).path)
        }
        let project = path("ACESTEP_PROJECT", "acestep-f0/ACE-Step-1.5")
        return EngineConfig(
            python: path("ACESTEP_PYTHON", "acestep-f0/ACE-Step-1.5/.venv/bin/python"),
            engineDir: path("ACESTEP_ENGINE", "acestep-mlx/engine"),
            project: project,
            songsDir: home.appendingPathComponent("Music/ACE-Step"),
            logFile: home.appendingPathComponent("Library/Logs/ACE-Step/engine.log"),
            preset: environment["ACESTEP_PRESET"] ?? (hasBattery ? "laptop" : "mini")
        )
    }

    /// Paths the engine needs that do not exist, for the "not installed" screen.
    public func missingPaths(fileManager: FileManager = .default) -> [String] {
        let required = [
            python,
            engineDir.appendingPathComponent("acestep_mlx/engine_service.py"),
            project.appendingPathComponent("checkpoints"),
        ]
        return required.map(\.path).filter { !fileManager.fileExists(atPath: $0) }
    }

    /// Command line of the engine process.
    public var arguments: [String] {
        ["-m", "acestep_mlx.engine_service", "--project", project.path, "--out", songsDir.path,
         "--preset", preset]
    }

    /// Environment of the engine process: the inherited one plus the engine on PYTHONPATH.
    public func environment(base: [String: String]) -> [String: String] {
        var env = base
        env["PYTHONPATH"] = engineDir.path
        env["PYTHONUNBUFFERED"] = "1"
        return env
    }
}
