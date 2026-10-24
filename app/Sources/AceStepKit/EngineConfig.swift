import Foundation

/// Where the engine lives and where songs go.
///
/// The app carries the engine in `Contents/Resources/runtime` (scripts/bundle.sh): a
/// standalone Python with upstream's locked dependencies, upstream's `acestep` package and
/// this repo's `engine/`. Model weights live outside the app, in
/// `~/Library/Application Support/ACE-Step/checkpoints`, downloaded by the engine on the
/// first start. Without a runtime (`swift run`), the engine runs from the development
/// install: the upstream venv's Python, this repo's `engine/`, and the upstream checkout with
/// its `checkpoints/`. `ACESTEP_RUNTIME` points at a runtime folder (an app's, for checks);
/// `ACESTEP_PYTHON`, `ACESTEP_ENGINE`, `ACESTEP_PROJECT` and `ACESTEP_SONGS` override single paths.
public struct EngineConfig: Sendable, Equatable {
    public var python: URL
    public var engineDir: URL
    /// Folder holding upstream's `acestep` package.
    public var upstreamDir: URL
    /// Working folder of the engine; model weights are in its `checkpoints/`.
    public var project: URL
    public var songsDir: URL
    public var logFile: URL
    /// Where Python writes bytecode, so a signed runtime is never written to; nil: next to the sources.
    public var pycachePrefix: URL?
    /// Engine preset: "mini" or "laptop" (engine/acestep_mlx/recipe.py).
    public var preset: String
    /// Seconds ready with nothing to do before the engine is stopped to free its memory
    /// (about 9 GB with the LM loaded); the next song then waits about 30 s for the load.
    public var idleSeconds: Double

    /// Idle periods per preset: the laptop gives memory back soon, the mini keeps the
    /// models warm through a working session.
    public static let idleSecondsByPreset: [String: Double] = ["laptop": 120, "mini": 1200]

    /// The app's own runtime folder, when it has one.
    public static var bundledRuntime: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("runtime")
    }

    public init(python: URL, engineDir: URL, upstreamDir: URL? = nil, project: URL, songsDir: URL,
                logFile: URL, pycachePrefix: URL? = nil, preset: String = "mini", idleSeconds: Double? = nil) {
        self.preset = preset
        self.idleSeconds = idleSeconds ?? Self.idleSecondsByPreset[preset] ?? 1200
        self.python = python
        self.engineDir = engineDir
        self.upstreamDir = upstreamDir ?? project
        self.project = project
        self.songsDir = songsDir
        self.logFile = logFile
        self.pycachePrefix = pycachePrefix
    }

    /// Defaults under `home`, overridden by the environment. The runtime is
    /// `ACESTEP_RUNTIME` when set (even if it does not exist, so a wrong path shows as
    /// missing instead of silently using the development install), else `runtime` when it
    /// holds a Python, else none. The preset is "laptop" on a machine with an internal
    /// battery and "mini" otherwise, unless ACESTEP_PRESET is set; ACESTEP_IDLE_SECONDS
    /// overrides the preset's idle period.
    public static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        hasBattery: Bool = PowerSource.hasInternalBattery(),
        runtime: URL? = bundledRuntime
    ) -> EngineConfig {
        func path(_ key: String, _ fallback: URL) -> URL {
            environment[key].map { URL(fileURLWithPath: $0) } ?? fallback
        }
        let found = runtime.flatMap {
            FileManager.default.fileExists(atPath: pythonIn($0).path) ? $0 : nil
        }
        let runtime = environment["ACESTEP_RUNTIME"].map { URL(fileURLWithPath: $0) } ?? found
        let idle = environment["ACESTEP_IDLE_SECONDS"].flatMap(Double.init).flatMap { $0 > 0 ? $0 : nil }
        let preset = environment["ACESTEP_PRESET"] ?? (hasBattery ? "laptop" : "mini")
        let songs = path("ACESTEP_SONGS", home.appendingPathComponent("Music/ACE-Step"))
        let log = home.appendingPathComponent("Library/Logs/ACE-Step/engine.log")
        if let runtime {
            return EngineConfig(
                python: path("ACESTEP_PYTHON", pythonIn(runtime)),
                engineDir: path("ACESTEP_ENGINE", runtime.appendingPathComponent("engine")),
                upstreamDir: runtime.appendingPathComponent("upstream"),
                project: path("ACESTEP_PROJECT", home.appendingPathComponent("Library/Application Support/ACE-Step")),
                songsDir: songs, logFile: log,
                pycachePrefix: home.appendingPathComponent("Library/Caches/ACE-Step/pycache"),
                preset: preset, idleSeconds: idle
            )
        }
        let project = path("ACESTEP_PROJECT", home.appendingPathComponent("acestep-f0/ACE-Step-1.5"))
        return EngineConfig(
            python: path("ACESTEP_PYTHON", home.appendingPathComponent("acestep-f0/ACE-Step-1.5/.venv/bin/python")),
            engineDir: path("ACESTEP_ENGINE", home.appendingPathComponent("acestep-mlx/engine")),
            project: project, songsDir: songs, logFile: log, preset: preset, idleSeconds: idle
        )
    }

    private static func pythonIn(_ runtime: URL) -> URL {
        runtime.appendingPathComponent("python/bin/python3")
    }

    /// Paths the engine needs that do not exist, for the "not installed" screen. Model
    /// weights are not among them: the engine downloads what is missing.
    public func missingPaths(fileManager: FileManager = .default) -> [String] {
        let required = [
            python,
            engineDir.appendingPathComponent("acestep_mlx/engine_service.py"),
            upstreamDir.appendingPathComponent("acestep/__init__.py"),
        ]
        return required.map(\.path).filter { !fileManager.fileExists(atPath: $0) }
    }

    /// Command line of the engine process.
    public var arguments: [String] {
        ["-m", "acestep_mlx.engine_service", "--project", project.path, "--out", songsDir.path,
         "--preset", preset]
    }

    /// Environment of the engine process: the inherited one with the engine and upstream on
    /// PYTHONPATH, and nothing from the user's own Python setup.
    public func environment(base: [String: String]) -> [String: String] {
        var env = base
        env["PYTHONPATH"] = [engineDir.path, upstreamDir.path].joined(separator: ":")
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONHOME"] = nil
        env["PYTHONUNBUFFERED"] = "1"
        if let pycachePrefix { env["PYTHONPYCACHEPREFIX"] = pycachePrefix.path }
        return env
    }
}
