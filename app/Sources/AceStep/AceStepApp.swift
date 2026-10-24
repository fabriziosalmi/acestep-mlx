import AceStepKit
import AppKit
import SwiftUI

@main
struct AceStepApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model = AceStepApp.makeModel()
    @State private var player = Player()
    @State private var attention = Attention()
    private static let config = EngineConfig.resolve()

    var body: some Scene {
        Window("ACE-Step", id: "main") {
            ContentView(model: model, player: player, attention: attention)
                .onAppear {
                    delegate.model = model
                    model.start()
                }
        }
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Show Songs Folder") { Self.reveal(Self.config.songsDir, folder: true) }
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                Button("Show Engine Log") { Self.reveal(Self.config.logFile, folder: false) }
                    .keyboardShortcut("l", modifiers: [.command, .shift])
            }
        }
    }

    /// Opens the songs folder in Finder (creating it if no song exists yet), or the log in
    /// its default app (Console).
    private static func reveal(_ url: URL, folder: Bool) {
        if folder {
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        } else if !FileManager.default.fileExists(atPath: url.path) {
            return NSSound.beep()
        }
        NSWorkspace.shared.open(url)
    }

    @MainActor
    private static func makeModel() -> AppModel {
        let config = Self.config
        let missing = config.missingPaths()
        return AppModel(
            transport: missing.isEmpty ? EngineProcess(config: config) : nil,
            missing: missing,
            songs: SongStore.load(from: config.songsDir),
            idleSeconds: config.idleSeconds
        )
    }
}

/// Makes the bare executable behave like an app and stops the engine on quit.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        model?.stop()
    }
}
