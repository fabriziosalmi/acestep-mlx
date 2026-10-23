import AceStepKit
import AppKit
import SwiftUI

@main
struct AceStepApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model = AceStepApp.makeModel()
    @State private var player = Player()

    var body: some Scene {
        Window("ACE-Step", id: "main") {
            ContentView(model: model, player: player)
                .onAppear {
                    delegate.model = model
                    model.start()
                }
        }
        .windowResizability(.contentMinSize)
    }

    @MainActor
    private static func makeModel() -> AppModel {
        let config = EngineConfig.resolve()
        let missing = config.missingPaths()
        return AppModel(
            transport: missing.isEmpty ? EngineProcess(config: config) : nil,
            missing: missing,
            songs: SongStore.load(from: config.songsDir)
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
