import AceStepKit
import AppKit
import UserNotifications

/// Tells the user what happened while the window was in the background: the Dock badge
/// counts songs made since, the Dock icon bounces once, and a notification names the song.
/// During the first-start download the badge shows the percentage instead.
@MainActor
final class Attention {
    private var unseen = 0
    private var observer: NSObjectProtocol?

    init() {
        observer = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.seen() }
        }
    }

    func songReady(_ song: Song) {
        guard !NSApp.isActive else { return }
        unseen += 1
        NSApp.dockTile.badgeLabel = "\(unseen)"
        NSApp.requestUserAttention(.informationalRequest)
        notify(song)
    }

    func engine(_ state: EngineState) {
        if case let .downloading(done, total) = state, total > 0 {
            NSApp.dockTile.badgeLabel = "\(Int(Double(done) / Double(total) * 100))%"
        } else if unseen == 0 {
            NSApp.dockTile.badgeLabel = nil
        }
    }

    private func seen() {
        unseen = 0
        NSApp.dockTile.badgeLabel = nil
    }

    /// Asks for permission the first time a song finishes in the background, not at launch.
    private func notify(_ song: Song) {
        // UserNotifications needs a bundle; `swift run` has none.
        guard Bundle.main.bundleIdentifier != nil else { return }
        let (id, prompt) = (song.id, song.prompt)
        Task {
            let center = UNUserNotificationCenter.current()
            guard (try? await center.requestAuthorization(options: [.alert, .sound])) == true else { return }
            let content = UNMutableNotificationContent()
            content.title = "Song ready"
            content.body = prompt
            try? await center.add(UNNotificationRequest(identifier: id, content: content, trigger: nil))
        }
    }
}
