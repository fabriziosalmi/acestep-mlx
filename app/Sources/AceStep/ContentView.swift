import AceStepKit
import AppKit
import SwiftUI

/// The whole app: describe a song, make it, play what you made.
struct ContentView: View {
    @Bindable var model: AppModel
    let player: Player
    @FocusState private var promptFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Describe a song")
                .font(.title2.weight(.semibold))
            promptEditor
            HStack(alignment: .center, spacing: 12) {
                StatusLine(state: model.state, lastError: model.lastError ?? player.failure)
                Spacer(minLength: 8)
                Button("Make song") { model.make() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!model.canMake)
            }
            Divider()
            songList
        }
        .padding(20)
        .frame(minWidth: 520, minHeight: 560)
        .onAppear { promptFocused = true }
    }

    private var promptEditor: some View {
        TextEditor(text: $model.prompt)
            .font(.title3)
            .scrollContentBackground(.hidden)
            .focused($promptFocused)
            .padding(8)
            .frame(height: 96)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
            .overlay(alignment: .topLeading) {
                if model.prompt.isEmpty {
                    Text("A warm Italian folk ballad with acoustic guitar…")
                        .font(.title3)
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 13)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
    }

    @ViewBuilder
    private var songList: some View {
        if model.songs.isEmpty {
            Text("Your songs will appear here.")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List(model.songs) { song in
                SongRow(song: song, playing: player.playingID == song.id,
                        canMakeAnother: model.canRequest,
                        toggle: { player.toggle(song) },
                        makeAnother: { model.makeAnother(like: song) })
            }
            .listStyle(.inset)
        }
    }
}

/// One line telling the truth about the engine: loading, working on a phase, or broken.
struct StatusLine: View {
    let state: EngineState
    let lastError: String?

    var body: some View {
        switch state {
        case .notInstalled(let paths):
            Label("Engine not installed: missing \(paths.joined(separator: ", "))",
                  systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
                .lineLimit(3)
        case .starting(let since):
            busy("Loading models", since: since)
        case .working(let phase, let since):
            busy(Self.label(phase), since: since)
        case .asleep:
            Text("Ready, models unloaded to free memory")
                .foregroundStyle(.secondary)
                .help("The next song loads them again first, about 30 seconds.")
        case .ready:
            if let lastError {
                Label(lastError, systemImage: "exclamationmark.circle")
                    .foregroundStyle(.red)
                    .lineLimit(2)
            } else {
                Text("Ready").foregroundStyle(.secondary)
            }
        case .failed(let message):
            Label(message, systemImage: "xmark.octagon")
                .foregroundStyle(.red)
                .lineLimit(3)
        }
    }

    private func busy(_ text: String, since: Date) -> some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            TimelineView(.periodic(from: since, by: 1)) { context in
                Text("\(text)… \(Int(context.date.timeIntervalSince(since))) s")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        }
    }

    static func label(_ phase: Phase) -> String {
        switch phase {
        case .lyrics: "Writing lyrics"
        case .composing: "Composing"
        case .rendering: "Rendering"
        }
    }
}

/// A made song: play button, prompt, musical facts, another version; lyrics on demand.
struct SongRow: View {
    let song: Song
    let playing: Bool
    let canMakeAnother: Bool
    let toggle: () -> Void
    let makeAnother: () -> Void
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Button(action: toggle) {
                    Image(systemName: playing ? "stop.circle.fill" : "play.circle.fill")
                        .font(.system(size: 26))
                }
                .buttonStyle(.plain)
                .help(playing ? "Stop" : "Play")
                VStack(alignment: .leading, spacing: 2) {
                    Text(song.prompt).font(.headline).lineLimit(1)
                    Text(facts).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button(action: makeAnother) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                }
                .buttonStyle(.plain)
                .disabled(!canMakeAnother)
                .help("Another like this")
                Button { expanded.toggle() } label: {
                    Image(systemName: expanded ? "chevron.up" : "text.quote")
                }
                .buttonStyle(.plain)
                .help("Lyrics")
            }
            if expanded {
                Text(song.caption).font(.callout).foregroundStyle(.secondary)
                Text(song.lyrics).font(.callout).textSelection(.enabled)
            }
        }
        .padding(.vertical, 4)
        .contextMenu {
            if let url = song.audioURL {
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            }
            Button("Copy lyrics") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(song.lyrics, forType: .string)
            }
        }
    }

    private var facts: String {
        var parts: [String] = []
        if let bpm = song.bpm { parts.append("\(bpm) BPM") }
        if let key = song.keyscale, !key.isEmpty { parts.append(key) }
        if song.instrumental == true {
            parts.append("instrumental")
        } else if let lang = song.language, !lang.isEmpty, lang != "unknown" {
            parts.append(lang)
        }
        if let seconds = song.duration { parts.append(String(format: "%d:%02d", Int(seconds) / 60, Int(seconds) % 60)) }
        return parts.joined(separator: " · ")
    }
}
