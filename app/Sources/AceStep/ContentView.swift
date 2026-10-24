import AceStepKit
import AppKit
import SwiftUI

/// The whole app: describe a song, make it, play what you made.
struct ContentView: View {
    @Bindable var model: AppModel
    let player: Player
    let attention: Attention
    @FocusState private var promptFocused: Bool
    @State private var selection: Song.ID?
    @State private var notice: String?
    @State private var lyricsShown = false

    /// Prompts the listening tests found reliable; a click puts one in the editor.
    static let examples = [
        "A warm Italian folk ballad with acoustic guitar and a female voice",
        "Lo-fi hip hop beat, 85 bpm, dusty vinyl, mellow piano",
        "Orchestral film score, epic strings and brass, D minor",
        "Drum and bass, 174 bpm, reese bass, amen breaks",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Describe a song")
                .font(.title2.weight(.semibold))
            promptEditor
            lyricsSection
            HStack(alignment: .center, spacing: 12) {
                StatusLine(state: model.state, lastError: model.lastError ?? player.failure ?? notice,
                           retry: model.canRetry ? { model.retry() } : nil)
                Spacer(minLength: 8)
                Button("Make song") { model.make() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!model.canMake)
                    .help("Make song (⌘↩)")
            }
            Divider()
            songList
        }
        .padding(20)
        .frame(minWidth: 520, minHeight: 560)
        .onAppear { promptFocused = true }
        // A new song goes on top; a removal shrinks the list and must not count as one.
        .onChange(of: model.songs.count) { old, new in
            if new > old, let song = model.songs.first {
                selection = song.id
                attention.songReady(song)
            }
        }
        .onChange(of: model.state) { _, state in attention.engine(state) }
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
                    Text("Genre, mood, instruments, a voice or none, a tempo…")
                        .font(.title3)
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 13)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
            .accessibilityLabel("Song description")
    }

    /// Optional own lyrics: closed by default; the button says when some are set.
    @ViewBuilder
    private var lyricsSection: some View {
        let lines = model.lyrics.split(whereSeparator: \.isNewline)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count
        Button {
            withAnimation(.easeInOut(duration: 0.15)) { lyricsShown.toggle() }
        } label: {
            Label(lines == 0 ? "Add your own lyrics" : "Your lyrics, \(lines) line\(lines == 1 ? "" : "s")",
                  systemImage: lyricsShown ? "chevron.down" : "chevron.right")
        }
        .buttonStyle(.link)
        .help("Leave empty and the lyrics are written for you")
        if lyricsShown {
            VStack(alignment: .leading, spacing: 6) {
                TextEditor(text: $model.lyrics)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .frame(height: 140)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                    .overlay(alignment: .topLeading) {
                        if model.lyrics.isEmpty {
                            Text("[Verse]\nYour words, one line each\n[Chorus]\n…")
                                .foregroundStyle(.tertiary)
                                .padding(.horizontal, 13)
                                .padding(.vertical, 8)
                                .allowsHitTesting(false)
                        }
                    }
                    .accessibilityLabel("Your lyrics")
                Text("Sung as written, always with a voice. Songs are 30 seconds, so a long text is cut after its first lines.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var songList: some View {
        if model.songs.isEmpty {
            emptyHistory
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollViewReader { proxy in
                List(model.songs, selection: $selection) { song in
                    SongRow(song: song, player: player, canMakeAnother: model.canRequest,
                            makeAnother: { model.makeAnother(like: song) },
                            trash: { moveToTrash(song) })
                        .tag(song.id)
                        .id(song.id)
                }
                .listStyle(.inset)
                // A new song is selected; keep it fully in view at the top.
                .onChange(of: selection) { _, id in
                    if let id, id == model.songs.first?.id { proxy.scrollTo(id, anchor: .top) }
                }
            .onKeyPress(.space) {
                guard let id = selection, let song = model.songs.first(where: { $0.id == id }) else {
                    return .ignored
                }
                player.toggle(song)
                return .handled
            }
            }
        }
    }

    @ViewBuilder
    private var emptyHistory: some View {
        if case .downloading = model.state {
            VStack(spacing: 8) {
                Image(systemName: "arrow.down.circle")
                    .font(.system(size: 34))
                    .foregroundStyle(.secondary)
                Text("First start: downloading the models, 7.7 GB, once.")
                    .font(.headline)
                Text("They stay in ~/Library/Application Support/ACE-Step; later starts take about 30 seconds.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 30)
        } else {
            VStack(spacing: 12) {
                Text("Your songs will appear here. Try one of these:")
                    .foregroundStyle(.secondary)
                ForEach(Self.examples, id: \.self) { example in
                    Button(example) {
                        model.prompt = example
                        promptFocused = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                }
            }
        }
    }

    private func moveToTrash(_ song: Song) {
        guard let dir = song.dir else { return }
        if player.playingID == song.id { player.stop() }
        NSWorkspace.shared.recycle([URL(fileURLWithPath: dir)]) { _, error in
            Task { @MainActor in
                if let error {
                    notice = "Could not move the song to the Trash: \(error.localizedDescription)"
                } else {
                    notice = nil
                    model.remove(song)
                }
            }
        }
    }
}

/// One line telling the truth about the engine: loading, working on a phase, or broken.
struct StatusLine: View {
    let state: EngineState
    let lastError: String?
    /// Shown as "Try again" next to a failure whose engine can be launched again.
    let retry: (() -> Void)?

    var body: some View {
        switch state {
        case .notInstalled(let paths):
            Label("Engine not installed: missing \(paths.joined(separator: ", "))",
                  systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
                .lineLimit(3)
        case .starting(let since):
            busy("Loading models", since: since)
        case let .downloading(done, total):
            HStack(spacing: 8) {
                ProgressView(value: Double(done), total: Double(max(total, 1)))
                    .frame(width: 120)
                Text("Downloading models, \(Self.gigabytes(done)) of \(Self.gigabytes(total)) GB")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            .help("First start only: the models stay in ~/Library/Application Support/ACE-Step.")
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
            HStack(spacing: 10) {
                Label(message, systemImage: "xmark.octagon")
                    .foregroundStyle(.red)
                    .lineLimit(3)
                if let retry {
                    Button("Try again", action: retry)
                }
            }
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

    static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.1f", Double(bytes) / 1e9)
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
/// While it plays, a slider shows and moves the position. It can be dragged out as a file.
struct SongRow: View {
    let song: Song
    let player: Player
    let canMakeAnother: Bool
    let makeAnother: () -> Void
    let trash: () -> Void
    @State private var expanded = false

    private var playing: Bool { player.playingID == song.id }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Button { player.toggle(song) } label: {
                    Image(systemName: playing ? "stop.circle.fill" : "play.circle.fill")
                        .font(.system(size: 26))
                }
                .buttonStyle(.plain)
                .help(playing ? "Stop (Space)" : "Play (Space)")
                .accessibilityLabel(playing ? "Stop" : "Play")
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
                .accessibilityLabel("Another like this")
                Button { expanded.toggle() } label: {
                    Image(systemName: expanded ? "chevron.up" : "text.quote")
                }
                .buttonStyle(.plain)
                .help("Lyrics")
                .accessibilityLabel(expanded ? "Hide lyrics" : "Show lyrics")
            }
            if playing {
                TimelineView(.periodic(from: .now, by: 0.25)) { _ in
                    HStack(spacing: 8) {
                        Slider(value: Binding(get: { player.currentTime }, set: { player.seek(to: $0) }),
                               in: 0...max(player.duration, 0.1))
                            .controlSize(.small)
                            .accessibilityLabel("Position")
                        Text("\(Self.clock(player.currentTime)) / \(Self.clock(player.duration))")
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }
            if expanded {
                Text(song.caption).font(.callout).foregroundStyle(.secondary)
                Text(song.lyrics).font(.callout).textSelection(.enabled)
            }
        }
        .padding(.vertical, 4)
        .onDrag {
            song.audioURL.flatMap { NSItemProvider(contentsOf: $0) } ?? NSItemProvider()
        }
        .contextMenu {
            if let url = song.audioURL {
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                ShareLink(item: url) { Label("Share…", systemImage: "square.and.arrow.up") }
            }
            Button("Copy Lyrics") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(song.lyrics, forType: .string)
            }
            Divider()
            Button("Move to Trash", role: .destructive, action: trash)
        }
    }

    private var facts: String {
        var parts: [String] = []
        if let ago = song.createdAgo() { parts.append(ago) }
        if let bpm = song.bpm { parts.append("\(bpm) BPM") }
        if let key = song.keyscale, !key.isEmpty { parts.append(key) }
        if song.instrumental == true {
            parts.append("instrumental")
        } else if let lang = song.language, !lang.isEmpty, lang != "unknown" {
            parts.append(Locale(identifier: "en_US").localizedString(forLanguageCode: lang) ?? lang)
        }
        if let seconds = song.duration { parts.append(Self.clock(seconds)) }
        return parts.joined(separator: " · ")
    }

    static func clock(_ seconds: Double) -> String {
        let whole = Int(seconds.rounded(.down))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}
