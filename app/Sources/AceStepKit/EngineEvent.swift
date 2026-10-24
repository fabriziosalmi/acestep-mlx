import Foundation

/// The three steps the engine reports while making a song, in order.
public enum Phase: String, Codable, Sendable, CaseIterable {
    case lyrics, composing, rendering
}

/// One finished song, as the engine writes it to `song.json` and sends it in `done`.
public struct Song: Codable, Sendable, Equatable, Identifiable {
    public var prompt: String
    public var caption: String
    public var lyrics: String
    public var bpm: Int?
    public var keyscale: String?
    public var language: String?
    public var duration: Double?
    public var seed: Int?
    /// Audio file name inside `dir`.
    public var audio: String
    /// ISO 8601 creation time.
    public var created: String
    public var renderSeconds: Double?
    public var instrumental: Bool?
    /// Folder name of the song this one is another version of.
    public var parent: String?
    public var preset: String?
    /// Absolute folder of the song; sent by the engine, filled in by `SongStore` from disk.
    public var dir: String?

    enum CodingKeys: String, CodingKey {
        case prompt, caption, lyrics, bpm, keyscale, language, duration, seed, audio, created, dir
        case instrumental, parent, preset
        case renderSeconds = "render_s"
    }

    public var id: String { dir ?? created + audio }

    /// The song's folder name, which the engine accepts as `like`.
    public var name: String? {
        dir.map { URL(fileURLWithPath: $0).lastPathComponent }
    }

    public var audioURL: URL? {
        dir.map { URL(fileURLWithPath: $0).appendingPathComponent(audio) }
    }

    /// "3 minutes ago", "yesterday": always in English, like the rest of the app, whatever
    /// the system's language.
    public func createdAgo(relativeTo now: Date = Date()) -> String? {
        guard let date = createdDate else { return nil }
        let formatter = RelativeDateTimeFormatter()
        formatter.locale = Locale(identifier: "en_US")
        formatter.dateTimeStyle = .named
        return formatter.localizedString(for: date, relativeTo: now)
    }

    /// `created` as a date. The engine writes microseconds ("…T17:40:21.149894+00:00"),
    /// which ISO8601DateFormatter does not read, so the fraction is dropped first.
    public var createdDate: Date? {
        let plain = created.replacingOccurrences(of: #"\.\d+"#, with: "", options: .regularExpression)
        return ISO8601DateFormatter().date(from: plain)
    }
}

/// A message from the engine process (see engine/acestep_mlx/protocol.py).
public enum EngineEvent: Sendable, Equatable {
    /// Model files being fetched before `ready`: bytes present of bytes needed; `complete`
    /// once every file is checked.
    case download(done: Int64, total: Int64, complete: Bool)
    case ready(loadSeconds: Double?)
    case phase(id: String, phase: Phase)
    case done(id: String, song: Song)
    case error(id: String?, message: String)

    public struct DecodingFailure: Error, Equatable {
        public let reason: String
    }

    private struct Wire: Decodable {
        let event: String
        let id: String?
        let phase: Phase?
        let song: Song?
        let message: String?
        let load_s: Double?
        let done: Int64?
        let total: Int64?
        let complete: Bool?
    }

    /// Decodes one protocol line.
    /// - Throws: `DecodingFailure` for invalid JSON, an unknown event or a missing field.
    public static func decode(_ line: Data) throws -> EngineEvent {
        let wire: Wire
        do {
            wire = try JSONDecoder().decode(Wire.self, from: line)
        } catch {
            throw DecodingFailure(reason: "invalid event: \(error)")
        }
        switch (wire.event, wire.id, wire.phase, wire.song, wire.message) {
        case ("download", _, _, _, _):
            guard let done = wire.done, let total = wire.total else {
                throw DecodingFailure(reason: "download without done and total")
            }
            return .download(done: done, total: total, complete: wire.complete ?? false)
        case ("ready", _, _, _, _):
            return .ready(loadSeconds: wire.load_s)
        case let ("phase", id?, phase?, _, _):
            return .phase(id: id, phase: phase)
        case let ("done", id?, _, song?, _):
            return .done(id: id, song: song)
        case let ("error", id, _, _, message?):
            return .error(id: id, message: message)
        default:
            throw DecodingFailure(reason: "unexpected event \(wire.event)")
        }
    }
}

/// Per-request options (engine/acestep_mlx/recipe.py); nil fields are not sent,
/// so the engine's defaults apply.
public struct GenerateOptions: Encodable, Sendable, Equatable {
    public var literal: Bool?
    public var vocals: Bool?
    public var duration: Double?
    public var format: String?
    /// Folder name of a song to make another version of.
    public var like: String?
    /// The user's own lyrics, sung as written.
    public var lyrics: String?

    public init(literal: Bool? = nil, vocals: Bool? = nil, duration: Double? = nil,
                format: String? = nil, like: String? = nil, lyrics: String? = nil) {
        self.literal = literal
        self.vocals = vocals
        self.duration = duration
        self.format = format
        self.like = like
        self.lyrics = lyrics
    }
}

/// The request that asks the engine for one song.
public struct GenerateRequest: Encodable, Sendable, Equatable {
    public let id: String
    public let cmd = "generate"
    public let prompt: String
    public let options: GenerateOptions

    public init(id: String, prompt: String, options: GenerateOptions = GenerateOptions()) {
        self.id = id
        self.prompt = prompt
        self.options = options
    }

    /// The request as one protocol line, newline included.
    public func line() throws -> Data {
        var data = try JSONEncoder().encode(self)
        data.append(0x0A)
        return data
    }
}
