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
    /// Absolute folder of the song; sent by the engine, filled in by `SongStore` from disk.
    public var dir: String?

    enum CodingKeys: String, CodingKey {
        case prompt, caption, lyrics, bpm, keyscale, language, duration, seed, audio, created, dir
        case renderSeconds = "render_s"
    }

    public var id: String { dir ?? created + audio }

    public var audioURL: URL? {
        dir.map { URL(fileURLWithPath: $0).appendingPathComponent(audio) }
    }
}

/// A message from the engine process (see engine/acestep_mlx/protocol.py).
public enum EngineEvent: Sendable, Equatable {
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

/// The request that asks the engine for one song.
public struct GenerateRequest: Encodable, Sendable, Equatable {
    public let id: String
    public let cmd = "generate"
    public let prompt: String

    public init(id: String, prompt: String) {
        self.id = id
        self.prompt = prompt
    }

    /// The request as one protocol line, newline included.
    public func line() throws -> Data {
        var data = try JSONEncoder().encode(self)
        data.append(0x0A)
        return data
    }
}
