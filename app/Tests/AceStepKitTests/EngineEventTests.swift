import Foundation
import Testing

@testable import AceStepKit

/// Same fields and types as a real `done` line from the Mac mini (2026-10-23), lyrics replaced.
let doneLine = """
{"event": "done", "id": "11111111-aaaa", "song": {"prompt": "freetekno rave track, 180 bpm", \
"caption": "An aggressive, high-energy track", "lyrics": "[Intro]\\nYeah!", "bpm": 168, \
"keyscale": "E minor", "language": "pt", "duration": 30.0, "seed": 2685936523, \
"audio": "8404482a-122b-385b-b573-be81a7ce8e38.flac", \
"created": "2026-10-23T17:40:21.149894+00:00", "render_s": 30.6, \
"dir": "/Users/fab/acestep-f0/app_songs/20261023-194021-11111111"}}
"""

@Suite struct EngineEventTests {
    @Test func decodesEveryEventKind() throws {
        #expect(try EngineEvent.decode(Data(#"{"event": "ready", "protocol": 1, "load_s": 33.3}"#.utf8))
            == .ready(loadSeconds: 33.3))
        #expect(try EngineEvent.decode(Data(#"{"event": "phase", "id": "a", "phase": "rendering"}"#.utf8))
            == .phase(id: "a", phase: .rendering))
        #expect(try EngineEvent.decode(Data(#"{"event": "error", "id": null, "message": "no"}"#.utf8))
            == .error(id: nil, message: "no"))
    }

    @Test func decodesARealDoneLine() throws {
        guard case let .done(id, song) = try EngineEvent.decode(Data(doneLine.utf8)) else {
            Issue.record("not a done event")
            return
        }
        #expect(id == "11111111-aaaa")
        #expect(song.seed == 2_685_936_523)
        #expect(song.renderSeconds == 30.6)
        #expect(song.lyrics == "[Intro]\nYeah!")
        #expect(song.audioURL?.path
            == "/Users/fab/acestep-f0/app_songs/20261023-194021-11111111/8404482a-122b-385b-b573-be81a7ce8e38.flac")
    }

    @Test func rejectsMalformedEvents() {
        for line in ["{", #"{"event": "dance"}"#, #"{"event": "phase", "phase": "lyrics"}"#,
                     #"{"event": "phase", "id": "a", "phase": "mixing"}"#, #"{"event": "done", "id": "a"}"#] {
            #expect(throws: EngineEvent.DecodingFailure.self) { try EngineEvent.decode(Data(line.utf8)) }
        }
    }

    @Test func requestIsOneJSONLine() throws {
        let line = try GenerateRequest(id: "r1", prompt: "ballata però").line()
        #expect(line.last == 0x0A)
        let object = try JSONSerialization.jsonObject(with: line.dropLast()) as? [String: String]
        #expect(object == ["id": "r1", "cmd": "generate", "prompt": "ballata però"])
    }
}

@Suite struct LineSplitterTests {
    @Test func reassemblesLinesAcrossChunks() {
        var splitter = LineSplitter()
        #expect(splitter.feed(Data("{\"a\"".utf8)).isEmpty)
        #expect(splitter.feed(Data(":1}\n\n{\"b\":2}\n{\"c\"".utf8)).map { String(decoding: $0, as: UTF8.self) }
            == [#"{"a":1}"#, #"{"b":2}"#])
        #expect(splitter.feed(Data(":3}\n".utf8)).map { String(decoding: $0, as: UTF8.self) } == [#"{"c":3}"#])
    }
}
