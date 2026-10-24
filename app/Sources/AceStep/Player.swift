import AVFoundation
import AceStepKit
import Observation

/// Plays one song at a time.
@MainActor
@Observable
final class Player: NSObject, AVAudioPlayerDelegate {
    private(set) var playingID: String?
    private(set) var failure: String?
    @ObservationIgnored private var audio: AVAudioPlayer?

    /// Starts `song`, or stops it if it is the one playing.
    func toggle(_ song: Song) {
        if playingID == song.id {
            stop()
            return
        }
        stop()
        guard let url = song.audioURL else { return }
        do {
            let next = try AVAudioPlayer(contentsOf: url)
            next.delegate = self
            next.play()
            audio = next
            playingID = song.id
            failure = nil
        } catch {
            failure = "Cannot play \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }

    /// Seconds into the playing song; read on every tick of the row's timeline.
    var currentTime: Double { audio?.currentTime ?? 0 }
    var duration: Double { audio?.duration ?? 0 }

    func seek(to seconds: Double) {
        audio?.currentTime = min(max(seconds, 0), duration)
    }

    func stop() {
        audio?.stop()
        audio = nil
        playingID = nil
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.stop() }
    }
}
