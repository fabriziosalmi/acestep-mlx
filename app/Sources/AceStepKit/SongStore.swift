import Foundation

/// Reads the songs the engine wrote: one folder per song with `song.json` and the audio.
public enum SongStore {
    /// Every readable song under `dir`, newest first. Folders without a valid
    /// `song.json` or without their audio file are skipped.
    public static func load(from dir: URL, fileManager: FileManager = .default) -> [Song] {
        guard let folders = try? fileManager.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return [] }
        let songs: [Song] = folders.compactMap { folder in
            guard let data = try? Data(contentsOf: folder.appendingPathComponent("song.json")),
                  var song = try? JSONDecoder().decode(Song.self, from: data) else { return nil }
            song.dir = folder.path
            guard let audio = song.audioURL, fileManager.fileExists(atPath: audio.path) else {
                return nil
            }
            return song
        }
        return songs.sorted { $0.created > $1.created }
    }
}
