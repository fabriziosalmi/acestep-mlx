import Foundation

/// Turns arbitrary pipe chunks into complete newline-terminated lines.
public struct LineSplitter: Sendable {
    private var pending = Data()

    public init() {}

    /// Appends `chunk` and returns every line it completes, without the newline.
    /// Empty lines are dropped; a trailing partial line waits for the next chunk.
    public mutating func feed(_ chunk: Data) -> [Data] {
        pending.append(chunk)
        var lines: [Data] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            let line = pending[pending.startIndex..<newline]
            if !line.isEmpty {
                lines.append(Data(line))
            }
            pending.removeSubrange(pending.startIndex...newline)
        }
        return lines
    }
}
