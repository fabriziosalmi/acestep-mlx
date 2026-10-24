// Draws the app icon and writes <out dir>/AppIcon.icns (via iconutil).
// Usage: swift scripts/make_icon.swift Resources
// A macOS-grid rounded square (824 of 1024 px) with a dusk gradient and a white waveform.
// Drawn with CoreGraphics only: SF Symbols may not be used in app icons.
import AppKit

func draw(size: Int) -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
                               bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                               colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let ctx = NSGraphicsContext.current!.cgContext
    let s = CGFloat(size) / 1024
    ctx.scaleBy(x: s, y: s)

    let tile = CGRect(x: 100, y: 100, width: 824, height: 824)
    let shape = CGPath(roundedRect: tile, cornerWidth: 185, cornerHeight: 185, transform: nil)
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -10), blur: 28, color: NSColor.black.withAlphaComponent(0.3).cgColor)
    ctx.addPath(shape)
    ctx.setFillColor(NSColor.black.cgColor)
    ctx.fillPath()
    ctx.restoreGState()

    ctx.saveGState()
    ctx.addPath(shape)
    ctx.clip()
    let colors = [NSColor(srgbRed: 0.93, green: 0.30, blue: 0.45, alpha: 1).cgColor,
                  NSColor(srgbRed: 0.17, green: 0.11, blue: 0.38, alpha: 1).cgColor] as CFArray
    let gradient = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: colors, locations: [0, 1])!
    ctx.drawLinearGradient(gradient, start: CGPoint(x: 512, y: 100), end: CGPoint(x: 512, y: 924), options: [])

    // Waveform: rounded bars, symmetric around the middle line.
    let heights: [CGFloat] = [0.22, 0.42, 0.68, 0.94, 0.58, 0.82, 0.48, 0.32, 0.18]
    let bar: CGFloat = 46, gap: CGFloat = 26, maxHeight: CGFloat = 520
    let width = CGFloat(heights.count) * bar + CGFloat(heights.count - 1) * gap
    var x = 512 - width / 2
    ctx.setFillColor(NSColor.white.withAlphaComponent(0.95).cgColor)
    for h in heights {
        let height = maxHeight * h
        let rect = CGRect(x: x, y: 512 - height / 2, width: bar, height: height)
        ctx.addPath(CGPath(roundedRect: rect, cornerWidth: bar / 2, cornerHeight: bar / 2, transform: nil))
        x += bar + gap
    }
    ctx.fillPath()
    ctx.restoreGState()

    NSGraphicsContext.restoreGraphicsState()
    return rep
}

let out = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? ".")
let iconset = FileManager.default.temporaryDirectory.appendingPathComponent("AppIcon.iconset")
try? FileManager.default.removeItem(at: iconset)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for base in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = scale == 1 ? "icon_\(base)x\(base).png" : "icon_\(base)x\(base)@2x.png"
        let png = draw(size: base * scale).representation(using: .png, properties: [:])!
        try png.write(to: iconset.appendingPathComponent(name))
    }
}
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", out.appendingPathComponent("AppIcon.icns").path]
try iconutil.run()
iconutil.waitUntilExit()
exit(iconutil.terminationStatus)
