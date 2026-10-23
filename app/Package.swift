// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AceStep",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "AceStep", targets: ["AceStep"])],
    targets: [
        .target(name: "AceStepKit"),
        .executableTarget(name: "AceStep", dependencies: ["AceStepKit"]),
        // Development check: drives the real engine through the app model (no UI).
        .executableTarget(name: "AceStepE2E", dependencies: ["AceStepKit"]),
        .testTarget(name: "AceStepKitTests", dependencies: ["AceStepKit"]),
    ]
)
