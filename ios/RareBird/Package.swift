// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "RareBird",
    platforms: [
        .iOS(.v17)
    ],
    products: [
        .executable(name: "RareBird", targets: ["RareBirdApp"])
    ],
    targets: [
        .executableTarget(
            name: "RareBirdApp",
            path: "Sources/RareBirdApp"
        )
    ]
)
