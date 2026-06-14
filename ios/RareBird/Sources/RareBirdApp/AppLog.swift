import Foundation
import OSLog

enum AppLog {
    static let feed = Logger(subsystem: "com.rarebird.app", category: "feed")
    static let classifier = Logger(subsystem: "com.rarebird.app", category: "classifier")
    static let location = Logger(subsystem: "com.rarebird.app", category: "location")

    static func info(_ message: String, logger: Logger) {
        logger.info("\(message, privacy: .public)")
        print("[RareBird] \(message)")
    }

    static func error(_ message: String, logger: Logger) {
        logger.error("\(message, privacy: .public)")
        print("[RareBird] ERROR \(message)")
    }
}
