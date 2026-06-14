import SwiftUI
import UserNotifications

private final class NotificationPresentationDelegate: NSObject, UNUserNotificationCenterDelegate {
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .list, .sound]
    }
}

@main
struct RareBirdApp: App {
    private static let notificationDelegate = NotificationPresentationDelegate()
    @StateObject private var store = RareBirdStore(classifier: Self.makeClassifier())

    init() {
        UNUserNotificationCenter.current().delegate = Self.notificationDelegate
    }

    private static func makeClassifier() -> RarityClassifying {
        if let textClassifier = try? TextRarityClassifier() {
            return textClassifier
        }
        if LocalModelRarityClassifier.isModelAvailable {
            return DeferredRarityClassifier { LocalModelRarityClassifier() }
        }
        return MockRarityClassifier()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(store)
        }
    }
}

struct RootView: View {
    var body: some View {
        TabView {
            NowView()
                .tabItem {
                    Label("Now", systemImage: "scope")
                }

            LogbookView()
                .tabItem {
                    Label("Logbook", systemImage: "book.closed")
                }

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "slider.horizontal.3")
                }
        }
        .tint(.red)
    }
}
