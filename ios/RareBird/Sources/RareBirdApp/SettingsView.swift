import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var store: RareBirdStore
    @State private var notificationsEnabled = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Alerts") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Radius", systemImage: "location.circle")
                            Spacer()
                            Text("\(Int(store.settings.alertRadiusNauticalMiles)) nm")
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $store.settings.alertRadiusNauticalMiles, in: 5...60, step: 1)
                    }
                    Toggle(isOn: $store.settings.includeIncoming) {
                        Label("Incoming aircraft", systemImage: "arrow.down.left.circle")
                    }
                    Toggle(isOn: $store.settings.showNearMisses) {
                        Label("Near misses", systemImage: "scope")
                    }
                    Toggle(isOn: $store.settings.showRoutineAircraft) {
                        Label("Routine aircraft", systemImage: "airplane")
                    }
                    Toggle(isOn: $store.settings.claimableOnlyMode) {
                        Label("Claimable only", systemImage: "checkmark.seal")
                    }
                    Toggle(isOn: $notificationsEnabled) {
                        Label("Rare bird notifications", systemImage: "bell.badge")
                    }
                    Toggle(isOn: $store.settings.quietHoursEnabled) {
                        Label("Quiet hours", systemImage: "moon")
                    }
                    HStack {
                        Label("Notification status", systemImage: "bell")
                        Spacer()
                        Text(store.notificationStatus)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Live feed") {
                    Stepper(value: $store.settings.pollingIntervalSeconds, in: 3...30, step: 1) {
                        HStack {
                            Label("Polling", systemImage: "arrow.clockwise")
                            Spacer()
                            Text("\(Int(store.settings.pollingIntervalSeconds)) sec")
                                .foregroundStyle(.secondary)
                        }
                    }
                    Stepper(value: $store.settings.classificationIntervalSeconds, in: 15...300, step: 15) {
                        HStack {
                            Label("Classification", systemImage: "sparkles")
                            Spacer()
                            Text("\(Int(store.settings.classificationIntervalSeconds)) sec")
                                .foregroundStyle(.secondary)
                        }
                    }
                    Toggle(isOn: $store.settings.usesLocalModelBridge) {
                        Label("Local model bridge", systemImage: "cpu")
                    }
                }

                Section("Local model") {
                    HStack {
                        Label("Runtime", systemImage: "cpu")
                        Spacer()
                        Text(store.settings.usesLocalModelBridge ? "Small model + bridge" : "Small model")
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Label("Status", systemImage: "dot.radiowaves.left.and.right")
                        Spacer()
                        Text(store.modelStatus)
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Label("Model", systemImage: "shippingbox")
                        Spacer()
                        Text("0.16 MB")
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Label("Eval", systemImage: "checkmark.shield")
                        Spacer()
                        Text("8/8 contrast")
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Collection") {
                    HStack {
                        Label("Claim window", systemImage: "timer")
                        Spacer()
                        Text("10 min grace")
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Label("Photos", systemImage: "camera")
                        Spacer()
                        Text("Personal only")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .onAppear {
                notificationsEnabled = store.settings.rareBirdNotificationsEnabled
                Task { await store.refreshNotificationAuthorizationStatus() }
            }
            .onChange(of: store.settings.rareBirdNotificationsEnabled) { _, newValue in
                notificationsEnabled = newValue
            }
            .onChange(of: notificationsEnabled) { _, newValue in
                guard newValue != store.settings.rareBirdNotificationsEnabled else { return }
                Task { await store.setRareBirdNotificationsEnabled(newValue) }
            }
        }
    }
}
