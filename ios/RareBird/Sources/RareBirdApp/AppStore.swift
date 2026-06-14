import Foundation
import MapKit
import UserNotifications

@MainActor
final class RareBirdStore: ObservableObject {
    @Published var sightings: [AircraftSighting]
    @Published var logbook: [LogbookEntry]
    @Published var settings: UserSettings {
        didSet {
            persistSettings()
        }
    }
    @Published var selectedSightingID: AircraftSighting.ID?
    @Published var modelStatus = "Model bridge idle"
    @Published var feedStatus = "Waiting for live ADS-B"
    @Published var notificationStatus = "Notifications off"
    @Published var liveFeedState: LiveFeedState = .idle

    private let classifier: RarityClassifying
    private let aircraftFeed: AircraftFeedFetching
    private let userDefaults: UserDefaults
    private var isRefreshingLiveSightings = false
    private var hasReceivedLiveFeed = false
    private var notifiedSightingKeys: Set<String> = []

    init(classifier: RarityClassifying? = nil, aircraftFeed: AircraftFeedFetching? = nil, userDefaults: UserDefaults = .standard) {
        self.classifier = classifier ?? MockRarityClassifier()
        self.aircraftFeed = aircraftFeed ?? ADSBFiAircraftFeedClient()
        self.userDefaults = userDefaults
        let persistedUserState = Self.loadPersistedUserState(from: userDefaults)
        self.sightings = []
        self.logbook = persistedUserState?.logbook ?? SampleData.logbook
        self.settings = Self.loadPersistedSettings(from: userDefaults) ?? UserSettings()
        Task { await refreshNotificationAuthorizationStatus() }
    }

    var featuredSightings: [AircraftSighting] {
        sightings.filter { $0.rarityBand == .alert || $0.rarityBand == .incoming }
    }

    var nearMisses: [AircraftSighting] {
        sightings.filter { $0.rarityBand == .nearMiss }
    }

    var selectedSighting: AircraftSighting? {
        guard let selectedSightingID else { return nil }
        return sightings.first { $0.id == selectedSightingID }
    }

    func claim(_ sighting: AircraftSighting) {
        guard let index = sightings.firstIndex(where: { $0.id == sighting.id }) else { return }
        sightings[index].claimState = .claimed

        if let logIndex = logbook.firstIndex(where: { $0.collectionKey == sighting.collectionKey }) {
            logbook[logIndex].claimCount += 1
            logbook[logIndex].firstClaimedAt = min(logbook[logIndex].firstClaimedAt, Date())
        } else {
            logbook.insert(
                LogbookEntry(
                    id: UUID(),
                    collectionKey: sighting.collectionKey,
                    title: sighting.description,
                    subtitle: sighting.operatorName,
                    firstClaimedAt: Date(),
                    claimCount: 1,
                    hasPhoto: false
                ),
                at: 0
            )
        }
        sightings[index] = applyingPersonalizedAlertPriority(to: sightings[index])
        persistUserState()
    }

    func markPhotoLogged(_ sighting: AircraftSighting) {
        guard let index = sightings.firstIndex(where: { $0.id == sighting.id }) else { return }
        sightings[index].claimState = .photoLogged

        if let logIndex = logbook.firstIndex(where: { $0.collectionKey == sighting.collectionKey }) {
            logbook[logIndex].hasPhoto = true
        } else {
            logbook.insert(
                LogbookEntry(
                    id: UUID(),
                    collectionKey: sighting.collectionKey,
                    title: sighting.description,
                    subtitle: sighting.operatorName,
                    firstClaimedAt: Date(),
                    claimCount: 1,
                    hasPhoto: true
                ),
                at: 0
            )
        }
        sightings[index] = applyingPersonalizedAlertPriority(to: sightings[index])
        persistUserState()
    }

    func refreshLiveSightings(
        near coordinate: CLLocationCoordinate2D? = nil,
        classify: Bool = true,
        classificationLimit: Int = 18,
        showsLoadingStatus: Bool = true
    ) async {
        guard !isRefreshingLiveSightings else {
            AppLog.info("live refresh skipped because another refresh is running", logger: AppLog.feed)
            return
        }

        isRefreshingLiveSightings = true
        defer { isRefreshingLiveSightings = false }

        let center = coordinate ?? CLLocationCoordinate2D(latitude: 33.84, longitude: -118.12)
        let radius = Int(settings.alertRadiusNauticalMiles.rounded())
        let isForegroundRefresh = showsLoadingStatus || liveFeedState == .idle
        if isForegroundRefresh {
            feedStatus = "Fetching ADS-B"
            liveFeedState = .loading
        }
        AppLog.info("live refresh start lat=\(center.latitude) lon=\(center.longitude) radius_nm=\(radius)", logger: AppLog.feed)

        do {
            let liveSightings = try await aircraftFeed.fetchAircraft(
                near: AircraftFeedRequest(center: center, radiusNauticalMiles: radius)
            ).map { $0.applyingRarityPolicyGuardrails() }
            hasReceivedLiveFeed = true
            liveFeedState = .available
            if liveSightings.isEmpty {
                feedStatus = "No ADS-B aircraft nearby"
                sightings = []
                selectedSightingID = nil
                AppLog.info("live refresh returned zero aircraft", logger: AppLog.feed)
                return
            }

            mergeLiveSightings(liveSightings)
            feedStatus = "\(liveSightings.count) live ADS-B"
            AppLog.info("live refresh loaded count=\(liveSightings.count)", logger: AppLog.feed)
            if classify {
                await refreshClassifications(limit: classificationLimit)
            } else {
                modelStatus = "Deterministic guardrails only"
                await notifyForNewRareBirdsIfNeeded()
            }
        } catch {
            if !hasReceivedLiveFeed {
                feedStatus = "Live feed unavailable"
                liveFeedState = .unavailable
                sightings = []
                selectedSightingID = nil
            } else {
                feedStatus = "Live feed stale"
                liveFeedState = .available
            }
            AppLog.error("live refresh failed error=\(error.localizedDescription)", logger: AppLog.feed)
            if classify {
                await refreshClassifications(limit: min(classificationLimit, 4))
            } else {
                modelStatus = "Deterministic guardrails only"
                await notifyForNewRareBirdsIfNeeded()
            }
        }
    }

    func refreshClassifications(limit: Int? = nil) async {
        modelStatus = "Classifying on device"
        let indices = classificationCandidateIndices(limit: limit)
        var failureCount = 0
        AppLog.info("classification start candidates=\(indices.count) total_sightings=\(sightings.count)", logger: AppLog.classifier)
        for index in indices {
            let sighting = sightings[index]
            AppLog.info("classification request hex=\(sighting.icaoHex) callsign=\(sighting.callsign) type=\(sighting.typeDesignator)", logger: AppLog.classifier)
            let input = RarityInput(
                aircraft: sighting,
                observerArea: sighting.distanceNauticalMiles < 6 ? "central Orange County near SNA" : "Los Angeles basin",
                distanceToNearestMilitaryNauticalMiles: sighting.operatorName.contains("Army") ? 2 : 24,
                militaryPattern: sighting.rarityBand == .nearMiss ? "base_pattern" : "not_base_pattern"
            )
            do {
                let result = try await classifier.classify(input)
                if let deterministicAlertReason = sighting.deterministicAlertReason {
                    sightings[index].confidence = max(result.confidence, 0.88)
                    sightings[index].reason = result.isRare ? result.reason : deterministicAlertReason
                    sightings[index].rarityBand = sighting.rarityBand == .incoming ? .incoming : .alert
                } else if let contextualSuppressionReason = sighting.contextualSuppressionReason {
                    sightings[index].confidence = min(result.confidence, 0.72)
                    sightings[index].reason = contextualSuppressionReason
                    if sighting.rarityBand != .nearMiss {
                        sightings[index].rarityBand = .routine
                    }
                } else if result.isRare {
                    sightings[index].confidence = result.confidence
                    sightings[index].reason = result.reason
                    sightings[index].rarityBand = sighting.rarityBand == .incoming ? .incoming : .alert
                } else if sighting.rarityBand != .nearMiss {
                    sightings[index].confidence = result.confidence
                    sightings[index].reason = result.reason
                    sightings[index].rarityBand = .routine
                }
                sightings[index] = applyingPersonalizedAlertPriority(to: sightings[index])
                AppLog.info("classification result hex=\(sighting.icaoHex) rare=\(result.isRare) confidence=\(result.confidence)", logger: AppLog.classifier)
            } catch {
                failureCount += 1
                sightings[index] = applyingPersonalizedAlertPriority(to: sighting.applyingRarityPolicyGuardrails())
                AppLog.error("classification failed hex=\(sighting.icaoHex) error=\(error.localizedDescription)", logger: AppLog.classifier)
            }
        }
        modelStatus = failureCount == 0 ? "Classified on device" : "Classified with local guardrails"
        AppLog.info("classification complete candidates=\(indices.count)", logger: AppLog.classifier)
        await notifyForNewRareBirdsIfNeeded()
    }

    func setRareBirdNotificationsEnabled(_ isEnabled: Bool) async {
        if isEnabled {
            let granted = await requestNotificationAuthorization()
            settings.rareBirdNotificationsEnabled = granted
            notificationStatus = granted ? "Notifications on" : "Notifications blocked"
            if granted {
                await notifyForNewRareBirdsIfNeeded()
            }
        } else {
            settings.rareBirdNotificationsEnabled = false
            notificationStatus = "Notifications off"
            UNUserNotificationCenter.current().removeAllPendingNotificationRequests()
        }
    }

    func refreshNotificationAuthorizationStatus() async {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        switch settings.authorizationStatus {
        case .authorized, .provisional, .ephemeral:
            notificationStatus = self.settings.rareBirdNotificationsEnabled ? "Notifications on" : "Notifications off"
        case .denied:
            notificationStatus = "Notifications blocked"
            self.settings.rareBirdNotificationsEnabled = false
        case .notDetermined:
            notificationStatus = "Notifications off"
        @unknown default:
            notificationStatus = "Notifications unavailable"
        }
    }

    private func requestNotificationAuthorization() async -> Bool {
        do {
            return try await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound])
        } catch {
            AppLog.error("notification authorization failed error=\(error.localizedDescription)", logger: AppLog.feed)
            return false
        }
    }

    private func notifyForNewRareBirdsIfNeeded() async {
        guard settings.rareBirdNotificationsEnabled else { return }
        guard !isQuietHoursNow else { return }
        let authorizationSettings = await UNUserNotificationCenter.current().notificationSettings()
        guard authorizationSettings.authorizationStatus == .authorized
            || authorizationSettings.authorizationStatus == .provisional
            || authorizationSettings.authorizationStatus == .ephemeral
        else {
            notificationStatus = authorizationSettings.authorizationStatus == .denied ? "Notifications blocked" : "Notifications off"
            return
        }

        let newClaimableSightings = featuredSightings.filter { sighting in
            guard sighting.claimState == .detected else { return false }
            return !notifiedSightingKeys.contains(notificationKey(for: sighting))
        }

        for sighting in newClaimableSightings {
            notifiedSightingKeys.insert(notificationKey(for: sighting))
            await scheduleRareBirdNotification(for: sighting)
        }
    }

    private var isQuietHoursNow: Bool {
        guard settings.quietHoursEnabled else { return false }
        let hour = Calendar.current.component(.hour, from: Date())
        return hour >= 22 || hour < 7
    }

    private func scheduleRareBirdNotification(for sighting: AircraftSighting) async {
        let content = UNMutableNotificationContent()
        content.title = "\(sighting.rarityBand.label): \(sighting.displayName)"
        content.body = "\(String(format: "%.1f", sighting.distanceNauticalMiles)) nm away - \(sighting.reason)"
        content.sound = .default
        content.userInfo = [
            "sightingID": sighting.id.uuidString,
            "icaoHex": sighting.icaoHex,
            "collectionKey": sighting.collectionKey,
        ]

        let request = UNNotificationRequest(
            identifier: notificationKey(for: sighting),
            content: content,
            trigger: nil
        )

        do {
            try await UNUserNotificationCenter.current().add(request)
            notificationStatus = "Notifications on"
            AppLog.info("rare bird notification scheduled hex=\(sighting.icaoHex)", logger: AppLog.feed)
        } catch {
            notificationStatus = "Notification failed"
            AppLog.error("rare bird notification failed hex=\(sighting.icaoHex) error=\(error.localizedDescription)", logger: AppLog.feed)
        }
    }

    private func notificationKey(for sighting: AircraftSighting) -> String {
        "rare-bird:\(sighting.icaoHex.uppercased()):\(sighting.collectionKey.uppercased())"
    }

    private func classificationCandidateIndices(limit: Int?) -> [Array<AircraftSighting>.Index] {
        let ranked = sightings.indices.filter { index in
            sightings[index].shouldRequestModelClassification
        }.sorted { lhs, rhs in
            let left = sightings[lhs]
            let right = sightings[rhs]
            if left.rarityPriority != right.rarityPriority {
                return left.rarityPriority > right.rarityPriority
            }
            return left.distanceNauticalMiles < right.distanceNauticalMiles
        }

        if let limit {
            return Array(ranked.prefix(limit))
        }
        return ranked
    }

    private func mergeLiveSightings(_ liveSightings: [AircraftSighting]) {
        var existingByHex: [String: AircraftSighting] = [:]
        for sighting in sightings {
            existingByHex[sighting.icaoHex.uppercased()] = sighting
        }
        let liveHexes = Set(liveSightings.map { $0.icaoHex.uppercased() })
        var merged: [AircraftSighting] = []

        for liveSighting in liveSightings {
            let key = liveSighting.icaoHex.uppercased()
            guard let existing = existingByHex[key] else {
                merged.append(applyingPersonalizedAlertPriority(to: restoringPersistedClaimState(for: liveSighting)))
                continue
            }

            var updated = liveSighting
            updated.id = existing.id
            if !liveSighting.hasPersonalizationBypassSignal {
                updated.rarityBand = existing.rarityBand
                updated.confidence = existing.confidence
                updated.reason = existing.reason
            }
            updated.claimState = existing.claimState
            updated.collectionKey = existing.collectionKey
            merged.append(applyingPersonalizedAlertPriority(to: restoringPersistedClaimState(for: updated)))
        }

        sightings = merged
        if let selectedSightingID, !sightings.contains(where: { $0.id == selectedSightingID }) {
            self.selectedSightingID = nil
        }
        AppLog.info("live refresh merged existing=\(existingByHex.count) current=\(liveHexes.count)", logger: AppLog.feed)
    }

    private func restoringPersistedClaimState(for sighting: AircraftSighting) -> AircraftSighting {
        guard let persistedUserState = Self.loadPersistedUserState(from: userDefaults) else {
            return sighting
        }
        var updated = sighting
        let keys = claimPersistenceKeys(for: sighting)
        let restoredState = keys.compactMap { persistedUserState.claimStates[$0] }.max { lhs, rhs in
            lhs.persistencePriority < rhs.persistencePriority
        }
        if let restoredState {
            updated.claimState = restoredState
        }
        return updated
    }

    private func applyingPersonalizedAlertPriority(to sighting: AircraftSighting) -> AircraftSighting {
        guard shouldDemotePreviouslyClaimed(sighting) else {
            return sighting
        }

        var updated = sighting
        updated.rarityBand = .nearMiss
        updated.confidence = min(updated.confidence, 0.74)
        if sighting.claimState == .detected {
            updated.reason = "Seen before: you already have \(sighting.collectionKey) in your logbook, so this is visible but no longer promoted as a fresh rare alert."
        } else {
            updated.reason = "Already claimed: keeping this sighting visible, but lowering its alert priority unless a new hard alert signal appears."
        }
        return updated
    }

    private func shouldDemotePreviouslyClaimed(_ sighting: AircraftSighting) -> Bool {
        guard sighting.rarityBand == .alert || sighting.rarityBand == .incoming else {
            return false
        }
        guard !sighting.hasPersonalizationBypassSignal else {
            return false
        }
        if sighting.claimState != .detected {
            return true
        }
        return logbook.contains { $0.collectionKey.caseInsensitiveCompare(sighting.collectionKey) == .orderedSame }
    }

    private func persistUserState() {
        var claimStates = Self.loadPersistedUserState(from: userDefaults)?.claimStates ?? [:]
        for sighting in sightings where sighting.claimState != .detected {
            for key in claimPersistenceKeys(for: sighting) {
                claimStates[key] = sighting.claimState
            }
        }

        let persisted = PersistedUserState(logbook: logbook, claimStates: claimStates)
        do {
            let data = try JSONEncoder().encode(persisted)
            userDefaults.set(data, forKey: Self.persistedUserStateKey)
        } catch {
            AppLog.error("persist user state failed error=\(error.localizedDescription)", logger: AppLog.feed)
        }
    }

    private func claimPersistenceKeys(for sighting: AircraftSighting) -> [String] {
        var keys = [
            "collection:\(sighting.collectionKey.uppercased())",
            "hex:\(sighting.icaoHex.uppercased())",
        ]
        if let registration = sighting.registration?.trimmingCharacters(in: .whitespacesAndNewlines), !registration.isEmpty {
            keys.append("registration:\(registration.uppercased())")
        }
        return keys
    }

    private static func loadPersistedUserState(from userDefaults: UserDefaults) -> PersistedUserState? {
        guard let data = userDefaults.data(forKey: persistedUserStateKey) else {
            return nil
        }
        do {
            return try JSONDecoder().decode(PersistedUserState.self, from: data)
        } catch {
            AppLog.error("load persisted user state failed error=\(error.localizedDescription)", logger: AppLog.feed)
            return nil
        }
    }

    private func persistSettings() {
        do {
            let data = try JSONEncoder().encode(settings)
            userDefaults.set(data, forKey: Self.persistedSettingsKey)
        } catch {
            AppLog.error("persist settings failed error=\(error.localizedDescription)", logger: AppLog.feed)
        }
    }

    private static func loadPersistedSettings(from userDefaults: UserDefaults) -> UserSettings? {
        guard let data = userDefaults.data(forKey: persistedSettingsKey) else {
            return nil
        }
        do {
            return try JSONDecoder().decode(UserSettings.self, from: data)
        } catch {
            AppLog.error("load persisted settings failed error=\(error.localizedDescription)", logger: AppLog.feed)
            return nil
        }
    }

    private static let persistedUserStateKey = "RareBird.persistedUserState.v1"
    private static let persistedSettingsKey = "RareBird.userSettings.v1"
}

private struct PersistedUserState: Codable {
    var logbook: [LogbookEntry]
    var claimStates: [String: ClaimState]
}

private extension ClaimState {
    var persistencePriority: Int {
        switch self {
        case .detected: 0
        case .claimed: 1
        case .photoLogged: 2
        }
    }
}

private extension AircraftSighting {
    func applyingRarityPolicyGuardrails() -> AircraftSighting {
        var sighting = self
        if let deterministicAlertReason {
            sighting.rarityBand = .alert
            sighting.confidence = 0.92
            sighting.reason = deterministicAlertReason
        } else if let nearMissReason = deterministicNearMissReason {
            sighting.rarityBand = .nearMiss
            sighting.confidence = 0.78
            sighting.reason = nearMissReason
        } else if let routineReason = deterministicRoutineReason {
            sighting.rarityBand = .routine
            sighting.confidence = 0.86
            sighting.reason = routineReason
        }
        return sighting
    }

    var shouldRequestModelClassification: Bool {
        deterministicRoutineReason == nil || deterministicAlertReason != nil || rarityBand == .nearMiss
    }

    var rarityPriority: Int {
        let type = typeDesignator.uppercased()
        let callsign = callsign.uppercased()
        let operatorName = operatorName.uppercased()
        let description = description.uppercased()

        if let squawk, Self.emergencySquawks.contains(squawk) {
            return 100
        }
        if let registration, Self.specialRegistrations.contains(registration.uppercased()) {
            return 100
        }
        if Self.hardAlertCallsignPrefixes.contains(where: { callsign.hasPrefix($0) }) {
            return 95
        }
        if Self.rareTypeDesignators.contains(type) {
            return 90
        }
        if Self.lastOfTypeDesignators.contains(type) {
            return 80
        }
        if Self.contextualTypeDesignators.contains(type) {
            return 65
        }
        if Self.modelCandidateCallsignPrefixes.contains(where: { callsign.hasPrefix($0) }) {
            return 75
        }
        if Self.modelCandidateTextSignals.contains(where: { operatorName.contains($0) || description.contains($0) }) {
            return 70
        }
        return 10
    }

    var deterministicAlertReason: String? {
        let type = typeDesignator.uppercased()
        let callsign = callsign.uppercased()
        let operatorName = operatorName.uppercased()
        let description = description.uppercased()

        if let squawk, Self.emergencySquawks.contains(squawk) {
            return "Emergency squawk \(squawk) is alert-worthy regardless of aircraft type."
        }
        if let registration, Self.specialRegistrations.contains(registration.uppercased()) {
            return "\(registration) is a known special-livery or notable individual aircraft."
        }
        if Self.hardAlertCallsignPrefixes.contains(where: { callsign.hasPrefix($0) }) {
            return "\(callsign) uses a rare or special-mission callsign prefix."
        }
        if (Self.rareTypeDesignators.contains(type) || Self.lastOfTypeDesignators.contains(type)) && !Self.contextualLocalMilitaryTypes.contains(type) {
            return "\(typeDesignator) is an uncommon or collection-worthy aircraft type for this region."
        }
        if Self.hardAlertMissionSignals.contains(where: { operatorName.contains($0) || description.contains($0) }) {
            return "A hard special-mission signal is alert-worthy even when the airframe type is common."
        }
        return nil
    }

    var hasPersonalizationBypassSignal: Bool {
        let type = typeDesignator.uppercased()
        let callsign = callsign.uppercased()
        let operatorName = operatorName.uppercased()
        let description = description.uppercased()

        if let squawk, Self.emergencySquawks.contains(squawk) {
            return true
        }
        if let registration, Self.specialRegistrations.contains(registration.uppercased()) {
            return true
        }
        if Self.hardAlertCallsignPrefixes.contains(where: { callsign.hasPrefix($0) }) {
            return true
        }
        if Self.hardAlertMissionSignals.contains(where: { operatorName.contains($0) || description.contains($0) }) {
            return true
        }
        return type == "BLCF"
    }

    var deterministicNearMissReason: String? {
        let type = typeDesignator.uppercased()
        let operatorName = operatorName.uppercased()
        let description = description.uppercased()

        if Self.localMilitaryHelicopterTypes.contains(type) {
            return "\(typeDesignator) helicopters are interesting, but commonly seen around Southern California military and training routes without a hard alert signal."
        }
        if Self.publicSafetySignals.contains(where: { operatorName.contains($0) || description.contains($0) }) {
            return "Public-safety helicopter activity is worth showing, but locally common without an emergency squawk, notable registration, unusual type, or special incident callsign."
        }
        if Self.contextualTypeDesignators.contains(type) {
            return "\(typeDesignator) is uncommon enough to watch, but not an automatic rare alert without a special livery, unusual operator, emergency, or notable route signal."
        }
        return nil
    }

    var deterministicRoutineReason: String? {
        let type = typeDesignator.uppercased()
        let operatorName = operatorName.uppercased()
        let description = description.uppercased()

        if Self.commonAirlineTypeDesignators.contains(type) || Self.commonAirlineSignals.contains(where: { operatorName.contains($0) || description.contains($0) }) {
            return "\(displayName) is routine Southern California airline traffic without an emergency, notable registration, special livery, or unusual mission signal."
        }
        if Self.commonBusinessJetTypeDesignators.contains(type) {
            return "\(typeDesignator) business jet traffic is common locally unless paired with a hard alert signal."
        }
        if Self.commonTrainerAndGATypeDesignators.contains(type)
            || Self.commonTrainerAndGADescriptionSignals.contains(where: { description.contains($0) || operatorName.contains($0) }) {
            return "\(typeDesignator) trainer or general-aviation traffic is common around local airports and practice areas without a watchlist or emergency signal."
        }
        if Self.unknownTypeDesignators.contains(type) {
            return "Unknown aircraft type is not rare by itself. Keep it routine unless ADS-B data also shows an emergency, special-mission callsign, notable registration, or identifiable rare aircraft signal."
        }
        return nil
    }

    var contextualSuppressionReason: String? {
        let type = typeDesignator.uppercased()
        if type == "B744" {
            return "Boeing 747-400 cargo traffic is contextual in Southern California: uncommon, but not an automatic rare alert without a special livery, notable registration, emergency, unusual operator, or unusual route signal."
        }
        if type == "B748" {
            return "Boeing 747-8 traffic is contextual: uncommon, but not an automatic rare alert without a special livery, unusual operator, notable registration, emergency, or unusual route signal."
        }
        return nil
    }

    static let rareTypeDesignators: Set<String> = [
        "A10", "A124", "A225", "A306", "A3ST", "A337", "A342", "A343", "A345", "A346",
        "B1", "B2", "B703", "B712", "B721", "B722", "B741", "B742", "B743",
        "BLCF", "B52", "C5", "C17", "C30J", "C130", "CONI", "CVLT", "CVLP",
        "DC10", "DC3", "DC6", "DC8", "DC85", "DC86", "DC87", "DC91", "DC93", "DC95",
        "E2", "E3", "E6", "F15", "F16", "F18", "F22", "F35", "H47", "H53", "H60",
        "IL18", "IL62", "IL76", "KC10", "KC135", "KC46", "L101", "MD11", "MD80",
        "MD81", "MD82", "MD83", "MD87", "MD88", "MD90", "P3", "P8", "T38", "U2", "V22",
    ]

    static let contextualTypeDesignators: Set<String> = ["B744", "B748"]
    static let unknownTypeDesignators: Set<String> = ["UNK", "UNKNOWN"]
    static let contextualLocalMilitaryTypes: Set<String> = ["H60"]
    static let localMilitaryHelicopterTypes: Set<String> = ["H60"]
    static let lastOfTypeDesignators: Set<String> = ["B753"]
    static let emergencySquawks: Set<String> = ["7500", "7600", "7700"]
    static let specialRegistrations: Set<String> = ["A7-BEG", "B-LRJ", "D-ABYN"]
    static let hardAlertCallsignPrefixes: [String] = [
        "ASY", "EVAC", "GUARD", "JOSA", "NASA", "RCH", "REACH", "RESCUE", "VENUS",
    ]
    static let modelCandidateCallsignPrefixes: [String] = [
        "AF", "BOLT", "DEATH", "DOOM", "FORGE", "GRIM", "NACHO", "NAVY", "PATON", "SHADY", "SPUR", "TITAN",
    ]
    static let hardAlertMissionSignals: [String] = [
        "NASA", "VINTAGE", "WARBIRD", "ANTONOV", "BELUGA", "DREAMLIFTER", "CONVAIR", "EXPERIMENTAL",
    ]
    static let modelCandidateTextSignals: [String] = [
        "AIR FORCE", "USAF", "NAVY", "MARINE", "ARMY", "COAST GUARD", "BORDER", "CUSTOMS",
        "HOMELAND", "GOVERNMENT", "LOCKHEED", "NORTHROP", "BOEING", "RAYTHEON", "GENERAL ATOMICS",
        "VINTAGE", "WARBIRD", "ANTONOV", "BELUGA", "DREAMLIFTER", "CONVAIR",
    ]
    static let publicSafetySignals: [String] = [
        "POLICE", "SHERIFF", "FIRE", "CAL FIRE", "COAST GUARD", "MEDEVAC", "MEDICAL", "RESCUE",
    ]
    static let commonAirlineSignals: [String] = [
        "AMERICAN AIRLINES", "DELTA AIR", "SOUTHWEST", "UNITED AIRLINES", "ALASKA AIRLINES",
        "JETBLUE", "SPIRIT", "FRONTIER", "SKYWEST", "ENVOY", "MESA", "HORIZON",
    ]
    static let commonAirlineTypeDesignators: Set<String> = [
        "A319", "A320", "A20N", "A321", "A21N", "B737", "B738", "B739", "B38M", "B39M",
        "E170", "E75L", "E75S", "CRJ2", "CRJ7", "CRJ9",
    ]
    static let commonBusinessJetTypeDesignators: Set<String> = [
        "C25A", "C25B", "C25C", "C510", "C525", "C550", "C560", "C56X", "C680", "C700",
        "CL30", "CL35", "CL60", "E50P", "E55P", "F2TH", "F900", "FA7X", "FA8X",
        "GLEX", "GLF4", "GLF5", "GLF6", "H25B", "LJ35", "LJ45", "PRM1",
    ]
    static let commonTrainerAndGATypeDesignators: Set<String> = [
        "AS50", "AS55", "B06", "B407",
        "BE20", "BE23", "BE24", "BE33", "BE35", "BE36", "BE55", "BE58", "BE65", "BE9L",
        "C150", "C152", "C172", "C177", "C180", "C182", "C185", "C206", "C207", "C208", "C210",
        "DA20", "DA40", "DA42",
        "M20P", "P28A", "P28R", "P32R", "PA24", "PA28", "PA32", "PA34", "PA44", "PA46",
        "P06T", "PC12",
        "R22", "R44", "R66",
        "SLG4", "SR20", "SR22", "S22T", "T6", "T206",
    ]
    static let commonTrainerAndGADescriptionSignals: [String] = [
        "BEECH", "BEECHCRAFT", "CESSNA", "CIRRUS", "DIAMOND", "MOONEY", "PILATUS", "PIPER", "ROBINSON", "SLING",
    ]
}

enum SampleData {
    static let now = Date()

    static let sightings: [AircraftSighting] = [
        AircraftSighting(
            id: UUID(),
            icaoHex: "A7BLCF",
            callsign: "GTI423",
            registration: "N780BA",
            typeDesignator: "BLCF",
            description: "Boeing 747-400 Dreamlifter",
            operatorName: "Atlas Air",
            latitude: 33.78,
            longitude: -118.24,
            altitudeFeet: 12_400,
            groundSpeedKnots: 318,
            headingDegrees: 274,
            squawk: nil,
            emergency: nil,
            distanceNauticalMiles: 8.2,
            seenAt: now.addingTimeInterval(-180),
            rarityBand: .alert,
            confidence: 0.94,
            reason: "Very limited modified 747 fleet, uncommon for Orange County and Los Angeles County.",
            claimState: .detected,
            collectionKey: "BLCF"
        ),
        AircraftSighting(
            id: UUID(),
            icaoHex: "AE4C17",
            callsign: "RCH321",
            registration: nil,
            typeDesignator: "C17",
            description: "Boeing C-17 Globemaster III",
            operatorName: "United States Air Force",
            latitude: 33.95,
            longitude: -118.40,
            altitudeFeet: 9_000,
            groundSpeedKnots: 260,
            headingDegrees: 83,
            squawk: nil,
            emergency: nil,
            distanceNauticalMiles: 14.9,
            seenAt: now.addingTimeInterval(-420),
            rarityBand: .incoming,
            confidence: 0.92,
            reason: "Military heavy transport away from a routine base pattern, entering the LAX corridor.",
            claimState: .detected,
            collectionKey: "C17"
        ),
        AircraftSighting(
            id: UUID(),
            icaoHex: "AEH060",
            callsign: "KNIFE07",
            registration: "17-20962",
            typeDesignator: "H60",
            description: "Sikorsky UH-60 Black Hawk",
            operatorName: "United States Army",
            latitude: 33.79,
            longitude: -118.05,
            altitudeFeet: 1_750,
            groundSpeedKnots: 90,
            headingDegrees: 141,
            squawk: nil,
            emergency: nil,
            distanceNauticalMiles: 2.0,
            seenAt: now.addingTimeInterval(-240),
            rarityBand: .nearMiss,
            confidence: 0.88,
            reason: "Globally noteworthy, but routine near the Los Alamitos base pattern.",
            claimState: .detected,
            collectionKey: "H60"
        ),
        AircraftSighting(
            id: UUID(),
            icaoHex: "AAL737",
            callsign: "AAL1204",
            registration: "N923AN",
            typeDesignator: "B738",
            description: "Boeing 737-800",
            operatorName: "American Airlines",
            latitude: 33.68,
            longitude: -117.86,
            altitudeFeet: 3_200,
            groundSpeedKnots: 210,
            headingDegrees: 196,
            squawk: nil,
            emergency: nil,
            distanceNauticalMiles: 4.1,
            seenAt: now.addingTimeInterval(-90),
            rarityBand: .routine,
            confidence: 0.89,
            reason: "Ordinary local airline traffic near SNA without a special livery or unusual operator.",
            claimState: .detected,
            collectionKey: "B738"
        )
    ]

    static let logbook: [LogbookEntry] = [
        LogbookEntry(
            id: UUID(),
            collectionKey: "T38",
            title: "Northrop T-38 Talon",
            subtitle: "Claimed over San Fernando Valley",
            firstClaimedAt: now.addingTimeInterval(-86_400 * 3),
            claimCount: 2,
            hasPhoto: true
        ),
        LogbookEntry(
            id: UUID(),
            collectionKey: "B744",
            title: "Boeing 747-400",
            subtitle: "Classic widebody sighting",
            firstClaimedAt: now.addingTimeInterval(-86_400 * 9),
            claimCount: 1,
            hasPhoto: false
        )
    ]
}
