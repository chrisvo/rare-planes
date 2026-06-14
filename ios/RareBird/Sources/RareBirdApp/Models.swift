import Foundation
import MapKit
import SwiftUI

struct ObservedAreaState: Equatable {
    private(set) var center: CLLocationCoordinate2D
    private(set) var previewCenter: CLLocationCoordinate2D?

    init(center: CLLocationCoordinate2D) {
        self.center = center
    }

    var canSearchPreviewArea: Bool {
        guard let previewCenter else { return false }
        return center.distanceNauticalMiles(to: previewCenter) >= 0.25
    }

    mutating func previewMapCenter(_ coordinate: CLLocationCoordinate2D) {
        previewCenter = coordinate
    }

    @discardableResult
    mutating func searchPreviewArea() -> CLLocationCoordinate2D {
        guard let previewCenter else { return center }
        center = previewCenter
        self.previewCenter = nil
        return center
    }

    mutating func reset(to coordinate: CLLocationCoordinate2D) {
        center = coordinate
        previewCenter = nil
    }

    static func == (lhs: ObservedAreaState, rhs: ObservedAreaState) -> Bool {
        coordinatesMatch(lhs.center, rhs.center)
            && coordinatesMatch(lhs.previewCenter, rhs.previewCenter)
    }

    private static func coordinatesMatch(_ lhs: CLLocationCoordinate2D?, _ rhs: CLLocationCoordinate2D?) -> Bool {
        switch (lhs, rhs) {
        case (.none, .none):
            return true
        case let (.some(lhs), .some(rhs)):
            return lhs.latitude == rhs.latitude && lhs.longitude == rhs.longitude
        default:
            return false
        }
    }
}

extension CLLocationCoordinate2D {
    func distanceNauticalMiles(to other: CLLocationCoordinate2D) -> Double {
        let origin = CLLocation(latitude: latitude, longitude: longitude)
        let destination = CLLocation(latitude: other.latitude, longitude: other.longitude)
        return origin.distance(from: destination) / 1_852
    }
}

enum RarityBand: String, Codable, CaseIterable, Identifiable {
    case alert
    case incoming
    case nearMiss
    case routine

    var id: String { rawValue }

    var label: String {
        switch self {
        case .alert: "Rare now"
        case .incoming: "Incoming"
        case .nearMiss: "Near miss"
        case .routine: "Routine"
        }
    }

    var symbolName: String {
        switch self {
        case .alert: "sparkles"
        case .incoming: "arrow.down.left.circle"
        case .nearMiss: "scope"
        case .routine: "circle"
        }
    }

    var tint: Color {
        switch self {
        case .alert: Color(red: 0.84, green: 0.27, blue: 0.19)
        case .incoming: Color(red: 0.08, green: 0.42, blue: 0.58)
        case .nearMiss: Color(red: 0.60, green: 0.47, blue: 0.18)
        case .routine: Color.secondary
        }
    }
}

enum ClaimState: String, Codable {
    case detected
    case claimed
    case photoLogged

    var label: String {
        switch self {
        case .detected: "Detected"
        case .claimed: "Claimed"
        case .photoLogged: "Photo logged"
        }
    }

    var symbolName: String {
        switch self {
        case .detected: "record.circle"
        case .claimed: "checkmark.seal.fill"
        case .photoLogged: "camera.fill"
        }
    }
}

enum LiveFeedState: Equatable {
    case idle
    case loading
    case available
    case unavailable
}

struct AircraftSighting: Identifiable, Codable, Equatable, Hashable {
    var id: UUID
    var icaoHex: String
    var callsign: String
    var registration: String?
    var typeDesignator: String
    var description: String
    var operatorName: String
    var latitude: Double
    var longitude: Double
    var altitudeFeet: Int
    var groundSpeedKnots: Int
    var headingDegrees: Int
    var squawk: String?
    var emergency: String?
    var distanceNauticalMiles: Double
    var seenAt: Date
    var rarityBand: RarityBand
    var confidence: Double
    var reason: String
    var claimState: ClaimState
    var collectionKey: String

    var coordinate: CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }

    var displayName: String {
        if let registration, !registration.isEmpty {
            "\(typeDesignator) \(registration)"
        } else if !callsign.isEmpty {
            "\(typeDesignator) \(callsign)"
        } else {
            typeDesignator
        }
    }

    var subtitle: String {
        "\(description) - \(operatorName)"
    }

    var isClaimable: Bool {
        rarityBand == .alert || rarityBand == .incoming
    }

    var isHelicopter: Bool {
        let type = typeDesignator.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let searchableText = [typeDesignator, description, operatorName]
            .joined(separator: " ")
            .uppercased()

        if Self.helicopterTypeDesignators.contains(type) {
            return true
        }

        return Self.helicopterTextSignals.contains { searchableText.contains($0) }
    }

    var aircraftSymbolName: String {
        isHelicopter ? "helicopter" : "airplane"
    }

    var aircraftIconRotationDegrees: Double {
        isHelicopter ? 0 : Double(headingDegrees) - 45
    }

    private static let helicopterTypeDesignators: Set<String> = [
        "A109", "A119", "A139", "A169", "A189",
        "AS50", "AS55", "B06", "B407", "B412", "B429",
        "EC30", "EC35", "EC45", "H47", "H53", "H60", "H500",
        "R22", "R44", "R66", "S61", "S76", "S92", "UH1", "UH60"
    ]

    private static let helicopterTextSignals = [
        "HELICOPTER", "ROTORCRAFT", "BLACK HAWK", "BLACKHAWK",
        "SIKORSKY", "ROBINSON R-", "BELL 407", "BELL 412", "BELL 429",
        "AIRBUS HELICOPTERS", "EUROCOPTER"
    ]
}

struct LogbookEntry: Identifiable, Codable, Equatable {
    let id: UUID
    var collectionKey: String
    var title: String
    var subtitle: String
    var firstClaimedAt: Date
    var claimCount: Int
    var hasPhoto: Bool

    var rarityScore: Int {
        min(99, 40 + claimCount * 7 + (hasPhoto ? 12 : 0))
    }
}

struct UserSettings: Codable, Equatable {
    var alertRadiusNauticalMiles: Double = 10
    var includeIncoming: Bool = true
    var showNearMisses: Bool = true
    var showRoutineAircraft: Bool = true
    var claimableOnlyMode: Bool = false
    var pollingIntervalSeconds: Double = 5
    var classificationIntervalSeconds: Double = 60
    var usesLocalModelBridge: Bool = true
    var rareBirdNotificationsEnabled: Bool = false
    var quietHoursEnabled: Bool = true
    var modelName: String = "Rare Planes 0.16 MB on-device rarity model"

    init() {}

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        alertRadiusNauticalMiles = try container.decodeIfPresent(Double.self, forKey: .alertRadiusNauticalMiles) ?? 10
        includeIncoming = try container.decodeIfPresent(Bool.self, forKey: .includeIncoming) ?? true
        showNearMisses = try container.decodeIfPresent(Bool.self, forKey: .showNearMisses) ?? true
        showRoutineAircraft = try container.decodeIfPresent(Bool.self, forKey: .showRoutineAircraft) ?? true
        claimableOnlyMode = try container.decodeIfPresent(Bool.self, forKey: .claimableOnlyMode) ?? false
        pollingIntervalSeconds = try container.decodeIfPresent(Double.self, forKey: .pollingIntervalSeconds) ?? 5
        classificationIntervalSeconds = try container.decodeIfPresent(Double.self, forKey: .classificationIntervalSeconds) ?? 60
        usesLocalModelBridge = try container.decodeIfPresent(Bool.self, forKey: .usesLocalModelBridge) ?? true
        rareBirdNotificationsEnabled = try container.decodeIfPresent(Bool.self, forKey: .rareBirdNotificationsEnabled) ?? false
        quietHoursEnabled = try container.decodeIfPresent(Bool.self, forKey: .quietHoursEnabled) ?? true
        modelName = try container.decodeIfPresent(String.self, forKey: .modelName) ?? "Rare Planes 0.16 MB on-device rarity model"
    }
}

struct RarityInput: Codable, Equatable {
    var aircraft: AircraftSighting
    var observerArea: String
    var distanceToNearestMilitaryNauticalMiles: Double
    var militaryPattern: String
}

struct RarityResult: Codable, Equatable {
    var isRare: Bool
    var confidence: Double
    var reason: String

    enum CodingKeys: String, CodingKey {
        case isRare = "is_rare"
        case confidence
        case reason
    }
}
