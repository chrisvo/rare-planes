import Foundation

struct TextRarityClassifier: RarityClassifying {
    private let model: TextClassifierModel

    init(resourceName: String = "RarityTextClassifier") throws {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "json") else {
            throw TextClassifierError.modelNotFound(resourceName)
        }
        let data = try Data(contentsOf: url)
        model = try JSONDecoder().decode(TextClassifierModel.self, from: data)
    }

    func classify(_ input: RarityInput) async throws -> RarityResult {
        let rawProbability = model.rareProbability(for: input)
        let probability = PolicyAdjuster.adjustedRareProbability(rawProbability, for: input)
        let isRare = probability >= model.threshold
        return RarityResult(
            isRare: isRare,
            confidence: model.confidence(probability: probability, isRare: isRare),
            reason: ReasonBuilder.reason(for: input, isRare: isRare)
        )
    }

    enum TextClassifierError: LocalizedError {
        case modelNotFound(String)

        var errorDescription: String? {
            switch self {
            case .modelNotFound(let resourceName):
                "Bundled text classifier resource \(resourceName).json was not found."
            }
        }
    }
}

private struct TextClassifierModel: Decodable {
    let threshold: Double
    let alpha: Double
    let vocabulary: Set<String>
    let classDocumentCounts: [String: Int]
    let classTotalFeatures: [String: Int]
    let classFeatureCounts: [String: [String: Int]]

    enum CodingKeys: String, CodingKey {
        case threshold
        case alpha
        case vocabulary
        case classDocumentCounts = "class_document_counts"
        case classTotalFeatures = "class_total_features"
        case classFeatureCounts = "class_feature_counts"
    }

    func rareProbability(for input: RarityInput) -> Double {
        let features = FeatureBuilder.features(for: input)
        let falseScore = logScore(label: "false", features: features)
        let trueScore = logScore(label: "true", features: features)
        let maxScore = max(falseScore, trueScore)
        let rareScore = exp(trueScore - maxScore)
        let commonScore = exp(falseScore - maxScore)
        return rareScore / (rareScore + commonScore)
    }

    func confidence(probability: Double, isRare: Bool) -> Double {
        let raw = isRare ? probability : 1 - probability
        return min(0.99, max(0.55, raw))
    }

    private func logScore(label: String, features: [String: Int]) -> Double {
        let totalDocuments = Double((classDocumentCounts["false"] ?? 0) + (classDocumentCounts["true"] ?? 0))
        let labelDocuments = Double(classDocumentCounts[label] ?? 0)
        var score = log((labelDocuments + alpha) / (totalDocuments + 2 * alpha))
        let vocabularySize = Double(max(vocabulary.count, 1))
        let totalFeatures = Double(classTotalFeatures[label] ?? 0)
        let denominator = totalFeatures + alpha * vocabularySize
        let counts = classFeatureCounts[label] ?? [:]

        for (feature, count) in features where vocabulary.contains(feature) {
            let numerator = Double(counts[feature] ?? 0) + alpha
            score += Double(count) * log(numerator / denominator)
        }
        return score
    }
}

private enum PolicyAdjuster {
    static func adjustedRareProbability(_ probability: Double, for input: RarityInput) -> Double {
        if hasHardAlertSignal(input) {
            return max(probability, 0.95)
        }
        let type = input.aircraft.typeDesignator.uppercased()
        if rareTypeDesignators.contains(type) {
            return max(probability, 0.95)
        }
        if contextualLonghaulTypes.contains(type) {
            return min(probability, 0.05)
        }
        return probability
    }

    private static func hasHardAlertSignal(_ input: RarityInput) -> Bool {
        let aircraft = input.aircraft
        let squawk = aircraft.squawk ?? ""
        let emergency = (aircraft.emergency ?? "").lowercased()
        if ["7500", "7600", "7700"].contains(squawk) || !["", "none", "null"].contains(emergency) {
            return true
        }
        if specialRegistrations.contains((aircraft.registration ?? "").uppercased()) {
            return true
        }
        let searchable = [
            aircraft.callsign,
            aircraft.registration ?? "",
            aircraft.description,
            aircraft.operatorName
        ].joined(separator: " ").lowercased()
        return specialMissionTerms.contains { searchable.contains($0) }
            || watchlistTerms.contains { searchable.contains($0) }
    }

    private static let contextualLonghaulTypes: Set<String> = [
        "A332", "A333", "A339", "A359", "B744", "B748", "B788", "B789", "B78X", "B77W"
    ]

    private static let rareTypeDesignators: Set<String> = [
        "A124", "A225", "A3ST", "A306", "A342", "A343", "A345", "A346",
        "B1", "B2", "B52", "B703", "B712", "B721", "B722", "B741", "B742", "B743", "B753", "BLCF",
        "CONI", "CVLP", "CVLT", "DC10", "DC3", "DC6", "DC8", "IL62", "IL76", "L101",
        "MD11", "MD80", "MD81", "MD82", "MD83", "MD87", "MD88", "MD90"
    ]

    private static let specialRegistrations: Set<String> = ["D-ABYN", "A7-BEG", "B-LRJ"]

    private static let specialMissionTerms = [
        "rescue", "evac", "medevac", "lifeguard", "search-and-rescue", "special incident"
    ]

    private static let watchlistTerms = [
        "vintage", "warbird", "restoration", "special livery", "special-livery", "watchlist"
    ]
}

private enum FeatureBuilder {
    static func features(for input: RarityInput) -> [String: Int] {
        var features: [String: Int] = [:]
        let aircraft = input.aircraft

        addTextFeatures(to: &features, prefix: "aircraft.type_designator", value: aircraft.typeDesignator)
        addTextFeatures(to: &features, prefix: "aircraft.description", value: aircraft.description)
        addTextFeatures(to: &features, prefix: "aircraft.operator", value: aircraft.operatorName)
        addTextFeatures(to: &features, prefix: "aircraft.callsign", value: aircraft.callsign)
        addTextFeatures(to: &features, prefix: "aircraft.registration", value: aircraft.registration)
        addTextFeatures(to: &features, prefix: "aircraft.squawk", value: aircraft.squawk)
        addTextFeatures(to: &features, prefix: "aircraft.emergency", value: aircraft.emergency)

        addTextFeatures(to: &features, prefix: "context.current_local_area", value: input.observerArea)
        addTextFeatures(to: &features, prefix: "context.nearest_airport", value: nearestAirport(for: input.observerArea))
        addTextFeatures(to: &features, prefix: "context.nearest_military_area", value: nearestMilitaryArea(for: input.observerArea))
        addTextFeatures(to: &features, prefix: "context.military_pattern", value: input.militaryPattern)
        addTextFeatures(to: &features, prefix: "context.region", value: "orange_county_los_angeles_southern_california")
        let frequency = localFrequencyContext(for: input)
        addTextFeatures(to: &features, prefix: "context.frequency_class", value: frequency.className)
        addTextFeatures(to: &features, prefix: "context.alert_policy", value: frequency.alertPolicy)

        addBucketFeature(to: &features, name: "aircraft.distance_nm_bucket", value: aircraft.distanceNauticalMiles)
        addBucketFeature(to: &features, name: "aircraft.distance_to_nearest_military_nm_bucket", value: input.distanceToNearestMilitaryNauticalMiles)
        addBucketFeature(to: &features, name: "aircraft.altitude_ft_bucket", value: Double(aircraft.altitudeFeet))
        addBucketFeature(to: &features, name: "aircraft.ground_speed_kt_bucket", value: Double(aircraft.groundSpeedKnots))

        if let prefix = callsignPrefix(aircraft.callsign) {
            increment(&features, key: "callsign_prefix=\(prefix)", by: 2)
        }

        let squawk = aircraft.squawk ?? ""
        let emergency = (aircraft.emergency ?? "").lowercased()
        let hardEmergency = ["7500", "7600", "7700"].contains(squawk) || !["", "none", "null"].contains(emergency)
        if hardEmergency {
            increment(&features, key: "signal.emergency", by: 10)
            increment(&features, key: "signal.emergency_squawk=\(squawk)", by: 10)
        }

        let registration = (aircraft.registration ?? "").uppercased()
        let specialRegistration = ["D-ABYN", "A7-BEG", "B-LRJ"].contains(registration)
        if specialRegistration {
            increment(&features, key: "signal.special_registration", by: 10)
            increment(&features, key: "signal.special_registration=\(registration)", by: 10)
        }

        let searchable = [
            aircraft.callsign,
            aircraft.registration ?? "",
            aircraft.description,
            aircraft.operatorName
        ].joined(separator: " ").lowercased()
        for term in ["rescue", "evac", "medevac", "lifeguard", "search-and-rescue", "special incident"] where searchable.contains(term) {
            increment(&features, key: "signal.special_mission", by: 8)
            increment(&features, key: "signal.special_mission=\(term)", by: 4)
        }
        for term in ["vintage", "warbird", "restoration", "special livery", "special-livery", "watchlist"] where searchable.contains(term) {
            increment(&features, key: "signal.watchlist", by: 8)
            increment(&features, key: "signal.watchlist=\(term)", by: 4)
        }
        if ["B744", "B748"].contains(aircraft.typeDesignator.uppercased()), !hardEmergency, !specialRegistration {
            increment(&features, key: "signal.contextual_widebody_suppressed", by: 12)
            increment(&features, key: "signal.contextual_widebody_suppressed=\(aircraft.typeDesignator.uppercased())", by: 8)
        }
        let type = aircraft.typeDesignator.uppercased()
        if isUnknownType(type), !hardEmergency, !specialRegistration {
            increment(&features, key: "signal.unknown_type_suppressed", by: 12)
            increment(&features, key: "signal.unknown_type_suppressed=\(type)", by: 8)
        }
        if isCommonGA(type) || containsCommonGAText(searchable) {
            increment(&features, key: "signal.common_ga_suppressed", by: 12)
            increment(&features, key: "signal.common_ga_suppressed=\(type)", by: 8)
        }

        return features
    }

    private static func addTextFeatures(to features: inout [String: Int], prefix: String, value: String?) {
        guard let text = value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !text.isEmpty else {
            return
        }
        increment(&features, key: "\(prefix)=\(text)", by: 3)
        let tokens = tokenize(text)
        for token in tokens {
            increment(&features, key: "\(prefix):\(token)", by: 1)
        }
        for index in 0..<max(tokens.count - 1, 0) {
            increment(&features, key: "\(prefix):\(tokens[index])_\(tokens[index + 1])", by: 1)
        }
    }

    private static func addBucketFeature(to features: inout [String: Int], name: String, value: Double) {
        let bucket = Int(floor(value / 10) * 10)
        increment(&features, key: "\(name)=\(bucket)", by: 1)
    }

    private static func increment(_ features: inout [String: Int], key: String, by amount: Int) {
        features[key, default: 0] += amount
    }

    private static func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        for scalar in text.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                current.unicodeScalars.append(scalar)
            } else if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        }
        if !current.isEmpty {
            tokens.append(current)
        }
        return tokens
    }

    private static func callsignPrefix(_ callsign: String) -> String? {
        let prefix = callsign.uppercased().prefix { $0.isLetter }
        return prefix.isEmpty ? nil : String(prefix)
    }

    private static func nearestAirport(for area: String) -> String? {
        let lower = area.lowercased()
        if lower.contains("lax") || lower.contains("downtown") { return "LAX" }
        if lower.contains("long beach") { return "LGB" }
        if lower.contains("valley") { return "VNY" }
        if lower.contains("los alamitos") { return "SLI" }
        if lower.contains("pendleton") { return "NFG" }
        if lower.contains("march") { return "RIV" }
        if lower.contains("palmdale") { return "PMD" }
        if lower.contains("edwards") { return "EDW" }
        return "SNA"
    }

    private static func nearestMilitaryArea(for area: String) -> String? {
        let lower = area.lowercased()
        if lower.contains("pendleton") { return "Camp Pendleton / MCAS Camp Pendleton" }
        if lower.contains("march") { return "March Air Reserve Base" }
        if lower.contains("palmdale") || lower.contains("valley") { return "Plant 42 / Palmdale flight test corridor" }
        if lower.contains("edwards") { return "Edwards Air Force Base" }
        return "Joint Forces Training Base Los Alamitos"
    }

    private static func localFrequencyContext(for input: RarityInput) -> (className: String, alertPolicy: String) {
        let type = input.aircraft.typeDesignator.uppercased()
        let militaryTypes: Set<String> = ["A10", "B1", "B2", "B52", "C5", "C17", "C30J", "C130", "E2", "E3", "E6", "F15", "F16", "F18", "F22", "F35", "H47", "H53", "H60", "KC10", "KC135", "KC46", "P3", "P8", "T38", "U2", "V22"]
        let commonAirlineTypes: Set<String> = ["A320", "A321", "A20N", "A21N", "B737", "B738", "B739", "B38M", "B39M", "E75L"]
        let commonGATypes = commonGATypeDesignators
        let prefix = callsignPrefix(input.aircraft.callsign)
        let commonAirlinePrefixes: Set<String> = ["AAL", "ASA", "DAL", "FFT", "JBU", "SKW", "SWA", "UAL", "UPS", "FDX"]

        if militaryTypes.contains(type), input.militaryPattern == "base_pattern" {
            return ("routine_near_military_base_or_test_range", "suppress_alert_unless_special_livery_emergency_or_unique_airframe")
        }
        if militaryTypes.contains(type) {
            return ("uncommon_military_away_from_base_pattern", "alert")
        }
        if commonAirlineTypes.contains(type) || prefix.map(commonAirlinePrefixes.contains) == true {
            return ("common_local_civil_traffic", "suppress_alert_unless_special_livery_emergency_or_unique_airframe")
        }
        if commonGATypes.contains(type) {
            return ("common_local_ga_training_or_private_traffic", "suppress_alert_unless_special_mission_emergency_vintage_or_watchlist")
        }
        if isUnknownType(type) {
            return ("unknown_type_unidentified_adsb_target", "suppress_alert_unless_special_mission_emergency_or_identified_rare_aircraft")
        }
        return ("unknown_or_contextual", "use_aircraft_rarity_signals")
    }

    private static func isUnknownType(_ type: String) -> Bool {
        ["UNK", "UNKNOWN"].contains(type)
    }

    private static func isCommonGA(_ type: String) -> Bool {
        commonGATypeDesignators.contains(type)
    }

    private static func containsCommonGAText(_ text: String) -> Bool {
        commonGADescriptionSignals.contains { text.contains($0) }
    }

    private static let commonGATypeDesignators: Set<String> = [
        "AS50", "AS55", "B06", "B407",
        "BE20", "BE23", "BE24", "BE33", "BE35", "BE36", "BE55", "BE58", "BE65", "BE9L",
        "C150", "C152", "C172", "C177", "C180", "C182", "C185", "C206", "C207", "C208", "C210",
        "DA20", "DA40", "DA42",
        "M20P", "P28A", "P28R", "P32R", "PA24", "PA28", "PA32", "PA34", "PA44", "PA46",
        "P06T", "PC12",
        "R22", "R44", "R66",
        "SLG4", "SR20", "SR22", "S22T", "T6", "T206",
    ]

    private static let commonGADescriptionSignals: [String] = [
        "beech", "beechcraft", "cessna", "cirrus", "diamond", "mooney", "pilatus", "piper", "robinson", "sling",
    ]
}

private enum ReasonBuilder {
    static func reason(for input: RarityInput, isRare: Bool) -> String {
        let aircraft = input.aircraft
        let type = aircraft.typeDesignator.uppercased()
        let squawk = aircraft.squawk ?? ""
        let registration = (aircraft.registration ?? "").uppercased()
        let searchable = [
            aircraft.callsign,
            aircraft.registration ?? "",
            aircraft.description,
            aircraft.operatorName
        ].joined(separator: " ").lowercased()

        if isRare {
            if ["7500", "7600", "7700"].contains(squawk) {
                return "\(squawk) emergency squawk makes this alert-worthy."
            }
            if ["D-ABYN", "A7-BEG", "B-LRJ"].contains(registration) {
                return "\(registration) is a known notable airframe."
            }
            if searchable.contains("rescue") || searchable.contains("evac") {
                return "Explicit rescue or evacuation signal makes this alert-worthy."
            }
            if input.militaryPattern != "base_pattern", isMilitary(type) {
                return "\(aircraft.description) is away from a routine local military/base pattern."
            }
            if isSpecialCargo(type) || searchable.contains("beluga") || searchable.contains("dreamlifter") || searchable.contains("antonov") {
                return "\(aircraft.description) is a specialized cargo or low-production type that is uncommon in this region."
            }
            if isClassicOrDisappearing(type) {
                return "\(aircraft.description) is a classic or disappearing type with strong spotter value."
            }
            if isRareType(type) {
                return "\(aircraft.description) is uncommon for local OC/LA traffic and has a rare type signal."
            }
            if isPublicSafety(searchable) {
                return "\(aircraft.description) has a public-safety signal plus enough context to make it alert-worthy."
            }
            return "\(aircraft.description) has rarity signals for this region."
        }

        if ["B744", "B748"].contains(type) {
            return "Routine \(type) traffic is contextual here and needs a stronger signal."
        }
        if isContextualLonghaul(type) {
            return "Routine \(type) long-haul traffic is contextual here and needs a stronger signal."
        }
        if input.militaryPattern == "base_pattern", isMilitary(type) {
            return "\(aircraft.description) is locally expected in this base or training pattern."
        }
        if isCommonGA(type) {
            return "\(aircraft.description) is common local GA without an emergency, watchlist, or special-incident signal."
        }
        if ["UNK", "UNKNOWN"].contains(type) {
            return "Unknown ADS-B type is not rare by itself; it needs an emergency, special-mission, notable-registration, or identified rare-aircraft signal."
        }
        if isPublicSafety(searchable) {
            return "Public-safety traffic is not rare by itself without an emergency, unusual type, notable registration, or special incident."
        }
        return "No strong rare-aircraft signal found for this local context."
    }

    private static func isMilitary(_ type: String) -> Bool {
        ["A10", "B1", "B2", "B52", "C5", "C17", "C30J", "C130", "E2", "E3", "E6", "F15", "F16", "F18", "F22", "F35", "H47", "H53", "H60", "KC10", "KC135", "KC46", "P3", "P8", "T38", "U2", "V22"].contains(type)
    }

    private static func isCommonGA(_ type: String) -> Bool {
        [
            "AS50", "AS55", "B06", "B407",
            "BE20", "BE23", "BE24", "BE33", "BE35", "BE36", "BE55", "BE58", "BE65", "BE9L",
            "C150", "C152", "C172", "C177", "C180", "C182", "C185", "C206", "C207", "C208", "C210",
            "DA20", "DA40", "DA42",
            "M20P", "P28A", "P28R", "P32R", "PA24", "PA28", "PA32", "PA34", "PA44", "PA46",
            "R22", "R44", "R66",
            "SR20", "SR22", "T206",
        ].contains(type)
    }

    private static func isPublicSafety(_ text: String) -> Bool {
        ["sheriff", "police", "patrol", "cal fire", " fire", "civil air patrol", "county"].contains { text.contains($0) }
    }

    private static func isSpecialCargo(_ type: String) -> Bool {
        ["A124", "A225", "A3ST", "A337", "BLCF"].contains(type)
    }

    private static func isClassicOrDisappearing(_ type: String) -> Bool {
        ["B741", "B742", "B743", "B753", "DC10", "DC3", "DC6", "DC8", "IL62", "IL76", "L101", "MD11", "MD80", "MD81", "MD82", "MD83", "MD87", "MD88", "MD90"].contains(type)
    }

    private static func isContextualLonghaul(_ type: String) -> Bool {
        ["A332", "A333", "A339", "A359", "B788", "B789", "B78X", "B77W"].contains(type)
    }

    private static func isRareType(_ type: String) -> Bool {
        isSpecialCargo(type)
            || isClassicOrDisappearing(type)
            || isMilitary(type)
            || ["A10", "A306", "A342", "A343", "A345", "A346", "B1", "B2", "B52", "B703", "B712", "B721", "B722", "CONI", "CVLP", "CVLT", "F15", "F16", "F18", "F22", "F35"].contains(type)
    }
}
