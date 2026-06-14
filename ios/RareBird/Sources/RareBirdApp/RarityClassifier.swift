import Foundation
#if canImport(MLXLLM) && canImport(MLXLMCommon) && canImport(Tokenizers)
import MLXLLM
import MLXLMCommon
import Tokenizers
#endif

@MainActor
protocol RarityClassifying {
    func classify(_ input: RarityInput) async throws -> RarityResult
}

struct PromptBuilder {
    func prompt(for input: RarityInput) -> String {
        let aircraft = input.aircraft
        let payload: [String: Any] = [
            "aircraft": [
                "icao_hex": aircraft.icaoHex,
                "callsign": aircraft.callsign,
                "registration": aircraft.registration ?? "",
                "type_designator": aircraft.typeDesignator,
                "description": aircraft.description,
                "operator": aircraft.operatorName,
                "lat": aircraft.latitude,
                "lon": aircraft.longitude,
                "altitude_ft": aircraft.altitudeFeet,
                "ground_speed_kt": aircraft.groundSpeedKnots,
                "heading_deg": aircraft.headingDegrees,
                "squawk": aircraft.squawk ?? "",
                "emergency": aircraft.emergency ?? "",
                "distance_nm": aircraft.distanceNauticalMiles
            ],
            "observer_context": [
                "region": "orange_county_los_angeles_southern_california",
                "current_local_area": input.observerArea,
                "distance_to_nearest_military_nm": input.distanceToNearestMilitaryNauticalMiles,
                "military_pattern": input.militaryPattern,
                "local_rules": [
                    "routine A320/737/E175/business-jet traffic near SNA/LAX/LGB/BUR/ONT is not rare without another signal",
                    "common Cessna/Piper/Cirrus/Diamond/Beech trainer and GA traffic is routine without a watchlist, emergency, or notable registration",
                    "H60 helicopters are locally common around Southern California military and training routes; promote only with an emergency, special mission callsign, notable registration, or unusual off-pattern context",
                    "ordinary Boeing 747-400 / B744 cargo traffic, including normal Kalitta-style cargo flow, is contextual rather than automatically rare; require special livery, notable registration, emergency, unusual operator, or unusual route evidence",
                    "ordinary Boeing 747-8 / B748 airline or cargo traffic is contextual rather than automatically rare; require special livery, unusual operator, notable airframe, emergency, or unusual route evidence",
                    "police, sheriff, fire, coast guard, medevac, and rescue helicopters are near-miss or routine unless paired with an emergency, unusual type, notable registration, or special incident signal",
                    "military aircraft near its base or flight-test pattern can be locally routine",
                    "the same military aircraft away from a base pattern can be locally alert-worthy",
                    "ultra-rare, emergency-squawk, special-livery, notable-registration, classic, or specialized-cargo aircraft remain alert-worthy"
                ]
            ],
            "task": "Return exactly one JSON object with keys is_rare, confidence, reason. Do not include markdown, commentary, metadata, or extra keys."
        ]
        let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        let json = data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return """
        ### System
        You are rarebirds, a strict aircraft rarity classifier for plane spotters. You must output exactly one JSON object with keys is_rare, confidence, reason. No markdown, no metadata, no extra keys.

        ### Input JSON
        \(json)

        ### Output JSON
        """
    }
}

struct MockRarityClassifier: RarityClassifying {
    func classify(_ input: RarityInput) async throws -> RarityResult {
        RarityResult(
            isRare: input.aircraft.rarityBand == .alert || input.aircraft.rarityBand == .incoming,
            confidence: input.aircraft.confidence,
            reason: input.aircraft.reason
        )
    }
}

final class LocalModelRarityClassifier: RarityClassifying {
    private let promptBuilder = PromptBuilder()

    #if canImport(MLXLLM) && canImport(MLXLMCommon) && canImport(Tokenizers)
    private let containerTask: Task<ModelContainer, Error>

    init(modelDirectory: URL? = nil) {
        containerTask = Task.detached {
            let directory = try modelDirectory ?? Self.defaultModelDirectory()
            AppLog.info("loading MLX rarity model directory=\(directory.path)", logger: AppLog.classifier)
            return try await loadModelContainer(
                from: directory,
                using: RareBirdsTokenizerLoader()
            )
        }
    }

    func classify(_ input: RarityInput) async throws -> RarityResult {
        let prompt = promptBuilder.prompt(for: input)
        let container = try await containerTask.value
        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 160, temperature: 0)
        )
        let output = try await session.respond(to: prompt)
        return try Self.parseRarityResult(from: output)
    }

    nonisolated static var isModelAvailable: Bool {
        (try? defaultModelDirectory()) != nil
    }

    nonisolated private static func defaultModelDirectory() throws -> URL {
        if let bundled = Bundle.main.url(forResource: "RareBirdsRarityModel", withExtension: nil) {
            return bundled
        }
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        ).appendingPathComponent("RareBirdsRarityModel", isDirectory: true)
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: appSupport.path, isDirectory: &isDirectory), isDirectory.boolValue {
            return appSupport
        }
        throw LocalModelError.modelNotFound
    }

    private static func parseRarityResult(from output: String) throws -> RarityResult {
        guard let start = output.firstIndex(of: "{"), let end = output.lastIndex(of: "}"), start < end else {
            return try repairRarityResult(from: output)
        }
        let json = String(output[start...end])
        guard let data = json.data(using: .utf8) else {
            return try repairRarityResult(from: output)
        }
        var result: RarityResult
        do {
            result = try JSONDecoder().decode(RarityResult.self, from: data)
        } catch {
            result = try repairRarityResult(from: output)
        }
        result.confidence = min(1, max(0, result.confidence))
        return result
    }

    private static func repairRarityResult(from output: String) throws -> RarityResult {
        guard let isRare = firstBooleanValue(in: output, keys: ["is_rare", "is_true"]) else {
            throw LocalModelError.invalidOutput(output)
        }
        let confidence = firstDoubleValue(in: output, key: "confidence") ?? (isRare ? 0.74 : 0.62)
        let reason = firstStringValue(in: output, key: "reason")
            ?? (isRare ? "Local model marked this sighting rare." : "Local model marked this sighting routine.")
        return RarityResult(isRare: isRare, confidence: min(1, max(0, confidence)), reason: reason)
    }

    private static func firstBooleanValue(in output: String, keys: [String]) -> Bool? {
        for key in keys {
            if output.range(of: "\"\(key)\"\\s*:\\s*true", options: .regularExpression) != nil {
                return true
            }
            if output.range(of: "\"\(key)\"\\s*:\\s*false", options: .regularExpression) != nil {
                return false
            }
        }
        return nil
    }

    private static func firstDoubleValue(in output: String, key: String) -> Double? {
        guard let range = output.range(
            of: "\"\(key)\"\\s*:\\s*[0-9]+(?:\\.[0-9]+)?",
            options: .regularExpression
        ) else {
            return nil
        }
        let match = output[range]
        guard let valueRange = match.range(of: "[0-9]+(?:\\.[0-9]+)?", options: .regularExpression) else {
            return nil
        }
        return Double(match[valueRange])
    }

    private static func firstStringValue(in output: String, key: String) -> String? {
        guard let range = output.range(
            of: "\"\(key)\"\\s*:\\s*\"[^\"]*",
            options: .regularExpression
        ) else {
            return nil
        }
        let match = output[range]
        guard let quote = match.lastIndex(of: "\""), quote < match.endIndex else {
            return nil
        }
        let value = String(match[match.index(after: quote)...]).trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }

    enum LocalModelError: LocalizedError {
        case modelNotFound
        case invalidOutput(String)

        var errorDescription: String? {
            switch self {
            case .modelNotFound:
                "RareBirdsRarityModel was not found in the app bundle or Application Support."
            case .invalidOutput(let output):
                "The local MLX rarity model did not return valid JSON: \(output)"
            }
        }
    }
    #else
    init(modelDirectory: URL? = nil) {}

    func classify(_ input: RarityInput) async throws -> RarityResult {
        _ = promptBuilder.prompt(for: input)
        throw LocalModelError.runtimeNotLinked
    }

    static var isModelAvailable: Bool { false }

    enum LocalModelError: LocalizedError {
        case runtimeNotLinked

        var errorDescription: String? {
            "Local model runtime is not linked yet. The app shell is wired through RarityClassifying for MLX Swift integration."
        }
    }
    #endif
}

#if canImport(MLXLMCommon) && canImport(Tokenizers)
struct RareBirdsTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return RareBirdsTokenizer(tokenizer)
    }
}

struct RareBirdsTokenizer: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
#endif

struct FallbackRarityClassifier: RarityClassifying {
    var primary: RarityClassifying
    var fallback: RarityClassifying

    func classify(_ input: RarityInput) async throws -> RarityResult {
        do {
            return try await primary.classify(input)
        } catch {
            AppLog.error("primary classifier failed; falling back error=\(error.localizedDescription)", logger: AppLog.classifier)
            return try await fallback.classify(input)
        }
    }
}

struct HybridRarityClassifier: RarityClassifying {
    var smallModel: RarityClassifying
    var adjudicator: RarityClassifying
    var adjudicationConfidenceThreshold = 0.85

    func classify(_ input: RarityInput) async throws -> RarityResult {
        let smallResult = try await smallModel.classify(input)
        guard shouldAdjudicate(input: input, smallResult: smallResult) else {
            AppLog.info(
                "small classifier accepted hex=\(input.aircraft.icaoHex) rare=\(smallResult.isRare) confidence=\(smallResult.confidence)",
                logger: AppLog.classifier
            )
            return smallResult
        }

        do {
            let adjudicated = try await adjudicator.classify(input)
            AppLog.info(
                "model adjudicated hex=\(input.aircraft.icaoHex) small_rare=\(smallResult.isRare) model_rare=\(adjudicated.isRare)",
                logger: AppLog.classifier
            )
            return reconciled(smallResult: smallResult, modelResult: adjudicated)
        } catch {
            AppLog.error(
                "model adjudication failed; using small classifier hex=\(input.aircraft.icaoHex) error=\(error.localizedDescription)",
                logger: AppLog.classifier
            )
            return smallResult
        }
    }

    private func shouldAdjudicate(input: RarityInput, smallResult: RarityResult) -> Bool {
        if smallResult.confidence < adjudicationConfidenceThreshold {
            return true
        }
        if smallResult.isRare {
            return true
        }

        let aircraft = input.aircraft
        let type = aircraft.typeDesignator.uppercased()
        let callsign = aircraft.callsign.uppercased()
        let searchable = [
            aircraft.callsign,
            aircraft.registration ?? "",
            aircraft.description,
            aircraft.operatorName
        ].joined(separator: " ").uppercased()

        if ["7500", "7600", "7700"].contains(aircraft.squawk ?? "") {
            return true
        }
        if highValueTypeDesignators.contains(type) {
            return true
        }
        if highValueCallsignPrefixes.contains(where: { callsign.hasPrefix($0) }) {
            return true
        }
        return highValueTextSignals.contains { searchable.contains($0) }
    }

    private func reconciled(smallResult: RarityResult, modelResult: RarityResult) -> RarityResult {
        if smallResult.isRare == modelResult.isRare {
            return RarityResult(
                isRare: modelResult.isRare,
                confidence: max(smallResult.confidence, modelResult.confidence),
                reason: modelResult.reason.isEmpty ? smallResult.reason : modelResult.reason
            )
        }

        if modelResult.confidence >= 0.70 || smallResult.confidence < adjudicationConfidenceThreshold {
            return modelResult
        }
        return smallResult
    }

    private let highValueTypeDesignators: Set<String> = [
        "A124", "A225", "A306", "A3ST", "BLCF", "B741", "B742", "B743", "B744", "B748",
        "C5", "C17", "C30J", "DC10", "DC3", "DC6", "DC8", "IL62", "IL76", "L101", "MD11",
        "MD80", "MD81", "MD82", "MD83", "MD87", "MD88", "MD90"
    ]

    private let highValueCallsignPrefixes = [
        "AF", "ASY", "EVAC", "GUARD", "JOSA", "NASA", "NAVY", "RCH", "REACH", "RESCUE", "VENUS"
    ]

    private let highValueTextSignals = [
        "ANTONOV", "BELUGA", "DREAMLIFTER", "NASA", "VINTAGE", "WARBIRD", "SPECIAL LIVERY",
        "RESCUE", "EVAC", "MEDEVAC"
    ]
}

final class DeferredRarityClassifier: RarityClassifying {
    private let factory: () -> RarityClassifying
    private var resolved: RarityClassifying?

    init(_ factory: @escaping () -> RarityClassifying) {
        self.factory = factory
    }

    func classify(_ input: RarityInput) async throws -> RarityResult {
        let classifier: RarityClassifying
        if let resolved {
            classifier = resolved
        } else {
            let created = factory()
            resolved = created
            classifier = created
        }
        return try await classifier.classify(input)
    }
}

struct NetworkRarityClassifier: RarityClassifying {
    var endpoint: URL = {
        #if targetEnvironment(simulator)
        URL(string: "http://127.0.0.1:8765/classify")!
        #else
        URL(string: "http://192.168.1.33:8765/classify")!
        #endif
    }()

    func classify(_ input: RarityInput) async throws -> RarityResult {
        AppLog.info("network classifier POST endpoint=\(endpoint.absoluteString) hex=\(input.aircraft.icaoHex)", logger: AppLog.classifier)
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 4
        request.httpBody = try JSONEncoder().encode(input)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            AppLog.error("network classifier bad response status=\(statusCode)", logger: AppLog.classifier)
            throw NetworkClassifierError.badResponse
        }
        let result = try JSONDecoder().decode(RarityResult.self, from: data)
        AppLog.info("network classifier decoded hex=\(input.aircraft.icaoHex) rare=\(result.isRare)", logger: AppLog.classifier)
        return result
    }

    enum NetworkClassifierError: LocalizedError {
        case badResponse

        var errorDescription: String? {
            "The local MLX rarity server did not return a successful response."
        }
    }
}
