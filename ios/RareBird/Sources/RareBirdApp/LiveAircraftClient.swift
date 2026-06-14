import CoreLocation
import Foundation

struct AircraftFeedRequest {
    var center: CLLocationCoordinate2D
    var radiusNauticalMiles: Int
}

@MainActor
protocol AircraftFeedFetching {
    func fetchAircraft(near request: AircraftFeedRequest) async throws -> [AircraftSighting]
}

struct ADSBFiAircraftFeedClient: AircraftFeedFetching {
    var session: URLSession = .shared

    func fetchAircraft(near request: AircraftFeedRequest) async throws -> [AircraftSighting] {
        let radius = max(1, min(250, request.radiusNauticalMiles))
        let url = URL(
            string: "https://opendata.adsb.fi/api/v3/lat/\(request.center.latitude)/lon/\(request.center.longitude)/dist/\(radius)"
        )!
        AppLog.info("ADSB.fi request url=\(url.absoluteString)", logger: AppLog.feed)
        var urlRequest = URLRequest(url: url)
        urlRequest.setValue("rare-bird-ios/0.1", forHTTPHeaderField: "User-Agent")
        urlRequest.timeoutInterval = 20

        let (data, response) = try await session.data(for: urlRequest)
        guard let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            AppLog.error("ADSB.fi bad response status=\(statusCode)", logger: AppLog.feed)
            throw LiveAircraftError.badResponse
        }

        let payload = try JSONDecoder().decode(ADSBFiResponse.self, from: data)
        let collectedAt = payload.now.map { value in
            Date(timeIntervalSince1970: value > 10_000_000_000 ? value / 1_000 : value)
        } ?? Date()
        let sightings = payload.aircraft
            .compactMap { $0.sighting(collectedAt: collectedAt, observerCoordinate: request.center) }
            .sorted { $0.distanceNauticalMiles < $1.distanceNauticalMiles }
        AppLog.info("ADSB.fi decoded raw=\(payload.aircraft.count) sightings=\(sightings.count)", logger: AppLog.feed)
        return sightings
    }
}

enum LiveAircraftError: LocalizedError {
    case badResponse

    var errorDescription: String? {
        "The live aircraft feed did not return a successful response."
    }
}

private struct ADSBFiResponse: Decodable {
    var now: Double?
    var aircraft: [ADSBFiAircraft]

    enum CodingKeys: String, CodingKey {
        case now = "now"
        case aircraft = "aircraft"
        case ac = "ac"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        now = try container.decodeIfPresent(Double.self, forKey: .now)
        aircraft = try container.decodeIfPresent([ADSBFiAircraft].self, forKey: .aircraft)
            ?? container.decodeIfPresent([ADSBFiAircraft].self, forKey: .ac)
            ?? []
    }
}

private struct ADSBFiAircraft: Decodable {
    var hex: String?
    var flight: String?
    var registration: String?
    var typeDesignator: String?
    var description: String?
    var operatorName: String?
    var latitude: Double?
    var longitude: Double?
    var altitude: ADSBFiAltitude?
    var groundSpeed: Double?
    var track: Double?
    var distanceNauticalMiles: Double?
    var seenSeconds: Double?
    var squawk: String?
    var emergency: String?

    enum CodingKeys: String, CodingKey {
        case hex
        case flight
        case registration = "r"
        case typeDesignator = "t"
        case description = "desc"
        case operatorName = "ownOp"
        case latitude = "lat"
        case longitude = "lon"
        case altitude = "alt_baro"
        case groundSpeed = "gs"
        case track
        case distanceNauticalMiles = "dst"
        case seenSeconds = "seen"
        case squawk
        case emergency
    }

    func sighting(collectedAt: Date, observerCoordinate: CLLocationCoordinate2D) -> AircraftSighting? {
        guard
            let hex = cleaned(hex),
            let latitude,
            let longitude
        else {
            return nil
        }

        let coordinate = CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
        let distance = distanceNauticalMiles
            ?? observerCoordinate.distanceNauticalMiles(to: coordinate)
        let type = cleaned(typeDesignator) ?? "UNK"
        let registration = cleaned(registration)
        let callsign = cleaned(flight) ?? registration ?? hex.uppercased()
        let description = cleaned(description) ?? "\(type) aircraft"
        let operatorName = cleaned(operatorName) ?? "Unknown operator"
        let heading = Int((track ?? 45).rounded())
        let seenAt = collectedAt.addingTimeInterval(-(seenSeconds ?? 0))

        return AircraftSighting(
            id: UUID(),
            icaoHex: hex,
            callsign: callsign,
            registration: registration,
            typeDesignator: type,
            description: description,
            operatorName: operatorName,
            latitude: latitude,
            longitude: longitude,
            altitudeFeet: Int((altitude?.feet ?? 0).rounded()),
            groundSpeedKnots: Int((groundSpeed ?? 0).rounded()),
            headingDegrees: heading,
            squawk: cleaned(squawk),
            emergency: cleaned(emergency),
            distanceNauticalMiles: distance,
            seenAt: seenAt,
            rarityBand: .routine,
            confidence: 0.50,
            reason: "Live ADS-B sighting awaiting rarity classification.",
            claimState: .detected,
            collectionKey: registration ?? type
        )
    }

    private func cleaned(_ value: String?) -> String? {
        guard let value else { return nil }
        let text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }
}

private enum ADSBFiAltitude: Decodable {
    case feet(Double)
    case ground

    var feet: Double {
        switch self {
        case .feet(let value): value
        case .ground: 0
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Double.self) {
            self = .feet(value)
            return
        }
        if let text = try? container.decode(String.self), text == "ground" {
            self = .ground
            return
        }
        self = .feet(0)
    }
}
