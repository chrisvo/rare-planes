import SwiftUI

struct StatusPill: View {
    var band: RarityBand

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: band.symbolName)
                .font(.caption.weight(.semibold))
            Text(band.label)
                .font(.caption.weight(.semibold))
        }
        .foregroundStyle(band.tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(band.tint.opacity(0.12), in: Capsule())
    }
}

struct MetricView: View {
    var title: String
    var value: String
    var symbolName: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Image(systemName: symbolName)
                .font(.callout.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

struct ConfidenceRing: View {
    var confidence: Double
    var tint: Color

    var body: some View {
        ZStack {
            Circle()
                .stroke(.quaternary, lineWidth: 5)
            Circle()
                .trim(from: 0, to: min(1, max(0, confidence)))
                .stroke(tint, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text("\(Int(confidence * 100))")
                .font(.caption.weight(.bold))
        }
        .frame(width: 42, height: 42)
        .accessibilityLabel("Confidence \(Int(confidence * 100)) percent")
    }
}

struct SightingRow: View {
    var sighting: AircraftSighting

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(sighting.rarityBand.tint.opacity(0.14))
                Image(systemName: sighting.aircraftSymbolName)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(sighting.rarityBand.tint)
                    .rotationEffect(.degrees(sighting.aircraftIconRotationDegrees))
            }
            .frame(width: 48, height: 48)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 7) {
                    Text(sighting.displayName)
                        .font(.headline)
                        .lineLimit(1)
                        .minimumScaleFactor(0.78)
                    StatusPill(band: sighting.rarityBand)
                }
                Text(sighting.subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                RarityReadSummary(sighting: sighting, lineLimit: 3)
            }

            Spacer(minLength: 8)

            ConfidenceRing(confidence: sighting.confidence, tint: sighting.rarityBand.tint)
        }
        .padding(.vertical, 8)
    }
}

struct RarityReadSummary: View {
    var sighting: AircraftSighting
    var lineLimit: Int? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Rarity read", systemImage: "sparkles")
                .font(.caption.weight(.semibold))
                .foregroundStyle(sighting.rarityBand.tint)
            Text(sighting.reason)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(lineLimit)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, 2)
    }
}

struct EmptyStateView: View {
    var body: some View {
        ContentUnavailableView(
            "No rare aircraft nearby",
            systemImage: "binoculars",
            description: Text("Near misses and routine traffic still appear when enabled.")
        )
    }
}

struct LiveFeedWaitingView: View {
    var body: some View {
        ContentUnavailableView(
            "Waiting for live traffic",
            systemImage: "dot.radiowaves.left.and.right",
            description: Text("Claimable aircraft will appear after the first ADS-B response.")
        )
    }
}

struct LiveFeedUnavailableView: View {
    var body: some View {
        ContentUnavailableView(
            "Live feed unavailable",
            systemImage: "exclamationmark.triangle",
            description: Text("Refresh again when the ADS-B feed is reachable.")
        )
    }
}
