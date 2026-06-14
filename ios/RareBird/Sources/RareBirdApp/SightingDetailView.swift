import PhotosUI
import SwiftUI

struct SightingDetailView: View {
    @EnvironmentObject private var store: RareBirdStore
    @Environment(\.dismiss) private var dismiss
    @State private var selectedPhoto: PhotosPickerItem?

    var sighting: AircraftSighting

    private var currentSighting: AircraftSighting {
        store.sightings.first { $0.id == sighting.id } ?? sighting
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                hero
                metrics
                reasonPanel
                liveDataPanel
                claimPanel
            }
            .padding(16)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle(currentSighting.typeDesignator)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark")
                }
                .accessibilityLabel("Close")
            }
        }
        .onChange(of: selectedPhoto) { _, newValue in
            guard newValue != nil else { return }
            store.markPhotoLogged(currentSighting)
        }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    StatusPill(band: currentSighting.rarityBand)
                    Text(currentSighting.description)
                        .font(.largeTitle.weight(.bold))
                        .lineLimit(3)
                        .minimumScaleFactor(0.72)
                    Text(currentSighting.operatorName)
                        .font(.headline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                ConfidenceRing(confidence: currentSighting.confidence, tint: currentSighting.rarityBand.tint)
            }

            HStack(spacing: 10) {
                Label(currentSighting.callsign, systemImage: "quote.bubble")
                if let registration = currentSighting.registration {
                    Label(registration, systemImage: "number")
                }
            }
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .minimumScaleFactor(0.8)
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var metrics: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
            MetricView(title: "Altitude", value: "\(currentSighting.altitudeFeet.formatted()) ft", symbolName: "arrow.up")
            MetricView(title: "Speed", value: "\(currentSighting.groundSpeedKnots) kt", symbolName: "speedometer")
            MetricView(title: "Distance", value: String(format: "%.1f nm", currentSighting.distanceNauticalMiles), symbolName: "location")
            MetricView(title: "Heading", value: "\(currentSighting.headingDegrees) deg", symbolName: "safari")
        }
    }

    private var reasonPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Rarity read", systemImage: "sparkles")
                .font(.headline)
            Text(currentSighting.reason)
                .font(.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            LabeledContent("Band", value: currentSighting.rarityBand.label)
            LabeledContent("Confidence", value: "\(Int(currentSighting.confidence * 100))%")
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var liveDataPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Live ADS-B feed", systemImage: "dot.radiowaves.left.and.right")
                .font(.headline)

            VStack(spacing: 8) {
                DetailFieldRow(title: "ICAO hex", value: currentSighting.icaoHex.uppercased())
                DetailFieldRow(title: "Callsign", value: currentSighting.callsign)
                DetailFieldRow(title: "Registration", value: currentSighting.registration)
                DetailFieldRow(title: "Type", value: currentSighting.typeDesignator)
                DetailFieldRow(title: "Operator", value: currentSighting.operatorName)
                DetailFieldRow(title: "Description", value: currentSighting.description)
                DetailFieldRow(title: "Altitude", value: "\(currentSighting.altitudeFeet.formatted()) ft")
                DetailFieldRow(title: "Ground speed", value: "\(currentSighting.groundSpeedKnots) kt")
                DetailFieldRow(title: "Heading", value: "\(currentSighting.headingDegrees) deg")
                DetailFieldRow(title: "Distance", value: String(format: "%.1f nm", currentSighting.distanceNauticalMiles))
                DetailFieldRow(title: "Squawk", value: currentSighting.squawk)
                DetailFieldRow(title: "Emergency", value: currentSighting.emergency)
                DetailFieldRow(title: "Last seen", value: lastSeenText)
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var lastSeenText: String {
        let seconds = max(0, Int(Date().timeIntervalSince(currentSighting.seenAt)))
        if seconds < 60 {
            return "\(seconds) sec ago"
        }
        let minutes = seconds / 60
        if minutes < 60 {
            return "\(minutes) min ago"
        }
        return "\(minutes / 60) hr ago"
    }

    private var claimPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(currentSighting.claimState.label, systemImage: currentSighting.claimState.symbolName)
                .font(.headline)
                .foregroundStyle(currentSighting.claimState == .detected ? .primary : currentSighting.rarityBand.tint)

            HStack(spacing: 10) {
                Button {
                    store.claim(currentSighting)
                } label: {
                    Label("Claim", systemImage: "checkmark.seal")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!currentSighting.isClaimable || currentSighting.claimState != .detected)

                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Label("Photo", systemImage: "camera")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(currentSighting.claimState == .detected)
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

private struct DetailFieldRow: View {
    var title: String
    var value: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 96, alignment: .leading)
            Text(displayValue)
                .font(.subheadline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .multilineTextAlignment(.leading)
        }
    }

    private var displayValue: String {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "Unavailable"
        }
        return value
    }
}
