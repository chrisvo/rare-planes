import SwiftUI

struct LogbookView: View {
    @EnvironmentObject private var store: RareBirdStore

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    progressPanel
                    VStack(alignment: .leading, spacing: 8) {
                        SectionHeader(title: "Collected", subtitle: "Types and airframes claimed nearby")
                        ForEach(store.logbook) { entry in
                            LogbookEntryRow(entry: entry)
                        }
                    }
                }
                .padding(16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Logbook")
        }
    }

    private var progressPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Southern California set")
                        .font(.headline)
                    Text("\(store.logbook.count) collected")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                ZStack {
                    Circle()
                        .stroke(.quaternary, lineWidth: 8)
                    Circle()
                        .trim(from: 0, to: min(Double(store.logbook.count) / 24.0, 1))
                        .stroke(Color(red: 0.84, green: 0.27, blue: 0.19), style: StrokeStyle(lineWidth: 8, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    Text("\(store.logbook.count)/24")
                        .font(.caption.weight(.bold))
                }
                .frame(width: 64, height: 64)
            }

            Text("A quiet collection of uncommon aircraft, special airframes, and near-disappearing types seen from your region.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

struct LogbookEntryRow: View {
    var entry: LogbookEntry

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(red: 0.08, green: 0.42, blue: 0.58).opacity(0.13))
                Text(entry.collectionKey)
                    .font(.caption.weight(.bold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.65)
                    .foregroundStyle(Color(red: 0.08, green: 0.42, blue: 0.58))
                    .padding(4)
            }
            .frame(width: 54, height: 54)

            VStack(alignment: .leading, spacing: 4) {
                Text(entry.title)
                    .font(.headline)
                    .lineLimit(1)
                Text(entry.subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                HStack(spacing: 10) {
                    Label("\(entry.claimCount)", systemImage: "checkmark.seal")
                    if entry.hasPhoto {
                        Label("Photo", systemImage: "camera.fill")
                    }
                }
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            }

            Spacer()

            Text("\(entry.rarityScore)")
                .font(.title3.weight(.bold))
                .foregroundStyle(.red)
                .accessibilityLabel("Rarity score \(entry.rarityScore)")
        }
        .padding(12)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}
