import MapKit
import SwiftUI

struct NowView: View {
    private static let defaultFeedCenter = CLLocationCoordinate2D(latitude: 33.84, longitude: -118.12)

    @EnvironmentObject private var store: RareBirdStore
    @StateObject private var locationManager = UserLocationManager()
    @State private var cameraPosition = MapCameraPosition.region(
        MKCoordinateRegion(
            center: defaultFeedCenter,
            span: MKCoordinateSpan(latitudeDelta: 0.75, longitudeDelta: 0.95)
        )
    )
    @State private var observedArea = ObservedAreaState(center: defaultFeedCenter)
    @State private var hasCenteredOnUserLocation = false
    @State private var hasFetchedForUserLocation = false
    @State private var isMapExpanded = false
    @State private var lastPollClassificationAt: Date?
    @State private var detailSighting: AircraftSighting?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    mapPanel
                    primarySightings
                    if store.settings.showNearMisses && !nearMissSightings.isEmpty {
                        nearMisses
                    }
                    if store.settings.showRoutineAircraft && !routineSightings.isEmpty {
                        liveTraffic
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 28)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Rare Planes")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await refreshLiveSightings() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh")
                }
            }
            .task {
                await runLivePolling()
            }
            .onAppear {
                locationManager.activate()
                frameClaimableRadius(animated: false)
            }
            .onChange(of: locationManager.currentFix) { _, fix in
                guard let fix, !hasCenteredOnUserLocation else { return }
                observedArea.reset(to: fix.coordinate)
                centerMap(on: fix.coordinate)
                hasCenteredOnUserLocation = true
                guard !hasFetchedForUserLocation else { return }
                hasFetchedForUserLocation = true
            }
            .onChange(of: store.settings.alertRadiusNauticalMiles) { _, _ in
                frameClaimableRadius(animated: true)
            }
            .navigationDestination(item: $detailSighting) { sighting in
                SightingDetailView(sighting: sighting)
            }
            .onChange(of: store.selectedSightingID) { _, selectedID in
                guard selectedID == nil else { return }
                detailSighting = nil
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Label("OC + LA", systemImage: "location.circle")
                Label("\(Int(store.settings.alertRadiusNauticalMiles)) nm", systemImage: "scope")
                Label("\(claimableSightings.count) active", systemImage: "sparkles")
                Label("\(store.logbook.count) logged", systemImage: "book.closed")
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
        }
        .padding(.top, 8)
    }

    private var mapPanel: some View {
        Map(position: $cameraPosition) {
            MapCircle(center: claimableRadiusCenter, radius: claimableRadiusMeters)
                .foregroundStyle(Color.accentColor.opacity(0.10))
                .stroke(Color.accentColor.opacity(0.58), lineWidth: 2)

            if locationManager.canShowUserLocation {
                UserAnnotation()
            }

            ForEach(visibleSightings) { sighting in
                Annotation(sighting.typeDesignator, coordinate: sighting.coordinate) {
                    Button {
                        withAnimation(.snappy(duration: 0.2)) {
                            store.selectedSightingID = sighting.id
                        }
                    } label: {
                        Image(systemName: sighting.aircraftSymbolName)
                            .font(.callout.weight(.bold))
                            .foregroundStyle(.white)
                            .padding(9)
                            .background(sighting.rarityBand.tint, in: Circle())
                            .rotationEffect(.degrees(sighting.aircraftIconRotationDegrees))
                            .shadow(radius: 3, y: 2)
                    }
                    .accessibilityLabel(sighting.displayName)
                }
            }
        }
        .onMapCameraChange(frequency: .onEnd) { context in
            observedArea.previewMapCenter(context.region.center)
        }
        .mapStyle(.standard(elevation: .realistic))
        .frame(height: isMapExpanded ? 520 : 250)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .onTapGesture {
            closeMapCallout()
        }
        .overlay(alignment: .topLeading) {
            Label(store.feedStatus, systemImage: "dot.radiowaves.left.and.right")
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.regularMaterial, in: Capsule())
                .padding(10)
        }
        .overlay(alignment: .topTrailing) {
            HStack(spacing: 8) {
                Button {
                    withAnimation(.snappy(duration: 0.25)) {
                        isMapExpanded.toggle()
                    }
                } label: {
                    Image(systemName: isMapExpanded ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.primary)
                        .frame(width: 34, height: 34)
                        .background(.regularMaterial, in: Circle())
                }
                .accessibilityLabel(isMapExpanded ? "Collapse map" : "Expand map")

                Button {
                    if let fix = locationManager.currentFix {
                        observedArea.reset(to: fix.coordinate)
                        centerMap(on: fix.coordinate)
                        Task { await refreshLiveSightings() }
                    } else {
                        locationManager.centerOnUser()
                        frameClaimableRadius(animated: true)
                    }
                } label: {
                    Image(systemName: "location.fill")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.primary)
                        .frame(width: 34, height: 34)
                        .background(.regularMaterial, in: Circle())
                }
                .accessibilityLabel("Center on current location")
            }
            .padding(10)
        }
        .overlay(alignment: .bottomTrailing) {
            if observedArea.canSearchPreviewArea && store.selectedSighting == nil {
                Button {
                    let center = observedArea.searchPreviewArea()
                    centerMap(on: center)
                    Task { await refreshLiveSightings() }
                } label: {
                    Label("Search this area", systemImage: "magnifyingglass")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 9)
                        .background(.regularMaterial, in: Capsule())
                }
                .padding(10)
                .accessibilityLabel("Search this map area")
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .overlay(alignment: .bottom) {
            if let sighting = store.selectedSighting {
                MapSightingCallout(sighting: sighting) {
                    detailSighting = sighting
                } close: {
                    closeMapCallout()
                }
                .padding(10)
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.snappy(duration: 0.22), value: store.selectedSightingID)
        .animation(.easeInOut(duration: 0.85), value: store.sightings)
        .animation(.snappy(duration: 0.25), value: isMapExpanded)
    }

    private var claimableRadiusCenter: CLLocationCoordinate2D {
        observedArea.center
    }

    private var claimableRadiusMeters: CLLocationDistance {
        max(1, store.settings.alertRadiusNauticalMiles) * 1_852
    }

    private var claimableRadiusSpan: MKCoordinateSpan {
        let diameterNauticalMiles = max(1, store.settings.alertRadiusNauticalMiles) * 2
        let latitudeDelta = max(0.22, diameterNauticalMiles / 60 * 1.35)
        let longitudeDelta = max(0.28, latitudeDelta * 1.25)
        return MKCoordinateSpan(latitudeDelta: latitudeDelta, longitudeDelta: longitudeDelta)
    }

    private func centerMap(on coordinate: CLLocationCoordinate2D) {
        frameMap(on: coordinate, animated: true)
    }

    private func frameClaimableRadius(animated: Bool) {
        frameMap(on: claimableRadiusCenter, animated: animated)
    }

    private func frameMap(on coordinate: CLLocationCoordinate2D, animated: Bool) {
        let update = {
            cameraPosition = .region(
                MKCoordinateRegion(
                    center: coordinate,
                    span: claimableRadiusSpan
                )
            )
        }

        if animated {
            withAnimation(.easeInOut(duration: 0.35), update)
        } else {
            update()
        }
    }

    private func refreshLiveSightings() async {
        let shouldClassify = store.settings.usesLocalModelBridge
        if shouldClassify {
            lastPollClassificationAt = Date()
        }
        await store.refreshLiveSightings(
            near: observedArea.center,
            classify: shouldClassify,
            classificationLimit: 18
        )
    }

    private func runLivePolling() async {
        while !Task.isCancelled {
            let pollStartedAt = Date()
            let shouldClassify = store.settings.usesLocalModelBridge && shouldClassifyDuringPoll()
            if shouldClassify {
                lastPollClassificationAt = Date()
            }

            await store.refreshLiveSightings(
                near: observedArea.center,
                classify: shouldClassify,
                classificationLimit: 18,
                showsLoadingStatus: false
            )

            let elapsedMilliseconds = Int(Date().timeIntervalSince(pollStartedAt) * 1_000)
            let pollIntervalMilliseconds = Int(clampedPollingIntervalSeconds * 1_000)
            let sleepMilliseconds = max(100, pollIntervalMilliseconds - elapsedMilliseconds)
            do {
                try await Task.sleep(for: .milliseconds(sleepMilliseconds))
            } catch {
                return
            }
        }
    }

    private func shouldClassifyDuringPoll() -> Bool {
        guard let lastPollClassificationAt else { return true }
        return Date().timeIntervalSince(lastPollClassificationAt) >= clampedClassificationIntervalSeconds
    }

    private func closeMapCallout() {
        guard store.selectedSightingID != nil else { return }
        withAnimation(.snappy(duration: 0.2)) {
            store.selectedSightingID = nil
        }
    }

    private var primarySightings: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "Claimable", subtitle: "Rare now and incoming aircraft")
            if claimableSightings.isEmpty {
                claimableEmptyState
            } else {
                VStack(spacing: 0) {
                    ForEach(claimableSightings) { sighting in
                        NavigationLink {
                            SightingDetailView(sighting: sighting)
                        } label: {
                            SightingRow(sighting: sighting)
                        }
                        .buttonStyle(.plain)
                        Divider()
                    }
                }
                .padding(.horizontal, 12)
                .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
    }

    @ViewBuilder
    private var claimableEmptyState: some View {
        switch store.liveFeedState {
        case .idle, .loading:
            LiveFeedWaitingView()
        case .unavailable:
            LiveFeedUnavailableView()
        case .available:
            EmptyStateView()
        }
    }

    private var nearMisses: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "Near Misses", subtitle: "Interesting, but locally routine")
            VStack(spacing: 0) {
                ForEach(nearMissSightings) { sighting in
                    NavigationLink {
                        SightingDetailView(sighting: sighting)
                    } label: {
                        SightingRow(sighting: sighting)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
    }

    private var liveTraffic: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "Live Traffic", subtitle: "Routine ADS-B aircraft nearby")
            VStack(spacing: 0) {
                ForEach(routineSightings) { sighting in
                    NavigationLink {
                        SightingDetailView(sighting: sighting)
                    } label: {
                        SightingRow(sighting: sighting)
                    }
                    .buttonStyle(.plain)
                    Divider()
                }
            }
            .padding(.horizontal, 12)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
    }

    private var claimableSightings: [AircraftSighting] {
        store.featuredSightings.filter { sighting in
            store.settings.includeIncoming || sighting.rarityBand != .incoming
        }
    }

    private var nearMissSightings: [AircraftSighting] {
        guard !store.settings.claimableOnlyMode else { return [] }
        return store.nearMisses
    }

    private var routineSightings: [AircraftSighting] {
        guard !store.settings.claimableOnlyMode else { return [] }
        return store.sightings
            .filter { $0.rarityBand == .routine }
            .sorted { lhs, rhs in
                lhs.distanceNauticalMiles < rhs.distanceNauticalMiles
            }
            .prefix(12)
            .map { $0 }
    }

    private var visibleSightings: [AircraftSighting] {
        store.sightings.filter { sighting in
            if store.settings.claimableOnlyMode {
                return claimableSightings.contains(where: { $0.id == sighting.id })
            }
            if !store.settings.showRoutineAircraft && sighting.rarityBand == .routine {
                return false
            }
            if !store.settings.includeIncoming && sighting.rarityBand == .incoming {
                return false
            }
            return true
        }
    }

    private var clampedPollingIntervalSeconds: Double {
        min(30, max(3, store.settings.pollingIntervalSeconds))
    }

    private var clampedClassificationIntervalSeconds: Double {
        min(300, max(15, store.settings.classificationIntervalSeconds))
    }
}

struct MapSightingCallout: View {
    var sighting: AircraftSighting
    var open: () -> Void
    var close: () -> Void

    var body: some View {
        Button(action: open) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top, spacing: 10) {
                    ZStack {
                        Circle()
                            .fill(sighting.rarityBand.tint)
                        Image(systemName: sighting.aircraftSymbolName)
                            .font(.caption.weight(.bold))
                            .foregroundStyle(.white)
                            .rotationEffect(.degrees(sighting.aircraftIconRotationDegrees))
                    }
                    .frame(width: 34, height: 34)

                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Text(sighting.displayName)
                                .font(.headline)
                                .lineLimit(1)
                                .minimumScaleFactor(0.75)
                            StatusPill(band: sighting.rarityBand)
                        }
                        Text(sighting.subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }

                    Spacer(minLength: 32)
                }

                HStack(spacing: 12) {
                    Label(String(format: "%.1f nm", sighting.distanceNauticalMiles), systemImage: "location")
                    Label("\(sighting.altitudeFeet.formatted()) ft", systemImage: "arrow.up")
                    Label("\(sighting.groundSpeedKnots) kt", systemImage: "speedometer")
                }
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.75)

                RarityReadSummary(sighting: sighting, lineLimit: 3)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Open \(sighting.displayName) details")
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(.quaternary)
        }
        .overlay(alignment: .topTrailing) {
            Button(action: close) {
                Image(systemName: "xmark")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.secondary)
                    .frame(width: 28, height: 28)
                    .background(.thinMaterial, in: Circle())
            }
            .padding(8)
            .accessibilityLabel("Close aircraft popover")
        }
        .shadow(color: .black.opacity(0.16), radius: 14, y: 6)
    }
}

struct SectionHeader: View {
    var title: String
    var subtitle: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.title3.weight(.bold))
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }
}
