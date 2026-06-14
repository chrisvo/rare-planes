import CoreLocation
import Foundation

struct UserLocationFix: Equatable {
    var latitude: Double
    var longitude: Double

    var coordinate: CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }
}

final class UserLocationManager: NSObject, ObservableObject {
    @Published private(set) var authorizationStatus: CLAuthorizationStatus
    @Published private(set) var currentFix: UserLocationFix?

    private let manager = CLLocationManager()

    override init() {
        authorizationStatus = manager.authorizationStatus
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        manager.distanceFilter = 25
    }

    var canShowUserLocation: Bool {
        switch authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            true
        default:
            false
        }
    }

    func activate() {
        authorizationStatus = manager.authorizationStatus
        AppLog.info("location activate status=\(authorizationStatus.rawValue)", logger: AppLog.location)

        switch authorizationStatus {
        case .notDetermined:
            AppLog.info("location request when-in-use authorization", logger: AppLog.location)
            manager.requestWhenInUseAuthorization()
        case .authorizedAlways, .authorizedWhenInUse:
            AppLog.info("location start updating", logger: AppLog.location)
            manager.startUpdatingLocation()
            manager.requestLocation()
        case .denied, .restricted:
            AppLog.error("location unavailable status=\(authorizationStatus.rawValue)", logger: AppLog.location)
            manager.stopUpdatingLocation()
            currentFix = nil
        @unknown default:
            manager.stopUpdatingLocation()
        }
    }

    func centerOnUser() {
        guard canShowUserLocation else {
            AppLog.info("location center requested before authorization", logger: AppLog.location)
            activate()
            return
        }
        AppLog.info("location center requested", logger: AppLog.location)
        manager.requestLocation()
    }
}

extension UserLocationManager: CLLocationManagerDelegate {
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        authorizationStatus = manager.authorizationStatus
        AppLog.info("location authorization changed status=\(authorizationStatus.rawValue)", logger: AppLog.location)

        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            AppLog.info("location authorized; starting updates", logger: AppLog.location)
            manager.startUpdatingLocation()
            manager.requestLocation()
        case .denied, .restricted:
            AppLog.error("location denied or restricted status=\(authorizationStatus.rawValue)", logger: AppLog.location)
            manager.stopUpdatingLocation()
            currentFix = nil
        case .notDetermined:
            break
        @unknown default:
            manager.stopUpdatingLocation()
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let coordinate = locations.last?.coordinate else { return }
        currentFix = UserLocationFix(latitude: coordinate.latitude, longitude: coordinate.longitude)
        AppLog.info("location fix lat=\(coordinate.latitude) lon=\(coordinate.longitude)", logger: AppLog.location)
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        AppLog.error("location failed error=\(error.localizedDescription)", logger: AppLog.location)
        if (error as? CLError)?.code == .denied {
            currentFix = nil
        }
    }
}
