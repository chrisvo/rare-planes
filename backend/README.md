# Backend

The backend watches aircraft feeds, evaluates rare-aircraft rules, and sends APNs notifications.

## Initial Components

- `FlightDataProvider`: provider boundary for OpenSky, ADSB Exchange, FlightAware, or a local mock.
- `AircraftNormalizer`: converts provider payloads into a stable internal shape.
- `RareRuleEngine`: evaluates aircraft states against configured rare-aircraft rules.
- `GeoMatcher`: finds user regions containing an aircraft.
- `AlertDedupe`: applies cooldowns per user and aircraft.
- `NotificationSender`: sends APNs pushes.

## Prototype Target

Start with a command that accepts a latitude, longitude, radius, and rules file, then logs matching aircraft.

```bash
rare-bird scan --lat 33.9425 --lon -118.4081 --radius-nm 50 --rules data/rare-aircraft-rules.example.json
```

