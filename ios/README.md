# iPhone App

Native SwiftUI is the default direction for the iPhone app.

Current app package:

```text
ios/RareBird
```

Open `ios/RareBird/RareBird.xcodeproj` in Xcode and run the `RareBird` scheme.

The app is local-model-first. The backend supplies normalized aircraft sightings
and regional context; the iPhone runs the rarity classifier on device and uses
the result to decide whether to alert, show a near miss, or make a sighting
claimable.

## Screens

- Now: map and field-journal list of rare aircraft, incoming aircraft, and near misses.
- Aircraft Detail: callsign, registration, type, altitude, heading, rarity reason, claim state, and last seen time.
- Logbook: collected types and airframes with optional user photos.
- Settings: alert radius, categories, watchlist, quiet hours, and notification status.

## App Responsibilities

- Collect location and notification permissions.
- Store user preferences.
- Download or bundle the on-device rarity model.
- Run local rarity classification against normalized aircraft sightings and observer context.
- Register the APNs device token with the backend for wake-up notifications.
- Display nearby matches, near misses, claimable sightings, and collection history.

The app should not continuously poll aircraft feeds in the background. The backend
monitors configured regions, sends compact candidate sightings, and wakes the app
when there is something worth evaluating locally.

## Model Deployment

Current iPhone artifact:

```text
ios/RareBird/Sources/RareBirdApp/Resources/RarityTextClassifier.json
```

It is a bundled multinomial text classifier, currently about 0.90 MB. It is the
first classifier the app tries on device. The runtime chain is:

```text
TextRarityClassifier -> deterministic guardrails
```

Regenerate it with:

```bash
python3 scripts/export_rarity_text_classifier.py \
  --train-csv data/datasets/rarity-training-v7-combined-split/train.csv \
  --threshold 0.70 \
  --output ios/RareBird/Sources/RareBirdApp/Resources/RarityTextClassifier.json
```

Current gate results:

```text
v7 held-out eval:          96.7% accuracy, precision 0.992, recall 0.935, F1 0.963
gold eval:                 92.6% accuracy, precision 1.000, recall 0.849, F1 0.919
gold invalid JSON rate:    0.0%
```

The previous MLX LLM fallback still exists:

```text
model/output/rarity-gemma4-oc-la-hard-v3-mlx-4bit
```

Install it into the simulator with:

```bash
scripts/install_ios_mlx_model.sh
```

MLX is not linked in the App Store project because the current artifact is about
2.5 GB and generated JSON reliability was weaker than the classifier gate. To
convert a future merged checkpoint to MLX for experiments:

```bash
scripts/convert_rarity_model_mlx.sh
```

`NetworkRarityClassifier` remains source-level development plumbing for bridge
testing, not the default iPhone classifier. The App Store plist does not include
localhost ATS exceptions. Deterministic guardrails remain active around the local
model for B744/B748 and hard alert policy.

## Build Status

Verified with:

```bash
xcodebuild -project ios/RareBird/RareBird.xcodeproj \
  -scheme RareBird \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  build

xcodebuild -project ios/RareBird/RareBird.xcodeproj \
  -scheme RareBird \
  -configuration Debug \
  -destination 'generic/platform=iOS Simulator' \
  build
```

## App Store Archive

The App Store build uses the bundled text classifier and includes:

```text
ios/RareBird/Sources/RareBirdApp/Resources/PrivacyInfo.xcprivacy
ios/RareBird/ExportOptions.app-store-connect.plist
```

`PrivacyInfo.xcprivacy` declares `UserDefaults` with reason `CA92.1` for
app-only settings and logbook storage. Apple documents privacy manifests and
required-reason APIs in:

```text
https://developer.apple.com/documentation/bundleresources/privacy-manifest-files
https://developer.apple.com/documentation/bundleresources/describing-use-of-required-reason-api
```

Unsigned archive smoke test:

```bash
UNSIGNED=1 scripts/archive_ios_release.sh
scripts/verify_ios_release_archive.sh
```

Signed App Store Connect export, once the Apple account/certificates are present
in Xcode:

```bash
scripts/preflight_ios_app_store_signing.sh
scripts/archive_ios_release.sh
REQUIRE_APP_STORE_SIGNING=1 scripts/verify_ios_release_archive.sh
scripts/verify_ios_exported_ipa.sh
```

Upload directly to App Store Connect after signing is configured:

```bash
UPLOAD=1 scripts/archive_ios_release.sh
```

`scripts/verify_ios_exported_ipa.sh` expects the exported IPA to be signed with
an Apple Distribution identity, team `84TUCN4A47`, an exact `com.rarebird.app`
provisioning profile, and `get-task-allow=false`. A wildcard
`iOS Team Provisioning Profile: *` or `Apple Development` identity is enough for
local device builds and archive creation, but not for the exported IPA.

If export succeeds but `UPLOAD=1 scripts/archive_ios_release.sh` fails with
`IDEDistribution.DistributionAppRecordProviderError.missingApp`, create the App
Store Connect app record for bundle ID `com.rarebird.app`, then retry the
upload command.

On a machine without an App Store provisioning profile for `com.rarebird.app`,
`xcodebuild -exportArchive` is expected to stop at signing with:

```text
No profiles for 'com.rarebird.app' were found
```
