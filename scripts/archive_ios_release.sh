#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT_DIR/ios/RareBird/RareBird.xcodeproj"
SCHEME="RareBird"
CONFIGURATION="Release"
ARCHIVE_PATH="${ARCHIVE_PATH:-$ROOT_DIR/build/RareBird.xcarchive}"
EXPORT_PATH="${EXPORT_PATH:-$ROOT_DIR/build/export}"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-$ROOT_DIR/build/DerivedData-archive}"
if [[ "${UPLOAD:-0}" == "1" ]]; then
  DEFAULT_EXPORT_OPTIONS="$ROOT_DIR/ios/RareBird/ExportOptions.app-store-connect-upload.plist"
else
  DEFAULT_EXPORT_OPTIONS="$ROOT_DIR/ios/RareBird/ExportOptions.app-store-connect.plist"
fi
EXPORT_OPTIONS="${EXPORT_OPTIONS:-$DEFAULT_EXPORT_OPTIONS}"

mkdir -p "$(dirname "$ARCHIVE_PATH")" "$EXPORT_PATH" "$DERIVED_DATA_PATH"

if [[ "${UNSIGNED:-0}" == "1" ]]; then
  xcodebuild archive \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -destination "generic/platform=iOS" \
    -archivePath "$ARCHIVE_PATH" \
    -derivedDataPath "$DERIVED_DATA_PATH" \
    CODE_SIGNING_ALLOWED=NO
  exit 0
fi

xcodebuild archive \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -destination "generic/platform=iOS" \
  -archivePath "$ARCHIVE_PATH" \
  -derivedDataPath "$DERIVED_DATA_PATH" \
  -allowProvisioningUpdates

xcodebuild -exportArchive \
  -archivePath "$ARCHIVE_PATH" \
  -exportPath "$EXPORT_PATH" \
  -exportOptionsPlist "$EXPORT_OPTIONS" \
  -allowProvisioningUpdates
