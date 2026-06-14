#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_PATH="${ARCHIVE_PATH:-$ROOT_DIR/build/RareBird.xcarchive}"
APP_PATH="$ARCHIVE_PATH/Products/Applications/RareBird.app"
EXECUTABLE_PATH="$APP_PATH/RareBird"
CLASSIFIER_PATH="$APP_PATH/RarityTextClassifier.json"
PRIVACY_PATH="$APP_PATH/PrivacyInfo.xcprivacy"
PROJECT_FILE="$ROOT_DIR/ios/RareBird/RareBird.xcodeproj/project.pbxproj"
EXPECTED_BUNDLE_ID="com.rarebird.app"
EXPECTED_TEAM_ID="84TUCN4A47"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

if [[ ! -d "$APP_PATH" ]]; then
  echo "missing archived app bundle: $APP_PATH" >&2
  exit 1
fi

require_file "$EXECUTABLE_PATH"
require_file "$CLASSIFIER_PATH"
require_file "$PRIVACY_PATH"

plutil -lint "$PRIVACY_PATH" >/dev/null

python3 - "$CLASSIFIER_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
model = json.loads(path.read_text())
required = {
    "model": "rarebirds_multinomial_nb_text_features",
    "version": 1,
}
for key, expected in required.items():
    actual = model.get(key)
    if actual != expected:
        raise SystemExit(f"{path}: expected {key}={expected!r}, found {actual!r}")
if float(model.get("threshold", 0)) < 0.70:
    raise SystemExit(f"{path}: threshold must be at least 0.70")
if int(model.get("train_examples", 0)) < 3000:
    raise SystemExit(f"{path}: expected at least 3000 training examples")
if len(model.get("vocabulary", [])) < 10000:
    raise SystemExit(f"{path}: expected at least 10000 vocabulary entries")
PY

if rg -q "MLXLLM|MLXLMCommon|SwiftTransformers|Tokenizers" "$PROJECT_FILE"; then
  echo "release project still references experimental MLX packages" >&2
  exit 1
fi

if plutil -p "$APP_PATH/Info.plist" | rg -q "NSAllowsArbitraryLoads|localhost|127\\.0\\.0\\.1"; then
  echo "release Info.plist contains development network exceptions" >&2
  exit 1
fi

signing_identity="unsigned"
team_identifier="none"
if codesign_output="$(codesign -dv --verbose=4 "$APP_PATH" 2>&1)"; then
  signing_identity="$(printf "%s\n" "$codesign_output" | awk -F= '/^Authority=/{print $2; exit}')"
  team_identifier="$(printf "%s\n" "$codesign_output" | awk -F= '/^TeamIdentifier=/{print $2; exit}')"
fi

profile_name="none"
profile_app_id="none"
profile_get_task_allow="unknown"
if [[ -f "$APP_PATH/embedded.mobileprovision" ]]; then
  profile_fields="$(
    /usr/bin/python3 - "$APP_PATH/embedded.mobileprovision" <<'PY'
import plistlib
import subprocess
import sys

decoded = subprocess.check_output(
    ["security", "cms", "-D", "-i", sys.argv[1]],
    stderr=subprocess.DEVNULL,
)
profile = plistlib.loads(decoded)
entitlements = profile.get("Entitlements", {})
print(profile.get("Name", "none"))
print(entitlements.get("application-identifier", "none"))
print(str(entitlements.get("get-task-allow", "unknown")).lower())
PY
  )"
  profile_name="$(sed -n '1p' <<<"$profile_fields")"
  profile_app_id="$(sed -n '2p' <<<"$profile_fields")"
  profile_get_task_allow="$(sed -n '3p' <<<"$profile_fields")"
fi

if [[ "${REQUIRE_APP_STORE_SIGNING:-0}" == "1" ]]; then
  if [[ "$signing_identity" != Apple\ Distribution:* ]]; then
    echo "expected Apple Distribution signing identity, found: $signing_identity" >&2
    exit 1
  fi
  if [[ "$team_identifier" != "$EXPECTED_TEAM_ID" ]]; then
    echo "expected team $EXPECTED_TEAM_ID, found: $team_identifier" >&2
    exit 1
  fi
  if [[ "$profile_app_id" != "$EXPECTED_TEAM_ID.$EXPECTED_BUNDLE_ID" ]]; then
    echo "expected App Store profile for $EXPECTED_TEAM_ID.$EXPECTED_BUNDLE_ID, found: $profile_app_id" >&2
    exit 1
  fi
  if [[ "$profile_get_task_allow" != "false" ]]; then
    echo "expected get-task-allow=false for App Store signing, found: $profile_get_task_allow" >&2
    exit 1
  fi
fi

app_size="$(du -sh "$APP_PATH" | awk '{print $1}')"
classifier_size="$(du -h "$CLASSIFIER_PATH" | awk '{print $1}')"
echo "Verified iOS release archive:"
echo "  archive: $ARCHIVE_PATH"
echo "  app bundle: $app_size"
echo "  classifier: $classifier_size"
echo "  privacy manifest: present"
echo "  signing identity: $signing_identity"
echo "  provisioning profile: $profile_name"
