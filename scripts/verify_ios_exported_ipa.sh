#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IPA_PATH="${IPA_PATH:-$ROOT_DIR/build/export/rarebirds.ipa}"
EXPECTED_BUNDLE_ID="${BUNDLE_ID:-com.rarebird.app}"
EXPECTED_TEAM_ID="${TEAM_ID:-84TUCN4A47}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rarebird-ipa.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ ! -f "$IPA_PATH" ]]; then
  echo "missing IPA: $IPA_PATH" >&2
  exit 1
fi

unzip -q "$IPA_PATH" -d "$WORK_DIR"
APP_PATH="$(find "$WORK_DIR/Payload" -maxdepth 1 -name "*.app" -type d | head -1)"

if [[ -z "$APP_PATH" || ! -d "$APP_PATH" ]]; then
  echo "IPA does not contain a Payload/*.app bundle" >&2
  exit 1
fi

for file in "$APP_PATH/RareBird" "$APP_PATH/RarityTextClassifier.json" "$APP_PATH/PrivacyInfo.xcprivacy" "$APP_PATH/embedded.mobileprovision"; do
  if [[ ! -f "$file" ]]; then
    echo "missing required IPA payload file: $file" >&2
    exit 1
  fi
done

codesign_output="$(codesign -dv --verbose=4 "$APP_PATH" 2>&1)"
signing_identity="$(printf "%s\n" "$codesign_output" | awk -F= '/^Authority=/{print $2; exit}')"
team_identifier="$(printf "%s\n" "$codesign_output" | awk -F= '/^TeamIdentifier=/{print $2; exit}')"
bundle_identifier="$(printf "%s\n" "$codesign_output" | awk -F= '/^Identifier=/{print $2; exit}')"

if [[ "$signing_identity" != Apple\ Distribution:* ]]; then
  echo "expected Apple Distribution signing identity, found: $signing_identity" >&2
  exit 1
fi
if [[ "$team_identifier" != "$EXPECTED_TEAM_ID" ]]; then
  echo "expected team $EXPECTED_TEAM_ID, found: $team_identifier" >&2
  exit 1
fi
if [[ "$bundle_identifier" != "$EXPECTED_BUNDLE_ID" ]]; then
  echo "expected bundle id $EXPECTED_BUNDLE_ID, found: $bundle_identifier" >&2
  exit 1
fi

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
print(len(profile.get("ProvisionedDevices", [])))
PY
)"
profile_name="$(sed -n '1p' <<<"$profile_fields")"
profile_app_id="$(sed -n '2p' <<<"$profile_fields")"
profile_get_task_allow="$(sed -n '3p' <<<"$profile_fields")"
profile_devices="$(sed -n '4p' <<<"$profile_fields")"

if [[ "$profile_app_id" != "$EXPECTED_TEAM_ID.$EXPECTED_BUNDLE_ID" ]]; then
  echo "expected App Store profile for $EXPECTED_TEAM_ID.$EXPECTED_BUNDLE_ID, found: $profile_app_id" >&2
  exit 1
fi
if [[ "$profile_get_task_allow" != "false" ]]; then
  echo "expected get-task-allow=false, found: $profile_get_task_allow" >&2
  exit 1
fi
if [[ "$profile_devices" != "0" ]]; then
  echo "expected App Store profile with no provisioned devices, found: $profile_devices" >&2
  exit 1
fi

python3 - "$APP_PATH/RarityTextClassifier.json" <<'PY'
import json
import sys
from pathlib import Path

model = json.loads(Path(sys.argv[1]).read_text())
if model.get("model") != "rarebirds_multinomial_nb_text_features":
    raise SystemExit("unexpected classifier model")
if float(model.get("threshold", 0)) < 0.70:
    raise SystemExit("classifier threshold below 0.70")
if int(model.get("train_examples", 0)) < 3000:
    raise SystemExit("classifier train_examples below 3000")
PY

ipa_size="$(du -h "$IPA_PATH" | awk '{print $1}')"
echo "Verified exported iOS IPA:"
echo "  ipa: $IPA_PATH ($ipa_size)"
echo "  bundle id: $bundle_identifier"
echo "  signing identity: $signing_identity"
echo "  provisioning profile: $profile_name"
echo "  classifier: present"
echo "  privacy manifest: present"
