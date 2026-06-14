#!/usr/bin/env bash
set -euo pipefail

TEAM_ID="${TEAM_ID:-84TUCN4A47}"
BUNDLE_ID="${BUNDLE_ID:-com.rarebird.app}"
PROFILE_DIR="${PROFILE_DIR:-$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles}"

failures=0

note() {
  printf "%s\n" "$1"
}

fail() {
  printf "FAIL: %s\n" "$1"
  failures=$((failures + 1))
}

pass() {
  printf "PASS: %s\n" "$1"
}

note "Checking App Store signing prerequisites for $TEAM_ID.$BUNDLE_ID"

if security find-identity -v -p codesigning | rg -q "Apple Distribution:.*\\($TEAM_ID\\)"; then
  pass "Apple Distribution signing identity found for team $TEAM_ID"
else
  note "Info: no local Apple Distribution identity found. Xcode can still export with a cloud-managed Apple Distribution certificate if an App Store profile exists."
fi

if security find-identity -v -p codesigning | rg -q "Apple Development:"; then
  note "Info: Apple Development identity exists, but it is not valid for App Store export"
fi

exact_profile_found=0
if [[ -d "$PROFILE_DIR" ]]; then
  while IFS= read -r -d '' profile; do
    profile_fields="$(
      /usr/bin/python3 - "$profile" <<'PY'
import datetime as dt
import plistlib
import subprocess
import sys

path = sys.argv[1]
decoded = subprocess.check_output(
    ["security", "cms", "-D", "-i", path],
    stderr=subprocess.DEVNULL,
)
profile = plistlib.loads(decoded)
entitlements = profile.get("Entitlements", {})
expiration = profile.get("ExpirationDate")
expired = expiration is not None and expiration <= dt.datetime.now(expiration.tzinfo)
print(profile.get("Name", "none"))
print(profile.get("UUID", "none"))
print(entitlements.get("application-identifier", "none"))
print(str(entitlements.get("get-task-allow", "unknown")).lower())
print("true" if expired else "false")
PY
    )"
    name="$(sed -n '1p' <<<"$profile_fields")"
    uuid="$(sed -n '2p' <<<"$profile_fields")"
    app_id="$(sed -n '3p' <<<"$profile_fields")"
    get_task_allow="$(sed -n '4p' <<<"$profile_fields")"
    expired="$(sed -n '5p' <<<"$profile_fields")"

    if [[ "$app_id" == "$TEAM_ID.$BUNDLE_ID" ]]; then
      if [[ "$get_task_allow" == "false" && "$expired" == "false" ]]; then
        exact_profile_found=1
        pass "App Store provisioning profile found: $name ($uuid)"
      else
        note "Info: exact profile is not App Store-ready: $name ($uuid), get-task-allow=$get_task_allow, expired=$expired"
      fi
    elif [[ "$app_id" == "$TEAM_ID.*" ]]; then
      note "Info: wildcard development profile found, not App Store-ready: $name ($uuid)"
    fi
  done < <(find "$PROFILE_DIR" -maxdepth 1 -name "*.mobileprovision" -print0)
fi

if [[ "$exact_profile_found" != "1" ]]; then
  fail "missing non-expired App Store provisioning profile for $TEAM_ID.$BUNDLE_ID with get-task-allow=false"
fi

if [[ "$failures" -gt 0 ]]; then
  note ""
  note "Next fix in Xcode: Settings > Accounts, re-authenticate the Apple ID, then let Xcode create/download an App Store profile for $BUNDLE_ID."
  exit 1
fi

note "App Store signing preflight passed."
