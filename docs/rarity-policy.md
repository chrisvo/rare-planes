# Rarity Policy

Rare Bird should alert on aircraft that are unusual, chase-worthy, safety-relevant, or personally notable in the observer's local context. The app must not promote routine Southern California traffic simply because the aircraft is globally interesting.

## Hard Alert Signals

These signals are claimable regardless of otherwise routine type or operator:

- Emergency squawks `7500`, `7600`, or `7700`.
- Known special registrations, special liveries, notable individual aircraft, or user watchlist aircraft.
- Rare callsign prefixes or special-mission callsigns such as `NASA`, `RCH`, `REACH`, `JOSA`, `VENUS`, `RESCUE`, `GUARD`, or similarly explicit military/government mission callsigns.
- Rare, historic, last-of-type, experimental, specialized cargo, bomber, tanker, or unusual military aircraft when not obviously operating as routine local pattern traffic.

## H60 Helicopters

H60 traffic is locally common around Los Alamitos, Camp Pendleton, San Diego, military training routes, and coastal support patterns.

- Claimable: H60 traffic away from a known local military/training pattern, carrying a special callsign or mission signal, using an emergency squawk, operating unusually low/slow through a city corridor, or matching a notable registration/watchlist.
- Near-miss: H60 traffic near a base, military route, or expected Southern California training area when it is visible and interesting but locally routine.
- Routine: Repeated base-pattern, training, transit, or ordinary military helicopter activity with no special callsign, emergency, watchlist, or unusual route signal.

## Public-Safety Helicopters

Police, sheriff, fire, CAL FIRE, coast guard, medevac, and rescue helicopters are operationally important but often common in Southern California.

- Claimable: Emergency squawk, explicit rescue/evacuation/special incident callsign, unusual agency or aircraft type for the area, notable registration, watchlist match, or a clearly rare special mission.
- Near-miss: Recognizable public-safety helicopter operating a local mission that a spotter may care about, but with no rare aircraft or special alert signal.
- Routine: LAPD, sheriff, fire, medevac, news, and patrol helicopter activity that is common locally and has no emergency, notable registration, unusual type, or special-mission signal.

## Common Fixed-Wing Traffic

The app should suppress obvious routine traffic unless a hard alert signal is present.

- Routine airline aircraft: common A320-family, 737-family, E-Jet, CRJ, 757/767 cargo trunk traffic, and other normal LAX/SNA/LGB/BUR/ONT airline flows.
- Routine business jets: common Gulfstream, Challenger, Citation, Phenom, Learjet, Falcon, and Global traffic without unusual operator, emergency, special registration, or rare route signal.
- Routine trainers and GA: common Cessna, Piper, Cirrus, Diamond, Beech, Mooney, and flight-school aircraft, especially near training airports and practice areas.

## Contextual Aircraft

Some aircraft are interesting but not automatic alerts in Southern California.

- `B744`, `B748`, and other uncommon widebodies are near-miss or routine unless paired with an unusual operator, special livery, notable registration, emergency, rare route, or unusual callsign.
- Military trainers can be near-miss around known bases and training corridors, but claimable away from expected patterns or with special mission signals.
- Common helicopters and business jets can become claimable only when a hard alert or clearly unusual context is present.

## User-Visible Reasons

Reasons must explain both promotion and suppression:

- Claimable reasons should name the decisive signal, such as emergency squawk, special registration, rare type, unusual mission callsign, or uncommon route context.
- Near-miss reasons should say why the aircraft is interesting but locally expected.
- Routine reasons should identify the local baseline, such as common airline flow, common business jet, common trainer, or routine public-safety helicopter.

## Regression Examples

- `C172`, `PA28`, `SR22`, `DA40`: routine trainer/general-aviation traffic unless emergency, watchlist, notable registration, or special mission.
- `B738`, `A320`, `A20N`, `E75L`, `CRJ9`: routine airline flow around LAX/SNA/LGB/BUR/ONT unless hard alert signal.
- `B744`: near-miss/contextual for normal cargo flow, including routine Kalitta-style operations, unless special livery, unusual operator, notable registration, emergency, or rare route context.
- `B748`: near-miss/contextual unless special livery, unusual operator, notable registration, emergency, or rare route context.
- `H60`: near-miss or routine around local military/training patterns; claimable only with hard alert or unusual off-pattern context.
- Public-safety helicopters: near-miss or routine by default; claimable for emergency squawk, unusual type, notable registration, or special incident signal.
- Common business jets such as `C56X`, `C680`, `CL30`, `GLF5`, `GLEX`: routine unless paired with a hard alert signal.
- True rare alert: `BLCF`, `A124`, `C17`, `MD11`, `V22`, emergency squawk `7700`, known special registration, or `NASA`/`REACH`/`RESCUE` callsign.
