from __future__ import annotations

from scripts.rarity_engine import score_aircraft, explanation_from_score


def test_coast_guard_helicopter_away_from_base_is_high_scoring_show_candidate():
    aircraft = {
        "callsign": "C6043",
        "registration": "N6043C",
        "type_designator": "H60",
        "description": "SIKORSKY MH-60T Jayhawk",
        "operator": "United States Coast Guard",
        "distance_nm": 18,
    }

    score = score_aircraft(aircraft, observer_context={"military_pattern": "away_from_base"})

    assert score["is_rare"] is True
    assert score["rarity_score"] >= 80
    assert score["recommendation"] == "show"
    assert "coast_guard" in score["reason_codes"]
    assert "military_or_special_away_from_base" in score["reason_codes"]
    assert any("Coast Guard" in factor["label"] for factor in score["factors"])


def test_routine_airline_traffic_is_suppressed_with_auditable_reason():
    aircraft = {
        "callsign": "SWA123",
        "type_designator": "B737",
        "description": "BOEING 737-700",
        "operator": "Southwest Airlines",
        "distance_nm": 5,
    }

    score = score_aircraft(aircraft)

    assert score["is_rare"] is False
    assert score["rarity_score"] <= 20
    assert score["recommendation"] == "suppress"
    assert "common_airline" in score["reason_codes"]
    assert "routine airline" in score["summary"].lower()


def test_military_type_near_base_is_contextual_not_automatically_rare():
    aircraft = {
        "callsign": "RCH456",
        "type_designator": "C17",
        "description": "BOEING C-17 Globemaster III",
        "operator": "USAF",
        "distance_nm": 8,
    }

    score = score_aircraft(aircraft, observer_context={"military_pattern": "base_pattern", "nearest_military_area": "March ARB"})

    assert score["is_rare"] is False
    assert 35 <= score["rarity_score"] <= 69
    assert score["recommendation"] == "review"
    assert "military_near_base_context" in score["reason_codes"]
    assert "globally interesting" in score["summary"].lower()


def test_emergency_squawk_overrides_common_suppression():
    aircraft = {
        "callsign": "SWA7700",
        "type_designator": "B738",
        "description": "BOEING 737-800",
        "operator": "Southwest Airlines",
        "squawk": "7700",
    }

    score = score_aircraft(aircraft)

    assert score["is_rare"] is True
    assert score["rarity_score"] >= 85
    assert score["recommendation"] == "show"
    assert "emergency_squawk" in score["reason_codes"]


def test_explanation_uses_rule_evidence_not_free_floating_model_claims():
    score = {
        "aircraft_label": "USCG MH-60T Jayhawk C6043",
        "rarity_score": 82,
        "recommendation": "show",
        "factors": [
            {"code": "coast_guard", "label": "Coast Guard operator", "points": 35},
            {"code": "military_or_special_away_from_base", "label": "Special-use aircraft away from a base pattern", "points": 25},
        ],
    }

    explanation = explanation_from_score(score)

    assert "USCG MH-60T Jayhawk C6043" in explanation
    assert "82/100" in explanation
    assert "Coast Guard operator" in explanation
    assert "Special-use aircraft away from a base pattern" in explanation
