#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import gradio as gr


DEFAULT_DATASET = Path("data/datasets/rarity-oc-la-socal-2000/train.csv")


def load_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            prompt = json.loads(row["prompt"])
            response = json.loads(row["response"])
            aircraft = prompt.get("aircraft") or {}
            context = prompt.get("observer_context") or {}
            rows.append(
                {
                    "index": index,
                    "is_rare": bool(response.get("is_rare")),
                    "confidence": response.get("confidence"),
                    "reason": response.get("reason"),
                    "provider": aircraft.get("provider") or "unknown",
                    "type": aircraft.get("type_designator") or "unknown",
                    "description": aircraft.get("description") or "",
                    "operator": aircraft.get("operator") or "",
                    "callsign": aircraft.get("callsign") or "",
                    "registration": aircraft.get("registration") or "",
                    "lat": aircraft.get("lat"),
                    "lon": aircraft.get("lon"),
                    "local_area": context.get("current_local_area") or aircraft.get("local_area") or "",
                    "nearest_airport": context.get("nearest_airport") or aircraft.get("nearest_airport") or "",
                    "nearest_military_area": context.get("nearest_military_area") or aircraft.get("nearest_military_area") or "",
                    "military_pattern": context.get("military_pattern") or aircraft.get("military_pattern") or "",
                }
            )
    return rows


def pct(part: int, whole: int) -> str:
    return f"{(part / whole * 100):.1f}%" if whole else "0.0%"


def bar_table(counter: Counter, total: int, title: str, limit: int = 12) -> str:
    lines = [f"<h3>{html.escape(title)}</h3>", "<div class='bars'>"]
    for label, count in counter.most_common(limit):
        width = pct(count, total)
        lines.append(
            "<div class='bar-row'>"
            f"<span class='bar-label'>{html.escape(str(label))}</span>"
            f"<span class='bar'><span style='width:{width}'></span></span>"
            f"<span class='bar-count'>{count}</span>"
            "</div>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def map_svg(rows: list[dict[str, Any]]) -> str:
    points = [row for row in rows if isinstance(row.get("lat"), (int, float)) and isinstance(row.get("lon"), (int, float))]
    if not points:
        return "<p>No latitude/longitude values available.</p>"
    min_lat = min(float(row["lat"]) for row in points)
    max_lat = max(float(row["lat"]) for row in points)
    min_lon = min(float(row["lon"]) for row in points)
    max_lon = max(float(row["lon"]) for row in points)
    lat_span = max(max_lat - min_lat, 0.01)
    lon_span = max(max_lon - min_lon, 0.01)

    circles = []
    for row in points[:1200]:
        x = 30 + ((float(row["lon"]) - min_lon) / lon_span) * 720
        y = 370 - ((float(row["lat"]) - min_lat) / lat_span) * 340
        color = "#d83131" if row["is_rare"] else "#2876d1"
        radius = 4.5 if row["is_rare"] else 2.8
        title = f"{row['type']} {row['callsign']} - {'rare' if row['is_rare'] else 'not rare'}"
        circles.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='{color}' opacity='0.72'><title>{html.escape(title)}</title></circle>")

    return (
        "<h3>Lat/Lon Scatter</h3>"
        "<svg viewBox='0 0 780 400' class='map-svg' role='img'>"
        "<rect x='20' y='20' width='740' height='360' rx='6' fill='#f8fafc' stroke='#cbd5e1'/>"
        + "\n".join(circles)
        + "<text x='30' y='392' font-size='12' fill='#475569'>blue = not rare, red = rare; hover points for type/callsign</text>"
        "</svg>"
    )


def summary_html(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    rare = sum(1 for row in rows if row["is_rare"])
    providers = Counter(row["provider"] for row in rows)
    types = Counter(row["type"] for row in rows)
    patterns = Counter(row["military_pattern"] or "none" for row in rows)
    areas = Counter(row["local_area"] or "none" for row in rows)
    return f"""
<style>
.summary-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:10px; margin-bottom:16px; }}
.metric {{ border:1px solid #d7dde7; border-radius:6px; padding:10px; background:#fff; }}
.metric b {{ display:block; font-size:22px; }}
.viz-grid {{ display:grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap:16px; }}
.bars {{ display:flex; flex-direction:column; gap:6px; }}
.bar-row {{ display:grid; grid-template-columns: 150px 1fr 45px; gap:8px; align-items:center; font-size:13px; }}
.bar-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.bar {{ height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }}
.bar span {{ display:block; height:100%; background:#2563eb; }}
.bar-count {{ text-align:right; color:#475569; }}
.map-svg {{ width:100%; height:auto; }}
</style>
<div class='summary-grid'>
  <div class='metric'><span>Total</span><b>{total}</b></div>
  <div class='metric'><span>Rare</span><b>{rare}</b><small>{pct(rare, total)}</small></div>
  <div class='metric'><span>Not Rare</span><b>{total - rare}</b><small>{pct(total - rare, total)}</small></div>
  <div class='metric'><span>Providers</span><b>{len(providers)}</b></div>
</div>
<div class='viz-grid'>
  <div>{bar_table(providers, total, "Source Mix")}{bar_table(patterns, total, "Military Pattern")}</div>
  <div>{bar_table(types, total, "Top Types")}{bar_table(areas, total, "Local Areas")}</div>
</div>
{map_svg(rows)}
"""


def filtered_table(rows: list[dict[str, Any]], label_filter: str, query: str, limit: int) -> list[list[Any]]:
    query = query.strip().lower()
    output = []
    for row in rows:
        if label_filter == "rare" and not row["is_rare"]:
            continue
        if label_filter == "not rare" and row["is_rare"]:
            continue
        haystack = " ".join(str(row.get(key) or "") for key in ["type", "description", "operator", "callsign", "registration", "reason", "local_area"]).lower()
        if query and query not in haystack:
            continue
        output.append(
            [
                row["index"],
                row["is_rare"],
                row["confidence"],
                row["type"],
                row["callsign"],
                row["description"],
                row["operator"],
                row["local_area"],
                row["military_pattern"],
                row["reason"],
            ]
        )
        if len(output) >= limit:
            break
    return output


def build_app(dataset: Path) -> gr.Blocks:
    rows = load_examples(dataset)
    with gr.Blocks(title="rarebirds Dataset Explorer") as app:
        gr.Markdown(f"# rarebirds Dataset Explorer\n`{dataset}`")
        gr.HTML(summary_html(rows))
        with gr.Row():
            label_filter = gr.Dropdown(["all", "rare", "not rare"], value="all", label="Label")
            query = gr.Textbox(label="Search", placeholder="H60, KNIFE, Dreamlifter, SNA, Camp Pendleton...")
            limit = gr.Slider(25, 500, value=100, step=25, label="Rows")
        table = gr.Dataframe(
            headers=["index", "is_rare", "confidence", "type", "callsign", "description", "operator", "local_area", "military_pattern", "reason"],
            value=filtered_table(rows, "all", "", 100),
            interactive=False,
            wrap=True,
        )
        for control in [label_filter, query, limit]:
            control.change(lambda label, text, n: filtered_table(rows, label, text, int(n)), [label_filter, query, limit], table)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    build_app(args.dataset).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
