"""Find AOIs where OSM swimming-pool outlines sit under fAIr-eligible OAM imagery.

This is a metadata/count-only preflight: it does not download imagery or OSM
geometries. Default search is OSM-first: each continent is gridded, cells with
enough mapped pool *ways* are kept, then OAM is checked only on those cells.
Pass `--aois` to skip the scan and use a hand list of boxes.

Indoor and outdoor pools are both counted. Training labels are polygons (OSM
ways). Centres are used here only as a cheap overlap test, not as labels.

OAM licenses: CC-BY-4.0, CC-BY-SA-4.0, CC-BY-NC-4.0. Ranked by pools whose
centre sits under eligible imagery.

    python scripts/select_global_pool_aois.py --region Africa --output data/pool-aoi-selection.json --overwrite
    python scripts/select_global_pool_aois.py --aois scripts/pool_aoi_candidates.json --output data/pool-aoi-selection.json --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from spd_hot.tiles import zoom_for_gsd_m

DEFAULT_CONTINENTS = Path(__file__).with_name("pool_continent_bboxes.json")
OAM_SEARCH_URL = "https://api.imagery.hotosm.org/stac/search"
OVERPASS_URLS = (
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
USER_AGENT = "spd-hot-pool-aoi-preflight/1.0 (metadata-only)"
REQUEST_TIMEOUT_S = (
    60  # Overpass answers in tens of seconds from some networks; the fallback endpoint is tried next
)
# Every licence OAM indexes today. This pack is humanitarian / non-commercial, so NC is included.
ELIGIBLE_LICENSES = frozenset({"CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0"})


def load_aois(path: Path) -> list[dict[str, Any]]:
    """Read and validate named WGS84 AOI bboxes in west,south,east,north order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("AOI file must contain a non-empty JSON list")

    validated: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, candidate in enumerate(payload):
        if not isinstance(candidate, dict):
            raise TypeError(f"AOI #{index + 1} must be a JSON object")
        name = candidate.get("name")
        region = candidate.get("region")
        bbox = candidate.get("bbox")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"AOI #{index + 1} needs a non-empty name")
        if name in names:
            raise ValueError(f"AOI name {name!r} is duplicated")
        if not isinstance(region, str) or not region.strip():
            raise ValueError(f"AOI {name!r} needs a region")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"AOI {name!r} bbox must be [west, south, east, north]")
        try:
            west, south, east, north = (float(value) for value in bbox)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"AOI {name!r} bbox must contain four numbers") from exc
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(f"AOI {name!r} bbox is outside valid WGS84 bounds")
        names.add(name)
        validated.append({"name": name.strip(), "region": region.strip(), "bbox": [west, south, east, north]})
    return validated


def iter_cells(bbox: list[float], size_deg: float) -> list[list[float]]:
    """Cover `bbox` with [west, south, east, north] cells of `size_deg` (last row/column may be smaller)."""
    if size_deg <= 0:
        raise ValueError("cell size must be positive")
    west, south, east, north = bbox
    cells: list[list[float]] = []
    lat = south
    while lat < north:
        next_lat = min(lat + size_deg, north)
        lon = west
        while lon < east:
            next_lon = min(lon + size_deg, east)
            if next_lon > lon and next_lat > lat:
                cells.append([lon, lat, next_lon, next_lat])
            lon = next_lon
        lat = next_lat
    return cells


def _pool_count_query(bbox: list[float]) -> str:
    west, south, east, north = bbox
    return f"[out:json][timeout:25];way[leisure=swimming_pool]({south},{west},{north},{east});out count;"


def osm_pool_count(bbox: list[float]) -> int:
    """How many OSM pool ways are in the bbox (count only; no geometries)."""
    encoded = urlencode({"data": _pool_count_query(bbox)}).encode("ascii")
    errors: list[str] = []
    for url in OVERPASS_URLS:
        try:
            response = _request_json(
                url,
                method="POST",
                body=encoded,
                content_type="application/x-www-form-urlencoded",
            )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        return _overpass_count(response)
    raise URLError("; ".join(errors))


def _overpass_count(response: dict[str, Any]) -> int:
    elements = response.get("elements")
    if not isinstance(elements, list) or not elements:
        return 0
    tags = elements[0].get("tags") if isinstance(elements[0], dict) else None
    if not isinstance(tags, dict):
        return 0
    raw = tags.get("ways", tags.get("total", 0))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def discover_pool_cells(
    continents: list[dict[str, Any]],
    *,
    coarse_deg: float,
    cell_deg: float,
    min_pools: int,
    min_coarse_pools: int,
    delay_s: float,
    max_cells: int,
    count_fn: Any = None,
) -> list[dict[str, Any]]:
    """OSM-first AOIs: only split a coarse cell if it has `min_coarse_pools` ways.

    Fine cells need `min_pools` ways. The densest fine cells are kept, capped at `max_cells`.
    OAM is not queried here — only cheap Overpass counts.
    """
    counter = count_fn or osm_pool_count
    found: list[dict[str, Any]] = []
    for continent in continents:
        region = continent["region"]
        for coarse in iter_cells(continent["bbox"], coarse_deg):
            try:
                coarse_n = counter(coarse)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"[{region}] coarse count failed {coarse}: {exc}", file=sys.stderr, flush=True)
                continue
            if delay_s:
                time.sleep(delay_s)
            if coarse_n < min_coarse_pools:
                continue
            print(
                f"[{region}] {coarse_n} pool ways in coarse {coarse}; splitting",
                file=sys.stderr,
                flush=True,
            )
            fines = iter_cells(coarse, cell_deg) if cell_deg < coarse_deg else [coarse]
            for fine in fines:
                try:
                    n = counter(fine) if fine != coarse else coarse_n
                except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                    print(f"[{region}] cell count failed {fine}: {exc}", file=sys.stderr, flush=True)
                    continue
                if fine != coarse and delay_s:
                    time.sleep(delay_s)
                if n < min_pools:
                    continue
                west, south, east, north = fine
                found.append(
                    {
                        "name": f"{region} {west:.2f},{south:.2f},{east:.2f},{north:.2f}",
                        "region": region,
                        "bbox": fine,
                        "osm_pool_way_count_cell": n,
                    }
                )
                print(f"[{region}] {n} pool ways in {found[-1]['name']}", file=sys.stderr, flush=True)
    found.sort(key=lambda aoi: (-aoi["osm_pool_way_count_cell"], aoi["name"]))
    if max_cells > 0:
        found = found[:max_cells]
    return found


def _request_json(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object from {url}")
    return result


def _pool_centre_query(bbox: list[float]) -> str:
    west, south, east, north = bbox
    bounds = f"{south},{west},{north},{east}"
    return f"[out:json][timeout:60];way[leisure=swimming_pool]({bounds});out center;"


def osm_pool_centres(bbox: list[float]) -> list[tuple[float, float]]:
    """The centre (lon, lat) of every mapped pool way in the bbox. Relations are not counted: they are not downloaded."""
    encoded = urlencode({"data": _pool_centre_query(bbox)}).encode("ascii")
    errors: list[str] = []
    for url in OVERPASS_URLS:
        try:
            response = _request_json(
                url,
                method="POST",
                body=encoded,
                content_type="application/x-www-form-urlencoded",
            )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        return _overpass_centres(response)
    raise URLError("; ".join(errors))


def _overpass_centres(response: dict[str, Any]) -> list[tuple[float, float]]:
    elements = response.get("elements")
    if not isinstance(elements, list):
        raise TypeError("Overpass response did not contain elements")
    centres: list[tuple[float, float]] = []
    for element in elements:
        center = element.get("center") if isinstance(element, dict) else None
        if isinstance(center, dict) and "lon" in center and "lat" in center:
            centres.append((float(center["lon"]), float(center["lat"])))
    return centres


def find_oam_items(bbox: list[float], *, limit: int = 100) -> list[dict[str, Any]]:
    """Return OAM catalog item summaries intersecting the AOI, with no asset downloads."""
    body = json.dumps({"collections": ["openaerialmap"], "bbox": bbox, "limit": limit}).encode("utf-8")
    response = _request_json(OAM_SEARCH_URL, method="POST", body=body, content_type="application/json")
    features = response.get("features", [])
    if not isinstance(features, list):
        raise TypeError("OAM STAC response 'features' must be a list")
    items: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties") or {}
        # Only the fields the selection needs: contact details of uploaders are not copied into the report.
        gsd = properties.get("gsd", properties.get("proj:gsd"))
        footprint = feature.get("bbox")
        items.append(
            {
                "id": feature.get("id"),
                "license": properties.get("license"),
                "start_datetime": properties.get("start_datetime") or properties.get("datetime"),
                "gsd_m": gsd,
                "suggested_zoom": _suggested_zoom(gsd, footprint),
                "providers": [
                    provider.get("name")
                    for provider in properties.get("providers") or []
                    if isinstance(provider, dict) and provider.get("name")
                ],
                "footprint": footprint,
            }
        )
    return items


def _suggested_zoom(gsd: Any, footprint: Any) -> int | None:
    """XYZ zoom whose 256 px tile GSD matches the item, or None when GSD is missing."""
    if not isinstance(gsd, (int, float)) or isinstance(gsd, bool) or gsd <= 0:
        return None
    lat = 0.0
    if isinstance(footprint, list) and len(footprint) >= 4:
        try:
            lat = (float(footprint[1]) + float(footprint[3])) / 2
        except (TypeError, ValueError):
            lat = 0.0
    return zoom_for_gsd_m(float(gsd), lat)


def _in_footprint(point: tuple[float, float], footprint: list[float] | None) -> bool:
    if not footprint or len(footprint) < 4:
        return False
    lon, lat = point
    return footprint[0] <= lon <= footprint[2] and footprint[1] <= lat <= footprint[3]


def _pools_under(centres: list[tuple[float, float]], items: list[dict[str, Any]]) -> int:
    return sum(1 for centre in centres if any(_in_footprint(centre, item.get("footprint")) for item in items))


def inspect_aoi(aoi: dict[str, Any], *, delay_s: float = 1.0, limit: int = 100) -> dict[str, Any]:
    """Check that OSM pools lie inside fAIr-eligible OAM imagery of the AOI.

    A bbox intersection is not enough: the OAM footprints are small, so an AOI is selected only when at least one mapped
    pool's centre lies inside the footprint of a CC-BY-4.0, CC-BY-SA-4.0 or CC-BY-NC-4.0 item. Indoor and outdoor ways
    both count. The OSM way count is recorded but is not the score.
    """
    result: dict[str, Any] = {
        **aoi,
        "osm_pool_way_count": None,
        "osm_pools_under_eligible_imagery": None,
        "osm_pools_under_cc_by_imagery": None,
        "oam_items": [],
        "oam_eligible_items": [],
        "oam_cc_by_items": [],
        "oam_license_counts": {},
        "score": None,
        "status": "unverified",
    }
    errors: list[str] = []
    centres: list[tuple[float, float]] = []
    try:
        centres = osm_pool_centres(aoi["bbox"])
        result["osm_pool_way_count"] = len(centres)
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"OSM pools unavailable: {exc}")

    if delay_s:
        time.sleep(delay_s)
    try:
        items = find_oam_items(aoi["bbox"], limit=limit)
        result["oam_items"] = items
        result["oam_license_counts"] = dict(Counter(item.get("license") or "unknown" for item in items))
        result["oam_eligible_items"] = [item for item in items if item.get("license") in ELIGIBLE_LICENSES]
        result["oam_cc_by_items"] = [item for item in items if item.get("license") == "CC-BY-4.0"]
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"OAM search unavailable: {exc}")

    if centres or not errors:
        result["osm_pools_under_eligible_imagery"] = _pools_under(centres, result["oam_eligible_items"])
        result["osm_pools_under_cc_by_imagery"] = _pools_under(centres, result["oam_cc_by_items"])
        result["score"] = result["osm_pools_under_eligible_imagery"]

    under_count = result["osm_pools_under_eligible_imagery"]
    if under_count is not None and under_count > 0:
        result["status"] = "selected"
    elif errors:
        result["status"] = "unverified"
    else:
        result["status"] = "not_eligible"
    result["errors"] = errors
    return result


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order AOIs by pools under eligible imagery. City-wide OSM count is not used."""

    def overlap(result: dict[str, Any]) -> int:
        value = result.get("osm_pools_under_eligible_imagery")
        return value if isinstance(value, int) else -1

    ordered = sorted(results, key=lambda result: (-overlap(result), result["name"]))
    for index, result in enumerate(ordered, start=1):
        result["rank"] = index
    return ordered


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aois",
        type=Path,
        help="hand list of boxes; skips the OSM density scan (use after you already know the cells)",
    )
    parser.add_argument(
        "--continents",
        type=Path,
        default=DEFAULT_CONTINENTS,
        help="continent bboxes to grid when scanning OSM density",
    )
    parser.add_argument("--region", help="scan only this continent name/region (for a faster trial)")
    parser.add_argument(
        "--coarse-cell",
        type=float,
        default=5.0,
        help="degrees; empty coarse cells are not split (default: 5)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=1.0,
        help="degrees of each OSM-density AOI (default: 1, about 100 km)",
    )
    parser.add_argument(
        "--min-coarse-pools",
        type=int,
        default=2000,
        help="do not split a 5° cell unless it has this many OSM pool ways (default: 2000; 5000 can drop Cape Town)",
    )
    parser.add_argument(
        "--min-pools",
        type=int,
        default=1000,
        help="ignore 1° cells below this OSM way count (default: 1000)",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=10,
        help="after OSM counts, inspect OAM on only this many densest 1° cells (default: 10)",
    )
    parser.add_argument("--output", type=Path, help="JSON report path; parent directories are not created")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace --output if it already exists (needed for trial runs)",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between service calls (default: 1)")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="max OAM flights to list per cell (catalog only, not chips). Do not set this to 10; 100-500 is the catalog page.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.delay < 0:
        raise SystemExit("--delay must be zero or greater")
    if not 1 <= args.limit <= 1000:
        raise SystemExit("--limit must be between 1 and 1000")
    if args.coarse_cell <= 0 or args.cell_size <= 0:
        raise SystemExit("--coarse-cell and --cell-size must be positive")
    if args.min_pools < 1:
        raise SystemExit("--min-pools must be >= 1")
    if args.min_coarse_pools < 1:
        raise SystemExit("--min-coarse-pools must be >= 1")
    if args.max_cells < 1:
        raise SystemExit("--max-cells must be >= 1")

    try:
        if args.aois:
            aois = load_aois(args.aois)
            search = "hand list"
        else:
            continents = load_aois(args.continents)
            if args.region:
                wanted = args.region.strip().lower()
                continents = [
                    item
                    for item in continents
                    if item["name"].lower() == wanted or item["region"].lower() == wanted
                ]
                if not continents:
                    raise SystemExit(f"no continent matched --region {args.region!r}")
            print(
                f"OSM density scan: {len(continents)} continent(s), coarse {args.coarse_cell}° "
                f"(min {args.min_coarse_pools} ways), cell {args.cell_size}° (min {args.min_pools} ways), "
                f"OAM on top {args.max_cells}",
                file=sys.stderr,
                flush=True,
            )
            aois = discover_pool_cells(
                continents,
                coarse_deg=args.coarse_cell,
                cell_deg=args.cell_size,
                min_pools=args.min_pools,
                min_coarse_pools=args.min_coarse_pools,
                delay_s=args.delay,
                max_cells=args.max_cells,
            )
            search = "osm density"
            print(f"checking OAM on {len(aois)} densest cells", file=sys.stderr, flush=True)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid AOI file: {exc}") from exc

    if not aois:
        raise SystemExit("no OSM-dense cells found; try a smaller --min-pools or another --region")

    results = rank_results([inspect_aoi(aoi, delay_s=args.delay, limit=args.limit) for aoi in aois])
    report = {
        "search": search,
        "label_query": (
            "OSM leisure=swimming_pool ways (indoor and outdoor); centres used only to test "
            "imagery overlap; training labels are the way polygons. Nodes and relations excluded."
        ),
        "imagery_query": (
            "OAM STAC collection openaerialmap; CC-BY-4.0, CC-BY-SA-4.0 and CC-BY-NC-4.0 items eligible"
        ),
        "eligible_licenses": sorted(ELIGIBLE_LICENSES),
        "ranked_by": "osm_pools_under_eligible_imagery",
        "downloads_performed": False,
        "results": results,
        "selected_count": sum(result["status"] == "selected" for result in results),
        "unverified_count": sum(result["status"] == "unverified" for result in results),
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        if args.output.exists() and not args.overwrite:
            raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["selected_count"] > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
