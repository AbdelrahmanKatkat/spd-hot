import json
from pathlib import Path

import pytest

from scripts.select_global_pool_aois import (
    discover_pool_cells,
    inspect_aoi,
    iter_cells,
    load_aois,
    rank_results,
)

AOI = {"name": "test", "region": "test", "bbox": [0, 0, 1, 1]}


def test_load_global_candidates_are_unique_and_globally_distributed():
    aois = load_aois(Path("scripts/pool_aoi_candidates.json"))

    assert len(aois) == 6
    assert len({aoi["region"] for aoi in aois}) == 6
    assert len({aoi["name"] for aoi in aois}) == 6


def test_continent_bboxes_cover_six_regions():
    continents = load_aois(Path("scripts/pool_continent_bboxes.json"))

    assert len(continents) == 6
    assert {item["region"] for item in continents} == {
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "Oceania",
        "South America",
    }


def test_iter_cells_covers_the_bbox_without_gaps():
    cells = iter_cells([0.0, 0.0, 2.0, 2.0], 1.0)

    assert cells == [
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
    ]


def test_discover_keeps_dense_cells_and_caps_them():
    def fake_count(bbox: list[float]) -> int:
        west, south, east, _north = bbox
        if east - west > 1.5:
            return 40
        if west == 1.0 and south == 1.0:
            return 30
        if west == 0.0 and south == 0.0:
            return 15
        return 0

    continent = {"name": "Test", "region": "Test", "bbox": [0.0, 0.0, 2.0, 2.0]}
    found = discover_pool_cells(
        [continent],
        coarse_deg=2.0,
        cell_deg=1.0,
        min_pools=15,
        min_coarse_pools=40,
        delay_s=0,
        max_cells=1,
        count_fn=fake_count,
    )

    assert len(found) == 1
    assert found[0]["bbox"] == [1.0, 1.0, 2.0, 2.0]
    assert found[0]["osm_pool_way_count_cell"] == 30


def test_sparse_coarse_cell_is_not_split():
    def fake_count(bbox: list[float]) -> int:
        west, _south, east, _north = bbox
        if east - west > 1.5:
            return 15
        return 100

    continent = {"name": "Test", "region": "Test", "bbox": [0.0, 0.0, 2.0, 2.0]}
    found = discover_pool_cells(
        [continent],
        coarse_deg=2.0,
        cell_deg=1.0,
        min_pools=15,
        min_coarse_pools=1000,
        delay_s=0,
        max_cells=10,
        count_fn=fake_count,
    )

    assert found == []


@pytest.mark.parametrize(
    "bbox",
    [
        [-181, 0, 0, 1],
        [0, -91, 1, 0],
        [2, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1],
    ],
)
def test_load_rejects_invalid_bbox(tmp_path, bbox):
    path = tmp_path / "aois.json"
    path.write_text(json.dumps([{"name": "bad", "region": "test", "bbox": bbox}]), encoding="utf-8")

    with pytest.raises(ValueError, match="bbox"):
        load_aois(path)


def test_inspect_aoi_forwards_stac_limit(monkeypatch):
    captured: dict[str, int] = {}

    def fake_find(bbox, limit=100):
        captured["limit"] = limit
        return []

    monkeypatch.setattr("scripts.select_global_pool_aois.osm_pool_centres", lambda bbox: [])
    monkeypatch.setattr("scripts.select_global_pool_aois.find_oam_items", fake_find)

    inspect_aoi(AOI, delay_s=0, limit=500)

    assert captured["limit"] == 500


def test_aoi_selected_when_a_pool_lies_inside_cc_by_imagery(monkeypatch):
    monkeypatch.setattr(
        "scripts.select_global_pool_aois.osm_pool_centres", lambda bbox: [(0.5, 0.5), (0.9, 0.9)]
    )
    monkeypatch.setattr(
        "scripts.select_global_pool_aois.find_oam_items",
        lambda bbox, limit=100: [
            {"id": "nc-image", "license": "CC-BY-NC-4.0", "footprint": [0.4, 0.4, 0.6, 0.6]},
            {"id": "cc-by-image", "license": "CC-BY-4.0", "footprint": [0.4, 0.4, 0.6, 0.6]},
        ],
    )

    result = inspect_aoi(AOI, delay_s=0)

    assert result["status"] == "selected"
    assert result["score"] == 1
    assert result["osm_pools_under_eligible_imagery"] == 1
    assert result["osm_pools_under_cc_by_imagery"] == 1
    assert [item["id"] for item in result["oam_eligible_items"]] == ["nc-image", "cc-by-image"]
    assert [item["id"] for item in result["oam_cc_by_items"]] == ["cc-by-image"]


def test_cc_by_nc_imagery_selects_the_aoi(monkeypatch):
    monkeypatch.setattr("scripts.select_global_pool_aois.osm_pool_centres", lambda bbox: [(0.5, 0.5)])
    monkeypatch.setattr(
        "scripts.select_global_pool_aois.find_oam_items",
        lambda bbox, limit=100: [
            {"id": "nc-image", "license": "CC-BY-NC-4.0", "footprint": [0.4, 0.4, 0.6, 0.6]}
        ],
    )

    result = inspect_aoi(AOI, delay_s=0)

    assert result["status"] == "selected"
    assert result["osm_pools_under_eligible_imagery"] == 1
    assert result["osm_pools_under_cc_by_imagery"] == 0
    assert [item["id"] for item in result["oam_eligible_items"]] == ["nc-image"]


def test_imagery_that_misses_every_pool_does_not_select_the_aoi(monkeypatch):
    monkeypatch.setattr("scripts.select_global_pool_aois.osm_pool_centres", lambda bbox: [(0.9, 0.9)])
    monkeypatch.setattr(
        "scripts.select_global_pool_aois.find_oam_items",
        lambda bbox, limit=100: [
            {"id": "cc-by-image", "license": "CC-BY-4.0", "footprint": [0.0, 0.0, 0.2, 0.2]}
        ],
    )

    result = inspect_aoi(AOI, delay_s=0)

    assert result["status"] == "not_eligible"
    assert result["osm_pools_under_cc_by_imagery"] == 0


def test_aoi_not_selected_without_osm_pools(monkeypatch):
    monkeypatch.setattr("scripts.select_global_pool_aois.osm_pool_centres", lambda bbox: [])
    monkeypatch.setattr(
        "scripts.select_global_pool_aois.find_oam_items",
        lambda bbox, limit=100: [{"id": "cc-by-image", "license": "CC-BY-4.0", "footprint": [0, 0, 1, 1]}],
    )

    assert inspect_aoi(AOI, delay_s=0)["status"] == "not_eligible"


def test_aoi_reports_unverified_when_source_query_fails(monkeypatch):
    def unavailable(_bbox):
        raise TimeoutError("offline")

    monkeypatch.setattr("scripts.select_global_pool_aois.osm_pool_centres", unavailable)
    monkeypatch.setattr("scripts.select_global_pool_aois.find_oam_items", lambda bbox, limit=100: [])

    result = inspect_aoi(AOI, delay_s=0)

    assert result["status"] == "unverified"
    assert "OSM pools unavailable" in result["errors"][0]


def test_cc_by_sa_imagery_selects_the_aoi(monkeypatch):
    monkeypatch.setattr("scripts.select_global_pool_aois.osm_pool_centres", lambda bbox: [(0.5, 0.5)])
    monkeypatch.setattr(
        "scripts.select_global_pool_aois.find_oam_items",
        lambda bbox, limit=100: [
            {"id": "sa-image", "license": "CC-BY-SA-4.0", "footprint": [0.4, 0.4, 0.6, 0.6]}
        ],
    )

    result = inspect_aoi(AOI, delay_s=0)

    assert result["status"] == "selected"
    assert result["osm_pools_under_eligible_imagery"] == 1
    assert result["osm_pools_under_cc_by_imagery"] == 0
    assert [item["id"] for item in result["oam_eligible_items"]] == ["sa-image"]


def test_rank_uses_overlap_not_city_wide_osm_count():
    madrid = {
        "name": "Madrid",
        "status": "not_eligible",
        "osm_pool_way_count": 27053,
        "osm_pools_under_eligible_imagery": 0,
    }
    manila = {
        "name": "Manila",
        "status": "selected",
        "osm_pool_way_count": 1739,
        "osm_pools_under_eligible_imagery": 55,
    }
    bogota = {
        "name": "Bogota",
        "status": "selected",
        "osm_pool_way_count": 60,
        "osm_pools_under_eligible_imagery": 9,
    }

    ranked = rank_results([madrid, bogota, manila])

    assert [result["name"] for result in ranked] == ["Manila", "Bogota", "Madrid"]
    assert [result["rank"] for result in ranked] == [1, 2, 3]
    assert ranked[0]["osm_pool_way_count"] < ranked[2]["osm_pool_way_count"]
