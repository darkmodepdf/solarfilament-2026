import json

import numpy as np
from PIL import Image
import pytest

from solarfilament.data import build_manifest, group_observations, load_manifest, rasterize_pass


def _fixture(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    names = ["20200101000000Ch.jpeg", "20200102000000Mh.jpeg"]
    for index, name in enumerate(names):
        Image.fromarray(np.full((8, 8), 40 + index, np.uint8)).save(root / name)
    rows = [{"id": "a-" + names[0][:-5], "file_name": names[0], "height": 8, "width": 8}, {"id": "b-" + names[0][:-5], "file_name": names[0], "height": 8, "width": 8}, {"id": "a-" + names[1][:-5], "file_name": names[1], "height": 8, "width": 8}]
    anns = [{"id": str(i), "image_id": r["id"], "category_id": 2, "spine": [1, 2, 3], "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]]} for i, r in enumerate(rows[:2])]
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"images": rows, "annotations": anns}))
    return annotations, root


def test_passes_kept_separate_and_only_masks_ingested(tmp_path):
    annotations, root = _fixture(tmp_path)
    path = tmp_path / "manifest.json"
    result = build_manifest(annotations, root, path)
    assert result["summary"]["physical_images"] == 2
    assert result["summary"]["annotation_passes"] == 3
    first = result["images"][0]
    assert [p["annotation_pass_id"] for p in first["passes"]] == ["a", "b"]
    assert all(len(p["annotations"]) == 1 for p in first["passes"])
    assert set(first["passes"][0]["annotations"][0]) == {"id", "segmentation"}
    assert result["images"][1]["passes"][0]["annotations"] == []
    assert load_manifest(path)["manifest_hash"] == result["manifest_hash"]
    assert first["passes"][0]["stats"]["foreground_area"] == 16
    assert len(result["near_duplicate_candidates"]) == 1


def test_coco_orientation_and_multiple_polygons_are_one_instance():
    record = {"annotations": [{"id": "x", "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6], [5, 0, 7, 0, 7, 1, 5, 1]]}]}
    masks = rasterize_pass(record, 8, 9)
    expected = np.zeros((8, 9), bool)
    expected[2:6, 1:4] = True
    expected[0:1, 5:7] = True
    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], expected)


def test_date_hash_and_explicit_groups_are_transitive():
    rows = [{"physical_id": "a", "capture_date": "2020-01-01", "sha256": "x"}, {"physical_id": "b", "capture_date": "2020-01-01", "sha256": "y"}, {"physical_id": "c", "capture_date": "2020-01-02", "sha256": "y"}, {"physical_id": "d", "capture_date": "2020-01-03", "sha256": "z"}]
    group_observations(rows, date_gap_days=0)
    assert [r["group_id"] for r in rows] == ["a", "a", "a", "d"]
    group_observations(rows, [["c", "d"]], date_gap_days=0)
    assert {r["group_id"] for r in rows} == {"a"}
    with pytest.raises(ValueError, match="missing"):
        group_observations(rows, [["unknown", "a"]])


def test_adjacent_dates_are_conservatively_grouped_by_default():
    rows = [{"physical_id": "a", "capture_date": "2020-01-01", "sha256": "x"}, {"physical_id": "b", "capture_date": "2020-01-02", "sha256": "y"}, {"physical_id": "c", "capture_date": "2020-01-03", "sha256": "z"}, {"physical_id": "d", "capture_date": "2020-01-05", "sha256": "q"}]
    group_observations(rows)
    assert [r["group_id"] for r in rows] == ["a", "a", "a", "d"]


def test_annotation_tampering_is_detected(tmp_path):
    annotations, root = _fixture(tmp_path)
    result = build_manifest(annotations, root)
    result["images"][0]["passes"][0]["annotations"][0]["segmentation"][0][0] = 4
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(path)


def test_bad_dimensions_are_rejected(tmp_path):
    annotations, root = _fixture(tmp_path)
    doc = json.loads(annotations.read_text())
    doc["images"][0]["height"] = 9
    annotations.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="dimension mismatch"):
        build_manifest(annotations, root)


def test_overlapping_instances_not_deduplicated(tmp_path):
    annotations, root = _fixture(tmp_path)
    doc = json.loads(annotations.read_text())
    extra = dict(doc["annotations"][0], id="new")
    doc["annotations"].append(extra)
    annotations.write_text(json.dumps(doc))
    result = build_manifest(annotations, root)
    first = result["images"][0]["passes"][0]
    assert len(rasterize_pass(first, 8, 8)) == 2
    assert first["stats"]["overlap_area"] == 16
