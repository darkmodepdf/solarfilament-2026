"""Synthetic contract tests plus parity against the pinned organizer source."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from solarfilament.metrics import evaluate_passes, overlap_matrices, panoptic_quality
from solarfilament.rle import decode_rle, encode_rle, segmentation_to_mask
from solarfilament.submission import validate_submission, write_submission, write_submission_stream


REFERENCE_SHA256 = "ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801"


def rectangle(y0=1, y1=5, x0=2, x1=7, shape=(11, 13)):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_asymmetric_rle_and_polygon():
    mask = rectangle()
    encoded = encode_rle(mask)
    assert isinstance(encoded, str)
    np.testing.assert_array_equal(decode_rle(encoded, mask.shape), mask)
    polygon = [[2, 1, 7, 1, 7, 5, 2, 5]]
    np.testing.assert_array_equal(segmentation_to_mask(polygon, *mask.shape), mask)
    np.testing.assert_array_equal(segmentation_to_mask({"size": list(mask.shape), "counts": encoded}, *mask.shape), mask)


@pytest.mark.parametrize("encoded", ["", "1", "~", "P", "-1", '"abc"'])
def test_malformed_rle_rejected_before_native_decode(encoded):
    with pytest.raises(ValueError):
        decode_rle(encoded, (11, 13))


def test_random_rle_round_trip_including_empty_and_full():
    rng = np.random.default_rng(9327)
    for shape in [(1, 1), (2, 7), (31, 29), (128, 131)]:
        for probability in [0, 0.01, 0.5, 0.99, 1]:
            mask = rng.random(shape) < probability
            np.testing.assert_array_equal(decode_rle(encode_rle(mask), shape), mask)


def test_pq_empty_missing_and_extra():
    mask, other = rectangle(), rectangle(7, 10, 9, 12)
    assert panoptic_quality([mask], [mask])["pq"] == 1
    assert panoptic_quality([mask], [])["fn"] == 1
    assert panoptic_quality([], [mask])["fp"] == 1
    assert panoptic_quality([], [])["pq"] == 0
    assert panoptic_quality([mask], [mask, other])["pq"] == pytest.approx(2 / 3)


def test_strict_threshold_and_split_merge():
    left, right = rectangle(1, 5, 1, 4), rectangle(1, 5, 4, 7)
    merged = left | right
    split = panoptic_quality([merged], [left, right])
    assert split["pq"] == 0  # Each IoU is exactly 0.5, not a match.
    assert (split["tp"], split["fp"], split["fn"], split["split_count"]) == (0, 2, 1, 1)
    merge = panoptic_quality([left, right], [merged])
    assert (merge["pq"], merge["merge_count"]) == (0, 1)


def test_reference_is_all_pairs_not_one_to_one():
    mask = rectangle()
    score = panoptic_quality([mask], [mask, mask])
    assert score["tp"] == 2
    assert score["fp"] == 0
    assert score["pq"] == 1


def test_passes_pool_counts_instead_of_image_means():
    mask, other = rectangle(), rectangle(7, 10, 9, 12)
    records = [
        {"image_id": "a", "annotation_pass_id": "one", "gt_masks": [mask], "pred_masks": [mask]},
        {"image_id": "a", "annotation_pass_id": "two", "gt_masks": [mask, other], "pred_masks": [mask]},
        {"image_id": "b", "annotation_pass_id": "one", "gt_masks": [], "pred_masks": [other]},
    ]
    result = evaluate_passes(records)
    assert result["pq"] == pytest.approx(2 / 2.5)
    assert result["excluded_empty_gt_passes"] == 1
    assert result["macro_pass_pq"] != result["pq"]
    assert evaluate_passes(records, include_empty_gt=True)["pq"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="Duplicate annotation"):
        evaluate_passes(records + records[:1])


@pytest.fixture(scope="module")
def original_functions():
    import pandas as pd
    import torch
    from pycocotools import mask as mask_util

    source = Path(__file__).resolve().parents[1] / "vendor/official/self-evaluation-notebook.ipynb"
    if not source.is_file():
        pytest.skip("Organizer source is not redistributed; download the pinned notebook to run parity checks")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == REFERENCE_SHA256
    notebook = json.loads(source.read_text(encoding="utf-8"))
    nodes = []
    allowed = {"get_overlap_matrices", "fp_count_hit", "fn_count_hit", "rles_to_layers", "process_entry", "get_overlap_df", "get_pq_score"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            nodes.extend(node for node in ast.parse("".join(cell["source"])).body if isinstance(node, ast.FunctionDef) and node.name in allowed)
    assert {node.name for node in nodes} == allowed
    namespace = {"pd": pd, "torch": torch, "np": np, "mask_util": mask_util}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def test_organizer_core_parity_on_synthetic_edges(original_functions):
    import pandas as pd
    import torch

    mask, other = rectangle(), rectangle(7, 10, 9, 12)
    half = mask.copy()
    half[3:] = False
    cases = [([mask], [mask]), ([mask], []), ([], [mask]), ([], []),
             ([mask], [mask, other]), ([mask], [half, mask & ~half]),
             ([mask, other], [mask | other]), ([mask], [mask, mask])]
    for gt, pred in cases:
        shape = mask.shape
        iou, dice = original_functions["get_overlap_matrices"](
            torch.from_numpy(np.asarray(gt, dtype=np.float32).reshape((-1, *shape))),
            torch.from_numpy(np.asarray(pred, dtype=np.float32).reshape((-1, *shape))),
        )
        ours_iou, ours_dice = overlap_matrices(gt, pred)
        np.testing.assert_allclose(ours_iou, iou.numpy(), atol=1e-7)
        np.testing.assert_allclose(ours_dice, dice.numpy(), atol=1e-7)
        overlap = pd.DataFrame([{"iou_matrix": iou, "n_gt": len(gt), "n_pred": len(pred)}])
        assert panoptic_quality(gt, pred)["pq"] == pytest.approx(original_functions["get_pq_score"](overlap), abs=1e-7)


def test_organizer_csv_pass_parity(original_functions):
    import pandas as pd

    # Exercise exact ID parsing and repeated predictions per independent annotator.
    mask = rectangle(shape=(2048, 2048))
    other = rectangle(20, 25, 30, 38, shape=(2048, 2048))
    ground_truth = pd.DataFrame([
        {"filament_id": "ann1-imageA_1", "segmentation_rle": encode_rle(mask)},
        {"filament_id": "ann2-imageA_1", "segmentation_rle": encode_rle(mask)},
        {"filament_id": "ann2-imageA_2", "segmentation_rle": encode_rle(other)},
    ])
    predictions = pd.DataFrame([
        {"filament_id": "imageA_1", "segmentation_rle": encode_rle(mask)},
        {"filament_id": "unknown_1", "segmentation_rle": encode_rle(other)},
    ])
    actual = original_functions["get_pq_score"](original_functions["get_overlap_df"](ground_truth, predictions))
    ours = evaluate_passes([
        {"image_id": "imageA", "annotation_pass_id": "ann1", "gt_masks": [mask], "pred_masks": [mask]},
        {"image_id": "imageA", "annotation_pass_id": "ann2", "gt_masks": [mask, other], "pred_masks": [mask]},
    ])
    assert ours["pq"] == pytest.approx(actual, abs=1e-7)


def test_submission_coverage_roundtrip_and_determinism(tmp_path):
    mask, other = rectangle(), rectangle(7, 10, 9, 12)
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    result = write_submission({"imageB": [], "imageA": [other, mask]}, first, mask.shape)
    write_submission({"imageA": [mask, other], "imageB": []}, second, mask.shape)
    assert first.read_bytes() == second.read_bytes()
    assert result["n_instances"] == 2
    assert result["zero_detection_images"] == ["imageB"]
    assert first.with_suffix(".csv.manifest.json").is_file()
    with pytest.raises(ValueError, match="Coverage mismatch"):
        validate_submission(first, ["imageA", "imageB"], ["imageA"], mask.shape)
    with pytest.raises(ValueError, match="processed_images"):
        validate_submission(first, ["imageA"], shape=mask.shape)


def test_submission_rejects_overlap_empty_unknown_and_wrong_shape(tmp_path):
    mask = rectangle()
    for masks, message in [([mask, mask], "Overlapping"), ([mask & False], "Empty"), ([mask.T], "shape")]:
        with pytest.raises(ValueError, match=message):
            write_submission({"image": masks}, tmp_path / "bad.csv", mask.shape)
    path = tmp_path / "good.csv"
    write_submission({"image": [mask]}, path, mask.shape)
    with pytest.raises(ValueError, match="Unknown image"):
        validate_submission(path, ["other"], ["other"], mask.shape)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(path.read_text().splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="Duplicate filament"):
        validate_submission(path, ["image"], ["image"], mask.shape)


def test_streaming_submission_matches_mapping_and_fails_atomically(tmp_path):
    mask = rectangle()
    mapping_path, stream_path = tmp_path / "mapping.csv", tmp_path / "stream.csv"
    write_submission({"a": [mask], "b": []}, mapping_path, mask.shape)
    write_submission_stream(iter([("a", [mask]), ("b", [])]), stream_path, ["a", "b"], mask.shape)
    assert mapping_path.read_bytes() == stream_path.read_bytes()
    before = stream_path.read_bytes()
    for records, message in [([("a", [])], "Coverage mismatch"),
                             ([("a", []), ("a", [])], "Duplicate image"),
                             ([("c", [])], "Unexpected image")]:
        with pytest.raises(ValueError, match=message):
            write_submission_stream(iter(records), stream_path, ["a", "b"], mask.shape)
        assert stream_path.read_bytes() == before


def test_competition_training_annotation_and_metric_parity(original_functions):
    """Optional real-annotation fixture; this does not measure a model's PQ."""
    from collections import defaultdict
    import pandas as pd
    from pycocotools.coco import COCO

    root = Path(__file__).resolve().parents[1]
    annotation_path = root / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not annotation_path.exists():
        pytest.skip("Competition training data is not attached")
    coco = COCO(str(annotation_path))
    for annotation in coco.dataset["annotations"][:20]:
        image = coco.imgs[annotation["image_id"]]
        ours = segmentation_to_mask(annotation["segmentation"], image["height"], image["width"])
        np.testing.assert_array_equal(ours, coco.annToMask(annotation).astype(bool))
    grouped = defaultdict(list)
    for image in coco.dataset["images"]:
        grouped[image["file_name"]].append(image)
    filename, images = next((name, passes) for name, passes in grouped.items() if len(passes) > 1)
    stem = Path(filename).stem
    ground_truth_rows, prediction_rows, records = [], [], []
    first_pass = [coco.annToMask(annotation).astype(bool) for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[images[0]["id"]]))]
    # Manufactured predictions deliberately include displacement and one omission.
    predictions = [np.roll(mask, 3, axis=1) for mask in first_pass[:-1]]
    for index, mask in enumerate(predictions):
        prediction_rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": encode_rle(mask)})
    for image in images:
        gt_masks = [coco.annToMask(annotation).astype(bool) for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image["id"]]))]
        records.append({"image_id": stem, "annotation_pass_id": image["id"], "gt_masks": gt_masks, "pred_masks": predictions})
        for index, mask in enumerate(gt_masks):
            ground_truth_rows.append({"filament_id": f"{image['id']}_{index}", "segmentation_rle": encode_rle(mask)})
    overlap = original_functions["get_overlap_df"](pd.DataFrame(ground_truth_rows), pd.DataFrame(prediction_rows, columns=["filament_id", "segmentation_rle"]))
    official = original_functions["get_pq_score"](overlap)
    ours = evaluate_passes(records)
    assert ours["pq"] == pytest.approx(official, abs=1e-7)
    assert ours["n_passes"] == len(images)


def test_lazy_component_sequences_do_not_retain_all_dense_masks(tmp_path):
    from collections.abc import Sequence
    import weakref

    class ObservedMasks(Sequence):
        def __init__(self, count):
            self.count = count
            self.refs = []
            self.max_live = 0

        def __len__(self):
            return self.count

        def __getitem__(self, index):
            if index < 0 or index >= self.count:
                raise IndexError(index)
            mask = np.zeros((64, 64), dtype=bool)
            mask[index, index] = True
            self.refs.append(weakref.ref(mask))
            self.max_live = max(self.max_live, sum(ref() is not None for ref in self.refs))
            assert self.max_live <= 2, "Consumer materialized multiple dense component masks"
            return mask

    gt, pred = ObservedMasks(60), ObservedMasks(60)
    scores = panoptic_quality(gt, pred)
    assert scores["pq"] == 1
    assert scores["tp"] == 60
    stream = ObservedMasks(60)
    report = write_submission_stream(iter([("test", stream)]), tmp_path / "lazy.csv", ["test"], (64, 64))
    assert report["n_instances"] == 60
    assert max(gt.max_live, pred.max_live, stream.max_live) <= 2
