"""The organizer notebook's pooled, all-hit-pairs PQ semantics.

This is intentionally not a Hungarian matching implementation. Reference source
and its exact checksum are recorded in docs/official-metric.md.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_util

from .rle import binary_mask


def overlap_matrices(gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray]):
    """Pairwise IoU/Dice, with at most one generated component mask at a time.

    Both arguments may be lazy mask sequences backed by a shared label image.
    Encode sequentially to avoid retaining N full-resolution boolean arrays.
    The compressed RLE lists and the N-GT by N-pred overlap matrices remain.
    """
    expected_shape = None

    def encode_masks(masks):
        nonlocal expected_shape
        rles = []
        for value in masks:
            mask = binary_mask(value)
            if expected_shape is None:
                expected_shape = mask.shape
            elif mask.shape != expected_shape:
                raise ValueError("Ground truth and predicted masks must have identical dimensions")
            rles.append(mask_util.encode(np.asfortranarray(mask, dtype=np.uint8)))
            # A Sequence may allocate its next component on demand. Release the
            # prior component before that allocation, not after assigning it.
            del mask, value
        return rles

    gt_rles = encode_masks(gt_masks)
    pred_rles = encode_masks(pred_masks)
    if not gt_rles or not pred_rles:
        empty = np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float32)
        return empty, empty.copy()
    # pycocotools returns predictions x ground truths. Float32 matches the
    # notebook tensor math at the strict 0.5 match boundary.
    iou = np.asarray(mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float32).T
    iou = np.nan_to_num(iou, nan=0.0)
    dice = 2 * iou / (1 + iou)
    return iou, dice


def _scores(tp: int, fp: int, fn: int, iou_sum: float) -> dict:
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": float(iou_sum / denominator) if denominator else 0.0,
        "sq": float(iou_sum / tp) if tp else 0.0,
        "rq": float(tp / denominator) if denominator else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "iou_sum": float(iou_sum),
    }


def panoptic_quality(gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray]) -> dict:
    """Score one annotation pass; matches every pair with IoU strictly > 0.5.

    Empty-vs-empty is 0, per the reference core function. Split/merge diagnostics
    count any positive overlap, separately from the PQ match threshold.
    """
    iou, dice = overlap_matrices(gt_masks, pred_masks)
    hits = iou > 0.5
    tp = int(hits.sum())
    fp = int(np.sum(hits.sum(axis=0) == 0))
    fn = int(np.sum(hits.sum(axis=1) == 0))
    scores = _scores(tp, fp, fn, sum(float(x) for x in iou[hits]))
    overlap = iou > 0
    scores.update({
        "n_gt": len(gt_masks), "n_pred": len(pred_masks),
        "split_count": int(np.sum(overlap.sum(axis=1) > 1)),
        "merge_count": int(np.sum(overlap.sum(axis=0) > 1)),
        "matched_ious": [float(x) for x in iou[hits]],
        "positive_ious": [float(x) for x in iou[overlap]],
        "positive_dice": [float(x) for x in dice[overlap]],
    })
    return scores


def aggregate_scores(scores: Sequence[Mapping]) -> dict:
    result = _scores(
        sum(s["tp"] for s in scores), sum(s["fp"] for s in scores),
        sum(s["fn"] for s in scores), sum(s["iou_sum"] for s in scores),
    )
    result.update({
        "n_passes": len(scores),
        "macro_pass_pq": float(np.mean([s["pq"] for s in scores])) if scores else 0.0,
        "split_count": sum(s.get("split_count", 0) for s in scores),
        "merge_count": sum(s.get("merge_count", 0) for s in scores),
        "matched_ious": [x for s in scores for x in s.get("matched_ious", [])],
        "positive_ious": [x for s in scores for x in s.get("positive_ious", [])],
        "positive_dice": [x for s in scores for x in s.get("positive_dice", [])],
    })
    return result


def evaluate_passes(records: Iterable[Mapping], *, include_empty_gt: bool = False) -> dict:
    """Pool original annotation passes, never concatenate their instance lists.

    Each record has image_id, annotation_pass_id, gt_masks, and pred_masks. By
    default, passes with no GT filaments are excluded to mirror the reference
    CSV wrapper, which discovers passes from GT rows only. Coverage is reported.
    """
    scores, per_image, seen = [], defaultdict(list), set()
    excluded_empty_gt = 0
    for record in records:
        key = (str(record["image_id"]), str(record["annotation_pass_id"]))
        if key in seen:
            raise ValueError(f"Duplicate annotation pass: {key}")
        seen.add(key)
        if not len(record["gt_masks"]) and not include_empty_gt:
            excluded_empty_gt += 1
            continue
        score = panoptic_quality(record["gt_masks"], record["pred_masks"])
        score.update({"image_id": key[0], "annotation_pass_id": key[1]})
        scores.append(score)
        per_image[key[0]].append(score)
    result = aggregate_scores(scores)
    result.update({
        "metric": "organizer_pooled_all_pairs_pq_iou_gt_0.5",
        "per_pass": scores,
        "per_image": {key: aggregate_scores(value) for key, value in sorted(per_image.items())},
        "excluded_empty_gt_passes": excluded_empty_gt,
        "empty_gt_policy": "include" if include_empty_gt else "official_csv_universe",
    })
    return result
