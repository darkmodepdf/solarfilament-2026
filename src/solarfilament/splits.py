"""Deterministic folds, with physical-observation/date groups kept intact."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random

import numpy as np


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_splits(manifest: dict, n_folds: int = 3, seed: int = 2026) -> dict:
    """Balance year, instrument and instance density without separating groups.

    The balancing algorithm uses only permitted training annotations. Every pass
    of one physical observation inherits that observation's fold. ``group_id``
    already incorporates date/hash groups and any explicit near-duplicate unions.
    """
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    groups: dict[str, list[dict]] = defaultdict(list)
    ids = set()
    for item in manifest["images"]:
        if item["physical_id"] in ids:
            raise ValueError("Duplicate physical_id in manifest")
        ids.add(item["physical_id"])
        groups[item["group_id"]].append(item)
    if len(groups) < n_folds:
        raise ValueError("Fewer independent groups than folds")

    counts = {}
    for group_id, items in sorted(groups.items()):
        features = Counter({"total": len(items)})
        for item in items:
            density = np.mean([len(p["annotations"]) for p in item["passes"]])
            bucket = "low" if density < 5 else "medium" if density < 15 else "high"
            features[f"year:{item['capture_date'][:4]}"] += 1
            features[f"instrument:{item['instrument']}"] += 1
            features[f"density:{bucket}"] += 1
        counts[group_id] = features
    keys = sorted({k for value in counts.values() for k in value})
    vectors = {g: np.array([c[k] for k in keys], dtype=float) for g, c in counts.items()}
    totals = sum(vectors.values())
    target = totals / n_folds
    # Each feature contributes comparably; total size has extra weight.
    weights = 1 / np.maximum(target, 1)
    weights[keys.index("total")] *= 4
    rng = random.Random(seed)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: (-len(groups[g]), -float((vectors[g] / np.maximum(totals, 1)).sum())))
    fold_vectors = np.zeros((n_folds, len(keys)), dtype=float)
    group_fold = {}
    for i, group_id in enumerate(order):
        candidates = list(range(n_folds))
        rng.shuffle(candidates)
        # Seed every fold. Remaining assignments minimize the incremental error.
        if i < n_folds:
            candidates = [f for f in candidates if not fold_vectors[f].any()]
        vector = vectors[group_id]
        chosen = min(candidates, key=lambda f: float((weights * ((fold_vectors[f] + vector - target) ** 2 - (fold_vectors[f] - target) ** 2)).sum()))
        fold_vectors[chosen] += vector
        group_fold[group_id] = chosen
    assignments = {item["physical_id"]: group_fold[item["group_id"]] for item in sorted(manifest["images"], key=lambda x: x["physical_id"])}
    data_hash = manifest.get("manifest_hash", manifest.get("annotation_sha256"))
    result = {
        "version": 1,
        "seed": seed,
        "n_folds": n_folds,
        "data_hash": data_hash,
        "assignments": assignments,
        "groups": {g: {"fold": group_fold[g], "physical_ids": sorted(i["physical_id"] for i in items)} for g, items in sorted(groups.items())},
        "fold_summary": [{"fold": f, **{k: int(v) for k, v in zip(keys, fold_vectors[f])}} for f in range(n_folds)],
    }
    result["split_hash"] = _digest({"seed": seed, "n_folds": n_folds, "data_hash": data_hash, "assignments": assignments})
    return result


def validate_splits(manifest: dict, splits: dict) -> None:
    """Fail if identities, data hash, date/hash isolation, or fold coverage drift."""
    assignments = splits["assignments"]
    if set(assignments) != {i["physical_id"] for i in manifest["images"]}:
        raise ValueError("Split identities do not match manifest")
    if splits["data_hash"] != manifest.get("manifest_hash", manifest.get("annotation_sha256")):
        raise ValueError("Split data hash differs from manifest")
    seen: dict[tuple[str, str], int] = {}
    for item in manifest["images"]:
        fold = assignments[item["physical_id"]]
        if not isinstance(fold, int) or not 0 <= fold < splits["n_folds"]:
            raise ValueError("Invalid fold index")
        for field in ("group_id", "capture_date", "sha256"):
            if field not in item:
                continue
            key = field, item[field]
            if key in seen and seen[key] != fold:
                raise ValueError(f"Leakage across folds for {field}: {item[field]}")
            seen[key] = fold
    expected = _digest({"seed": splits["seed"], "n_folds": splits["n_folds"], "data_hash": splits["data_hash"], "assignments": assignments})
    if expected != splits["split_hash"]:
        raise ValueError("Split hash mismatch")
