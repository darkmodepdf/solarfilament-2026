from copy import deepcopy

import pytest

from solarfilament.splits import make_splits, validate_splits


def _manifest():
    images = [{"physical_id": str(i), "group_id": str(i // 2), "sha256": str(i), "capture_date": f"2020-01-{i // 2 + 1:02d}", "instrument": "Ch" if i % 2 else "Mh", "passes": [{"annotations": [{}] * (i + 1)}]} for i in range(18)]
    return {"images": images, "manifest_hash": "example-data-hash"}


def test_group_isolation_reproducibility_and_coverage():
    manifest = _manifest()
    result = make_splits(manifest)
    validate_splits(manifest, result)
    assert result == make_splits(manifest)
    assert set(result["assignments"].values()) == {0, 1, 2}
    for i in range(0, 18, 2):
        assert result["assignments"][str(i)] == result["assignments"][str(i + 1)]
    counts = [sum(f == fold for f in result["assignments"].values()) for fold in range(3)]
    assert max(counts) - min(counts) <= 2
    assert len(result["split_hash"]) == 64


def test_hash_or_group_leak_is_rejected():
    manifest = _manifest()
    result = make_splits(manifest)
    bad = deepcopy(result)
    bad["assignments"]["0"] = (bad["assignments"]["1"] + 1) % 3
    with pytest.raises(ValueError, match="Leakage"):
        validate_splits(manifest, bad)
    bad = deepcopy(result)
    bad["split_hash"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_splits(manifest, bad)


def test_insufficient_groups_fail():
    with pytest.raises(ValueError, match="Fewer"):
        make_splits(_manifest(), n_folds=10)
