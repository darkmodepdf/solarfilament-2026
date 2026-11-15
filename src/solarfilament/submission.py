"""Deterministic submission export and strict, coverage-aware verification."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np

from .rle import binary_mask, decode_rle, encode_rle


COLUMNS = ["filament_id", "segmentation_rle"]


def _image_ids(values: Iterable[str]) -> set[str]:
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError("Duplicate image identifiers in coverage manifest")
    for value in values:
        if not isinstance(value, str) or not value or "_" in value or Path(value).name != value:
            raise ValueError(f"Invalid image stem: {value!r}; official parser splits at first underscore")
    return set(values)


def validate_submission(
    path: str | Path, expected_images: Iterable[str],
    processed_images: Iterable[str] | None = None,
    shape: tuple[int, int] = (2048, 2048), *, allow_overlap: bool = False,
) -> dict:
    """Check CSV structure, mask bounds, coverage, overlap and exact RLE round trips.

    Zero-detection images cannot appear in CSV rows. A complete processed-image
    manifest is mandatory so omitted images cannot silently pass verification.
    """
    path = Path(path)
    expected = _image_ids(expected_images)
    if not expected:
        raise ValueError("Expected-image manifest is empty")
    if processed_images is None:
        raise ValueError("processed_images is required to verify zero-detection coverage")
    processed = _image_ids(processed_images)
    if processed != expected:
        raise ValueError(f"Coverage mismatch: missing={sorted(expected - processed)}, extra={sorted(processed - expected)}")
    seen, occupied, counts = set(), {}, {image: 0 for image in sorted(expected)}
    areas, overlap_pixels = [], 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Expected CSV columns {COLUMNS}, got {reader.fieldnames}")
        for line, row in enumerate(reader, start=2):
            if set(row) != set(COLUMNS) or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {line}")
            filament_id, encoded = row["filament_id"], row["segmentation_rle"]
            if filament_id in seen:
                raise ValueError(f"Duplicate filament ID on row {line}: {filament_id}")
            seen.add(filament_id)
            if "_" not in filament_id:
                raise ValueError(f"Missing instance suffix on row {line}")
            stem, suffix = filament_id.split("_", 1)
            if stem not in expected or not suffix:
                raise ValueError(f"Unknown image or empty suffix on row {line}: {filament_id}")
            mask = decode_rle(encoded, shape)
            area = int(mask.sum())
            if not area:
                raise ValueError(f"Empty filament mask on row {line}")
            reencoded = encode_rle(mask)
            if not np.array_equal(mask, decode_rle(reencoded, shape)):
                raise ValueError(f"RLE round trip failed on row {line}")
            packed = np.packbits(mask.reshape(-1))
            if stem in occupied:
                overlap = np.bitwise_and(occupied[stem], packed)
                if overlap.any():
                    if not allow_overlap:
                        raise ValueError(f"Overlapping predicted instances in image {stem}")
                    overlap_pixels += int(np.unpackbits(overlap).sum())
                occupied[stem] |= packed
            else:
                occupied[stem] = packed
            counts[stem] += 1
            areas.append(area)
    return {
        "valid": True, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(shape), "n_images": len(expected), "n_instances": len(seen),
        "zero_detection_images": [stem for stem, count in counts.items() if not count],
        "instances_per_image": counts, "overlap_pixels": overlap_pixels,
        "min_area": min(areas) if areas else 0, "max_area": max(areas) if areas else 0,
        "note": "File checks do not establish Kaggle acceptance or eligibility.",
    }


def write_submission(
    predictions: Mapping[str, Sequence[np.ndarray]], path: str | Path,
    shape: tuple[int, int] = (2048, 2048), *, expected_images: Iterable[str] | None = None,
) -> dict:
    """Write masks in stable image/component order and save the coverage sidecar.

    Include every intended image as a key, with [] for zero detections. Instances
    are ordered by their first foreground pixel, area and encoded counts.
    """
    expected = _image_ids(predictions.keys() if expected_images is None else expected_images)
    return write_submission_stream(
        ((stem, predictions[stem]) for stem in sorted(predictions)), path, sorted(expected), shape,
    )


def write_submission_stream(
    predictions: Iterable[tuple[str, Sequence[np.ndarray]]], path: str | Path,
    expected_images: Iterable[str], shape: tuple[int, int] = (2048, 2048),
) -> dict:
    """Export one full image at a time without retaining a dataset's masks.

    Iterate in sorted image order for byte-reproducible output. Per-image instance
    order is normalized by the writer. Duplicate, omitted and unexpected image
    keys fail atomically; yield ``(stem, [])`` to account for zero detections.
    """
    path = Path(path)
    expected = _image_ids(expected_images)
    processed: set[str] = set()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for stem, masks in predictions:
                _image_ids([stem])
                if stem not in expected:
                    raise ValueError(f"Unexpected image in prediction stream: {stem}")
                if stem in processed:
                    raise ValueError(f"Duplicate image in prediction stream: {stem}")
                processed.add(stem)
                instances = []
                occupied = np.zeros(shape, dtype=bool)
                for value in masks:
                    mask = binary_mask(value, shape)
                    area = int(mask.sum())
                    if not area:
                        raise ValueError(f"Empty instance in image {stem}")
                    if (occupied & mask).any():
                        raise ValueError(f"Overlapping instances in image {stem}")
                    occupied |= mask
                    encoded = encode_rle(mask)
                    if not np.array_equal(decode_rle(encoded, shape), mask):
                        raise ValueError(f"RLE changed image orientation for {stem}")
                    first_pixel = int(np.flatnonzero(mask)[0])
                    instances.append((first_pixel, area, encoded))
                for index, (_, _, encoded) in enumerate(sorted(instances), start=1):
                    writer.writerow({"filament_id": f"{stem}_{index}", "segmentation_rle": encoded})
        result = validate_submission(temporary_path, sorted(expected), sorted(processed), shape)
        os.replace(temporary_path, path)
        result["path"] = str(path)
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        sidecar.write_text(json.dumps({"processed_images": sorted(processed), **result}, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        temporary_path.unlink(missing_ok=True)
