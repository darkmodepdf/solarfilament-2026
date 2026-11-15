"""COCO-compatible binary masks and compressed counts, with strict decoding checks."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pycocotools import mask as mask_util


def binary_mask(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2 or (shape is not None and value.shape != tuple(shape)):
        raise ValueError(f"Expected a 2-D mask of shape {shape}; got {value.shape}")
    if value.dtype != np.bool_ and not np.isin(value, (0, 1)).all():
        raise ValueError("Masks must contain only binary 0/1 values")
    return value.astype(bool, copy=False)


def encode_rle(mask: np.ndarray) -> str:
    """Return ASCII compressed COCO counts, in column-major orientation."""
    value = binary_mask(mask)
    encoded = mask_util.encode(np.asfortranarray(value, dtype=np.uint8))
    return encoded["counts"].decode("ascii")


def _compressed_runs(counts: str, pixel_count: int) -> list[int]:
    """Validate the COCO byte stream before handing it to the native decoder.

    pycocotools does not reliably reject underfilled streams. Decode run lengths
    using the upstream maskApi.c signed, delta-coded variable-length format.
    """
    if not isinstance(counts, str) or not counts:
        raise ValueError("RLE counts must be a nonempty compressed ASCII string")
    runs: list[int] = []
    index = 0
    while index < len(counts):
        value, shift = 0, 0
        while True:
            if index >= len(counts):
                raise ValueError("Truncated compressed RLE")
            code = ord(counts[index]) - 48
            index += 1
            if code < 0 or code > 63:
                raise ValueError("Invalid character in compressed RLE")
            value |= (code & 0x1F) << shift
            shift += 5
            if shift > 65:
                raise ValueError("Compressed RLE integer is too large")
            if not code & 0x20:
                if code & 0x10:
                    value |= -1 << shift
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0 or value > pixel_count:
            raise ValueError("RLE run length is outside image bounds")
        runs.append(value)
    if sum(runs) != pixel_count:
        raise ValueError("RLE run lengths do not fill the requested image")
    return runs


def decode_rle(counts: str, shape: tuple[int, int] = (2048, 2048)) -> np.ndarray:
    shape = tuple(int(x) for x in shape)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Mask dimensions must be positive height and width")
    _compressed_runs(counts, shape[0] * shape[1])
    try:
        mask = mask_util.decode({"size": list(shape), "counts": counts.encode("ascii")})
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid COCO compressed RLE") from exc
    return binary_mask(mask, shape)


def segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Rasterize polygons or COCO RLE without bounding-box/skeleton supervision."""
    shape = (int(height), int(width))
    if min(shape) < 1:
        raise ValueError("Invalid image dimensions")
    if isinstance(segmentation, dict):
        if tuple(segmentation.get("size", shape)) != shape:
            raise ValueError("COCO RLE size disagrees with image dimensions")
        counts = segmentation.get("counts")
        if isinstance(counts, (str, bytes)):
            return decode_rle(counts.decode("ascii") if isinstance(counts, bytes) else counts, shape)
        if not isinstance(counts, list) or any(not isinstance(x, int) or x < 0 for x in counts):
            raise ValueError("Invalid uncompressed COCO counts")
        if sum(counts) != height * width:
            raise ValueError("Uncompressed counts do not fill image")
        rle = mask_util.frPyObjects({"size": list(shape), "counts": counts}, height, width)
    elif isinstance(segmentation, (list, tuple)):
        if not segmentation:
            return np.zeros(shape, dtype=bool)
        polygons: Sequence = segmentation
        if isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2 or not np.isfinite(polygon).all():
                raise ValueError("Polygon must have at least three finite coordinate pairs")
        rle = mask_util.merge(mask_util.frPyObjects(polygons, height, width))
    else:
        raise ValueError("Segmentation must be COCO polygons or RLE")
    return binary_mask(mask_util.decode(rle), shape)
