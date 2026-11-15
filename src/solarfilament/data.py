"""Competition-only COCO data audit; complete annotation passes stay separate."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_rle(annotation: dict, height: int, width: int) -> dict:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, dict):
        if list(segmentation.get("size", [])) != [height, width]:
            raise ValueError("Annotation RLE dimensions do not match image")
        if isinstance(segmentation["counts"], list):
            return coco_mask.frPyObjects(segmentation, height, width)
        counts = segmentation["counts"]
        return {"size": [height, width], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError(f"Missing polygons for annotation {annotation.get('id')}")
    # COCO stores each disconnected polygon as its own flat coordinate sequence.
    if isinstance(segmentation[0], (float, int)):
        raise ValueError("COCO polygons must be a list of coordinate lists")
    for polygon in segmentation:
        if len(polygon) < 6 or len(polygon) % 2 or not np.isfinite(polygon).all():
            raise ValueError(f"Invalid polygon for annotation {annotation.get('id')}")
    return coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))


def rasterize_pass(pass_record: dict, height: int, width: int) -> list[np.ndarray]:
    """Rasterize original instances with COCO conventions; never merge passes.

    All filament categories are one class. Empty masks are retained for audit,
    rather than silently deleting validation annotations.
    """
    return [coco_mask.decode(_annotation_rle(a, height, width)).astype(bool) for a in pass_record["annotations"]]


class _UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        self.parent[max(a, b)] = min(a, b)


def group_observations(images: list[dict], extra_groups: list[list[str]] | None = None, *, date_gap_days: int = 1) -> None:
    """Union exact hashes and same/adjacent-date observations across instruments.

    One-day temporal chains are conservative leakage protection: solar structures
    can survive between adjacent observations even while moving across the disk.
    Set zero only for an explicit same-day-only ablation, never implicitly.
    """
    if not isinstance(date_gap_days, int) or date_gap_days < 0:
        raise ValueError("date_gap_days must be a nonnegative integer")
    uf = _UnionFind(i["physical_id"] for i in images)
    for key in ("capture_date", "sha256"):
        first = {}
        for item in images:
            value = item[key]
            if value in first:
                uf.union(first[value], item["physical_id"])
            else:
                first[value] = item["physical_id"]
    ordered = sorted(images, key=lambda i: (i["capture_date"], i["physical_id"]))
    for first, second in zip(ordered, ordered[1:]):
        gap = (datetime.fromisoformat(second["capture_date"]) - datetime.fromisoformat(first["capture_date"])).days
        if gap <= date_gap_days:
            uf.union(first["physical_id"], second["physical_id"])
    for group in extra_groups or []:
        if not group or any(i not in uf.parent for i in group):
            raise ValueError("Near-duplicate override contains missing physical ids")
        for item in group[1:]:
            uf.union(group[0], item)
    for item in images:
        item["group_id"] = uf.find(item["physical_id"])


def manifest_digest(manifest: dict) -> str:
    """Hash data, annotations and grouping, independent of filesystem location."""
    identity = {
        "annotation_sha256": manifest["annotation_sha256"],
        "images": [{**{k: item[k] for k in ("physical_id", "sha256", "width", "height", "group_id")}, "passes": [{k: p[k] for k in ("annotation_pass_id", "image_id", "annotations")} for p in item["passes"]]} for item in manifest["images"]],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_manifest(annotations_path: str | Path, image_root: str | Path, output_path: str | Path | None = None, *, extra_groups: list[list[str]] | None = None, date_gap_days: int = 1) -> dict:
    """Audit official annotations/images and produce a portable data identity.

    Paths are recorded for convenience and can be replaced when moving to a new
    machine; ``manifest_hash`` deliberately excludes their absolute locations.
    Never ingest external annotations: only ``segmentation`` and ``id`` survive.
    """
    annotations_path, image_root = Path(annotations_path), Path(image_root)
    source = json.loads(annotations_path.read_text(encoding="utf-8-sig"))
    rows = source["images"]
    row_by_id = {str(r["id"]): r for r in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("Duplicate image/annotation-pass ids")
    annotations = defaultdict(list)
    annotation_ids = set()
    for annotation in source["annotations"]:
        image_id = str(annotation["image_id"])
        if image_id not in row_by_id:
            raise ValueError(f"Annotation references unknown image {image_id}")
        if str(annotation["id"]) in annotation_ids:
            raise ValueError("Duplicate annotation id")
        annotation_ids.add(str(annotation["id"]))
        annotations[image_id].append({"id": str(annotation["id"]), "segmentation": annotation["segmentation"]})

    physical_rows = defaultdict(list)
    for row in rows:
        file_name = Path(row["file_name"])
        if file_name.is_absolute() or ".." in file_name.parts or "\\" in row["file_name"]:
            raise ValueError("Unsafe image path in annotations")
        physical_rows[file_name.as_posix()].append(row)
    images = []
    thumbnail_features = {}
    for file_name, passes in sorted(physical_rows.items()):
        image_path = image_root / file_name
        with Image.open(image_path) as opened:
            width, height = opened.size
            # High-pass thumbnails aid review without treating similar disks as
            # proof of duplicate structures. No automatic approximate unions.
            thumb = np.asarray(opened.convert("L").resize((64, 64)), dtype=np.float32)
            smooth = np.asarray(opened.convert("L").resize((16, 16)).resize((64, 64)), dtype=np.float32)
            feature = (thumb - smooth)[8:56, 8:56].ravel()
            thumbnail_features[Path(file_name).stem] = feature / max(float(np.linalg.norm(feature)), 1e-8)
        if any((int(p["width"]), int(p["height"])) != (width, height) for p in passes):
            raise ValueError(f"Annotation/image dimension mismatch: {file_name}")
        physical_id = Path(file_name).stem
        match = re.fullmatch(r"(\d{14})([A-Za-z]+)", physical_id)
        if not match:
            raise ValueError(f"Cannot infer capture date/instrument: {file_name}")
        date = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").date().isoformat()
        for p in passes:
            if p.get("date_captured") and str(p["date_captured"])[:10] != date:
                raise ValueError(f"Filename/capture-date mismatch: {file_name}")
        item = {"physical_id": physical_id, "file_name": file_name, "sha256": sha256_file(image_path), "width": width, "height": height, "capture_date": date, "instrument": match.group(2), "passes": []}
        for row in sorted(passes, key=lambda r: str(r["id"])):
            image_id = str(row["id"])
            prefix, separator, suffix = image_id.rpartition("-")
            pass_id = prefix if separator and suffix == physical_id else image_id
            clean_annotations = sorted(annotations[image_id], key=lambda a: a["id"])
            rles = [_annotation_rle(a, height, width) for a in clean_annotations]
            areas = [int(coco_mask.area(r)) for r in rles]
            union_area = int(coco_mask.area(coco_mask.merge(rles))) if rles else 0
            stats = {"instance_count": len(areas), "empty_instances": sum(a == 0 for a in areas), "mask_area_sum": sum(areas), "foreground_area": union_area, "overlap_area": sum(areas) - union_area, "area_min": min(areas, default=0), "area_median": float(np.median(areas)) if areas else 0.0, "area_max": max(areas, default=0)}
            item["passes"].append({"annotation_pass_id": pass_id, "image_id": image_id, "annotations": clean_annotations, "stats": stats})
        images.append(item)
    if len({i["physical_id"] for i in images}) != len(images):
        raise ValueError("Physical stem collision across different file names")
    group_observations(images, extra_groups, date_gap_days=date_gap_days)
    candidates = []
    by_date = sorted(images, key=lambda i: (i["capture_date"], i["physical_id"]))
    for index, first in enumerate(by_date):
        first_date = datetime.fromisoformat(first["capture_date"])
        for second in by_date[index + 1:]:
            delta = (datetime.fromisoformat(second["capture_date"]) - first_date).days
            if delta > 1:
                break
            if delta == 1:
                candidates.append({"first": first["physical_id"], "second": second["physical_id"], "days": delta, "same_group": first["group_id"] == second["group_id"], "highpass_cosine": round(float(thumbnail_features[first["physical_id"]] @ thumbnail_features[second["physical_id"]]), 6)})
    candidates.sort(key=lambda c: (-c["highpass_cosine"], c["first"], c["second"]))
    summary = {"physical_images": len(images), "annotation_passes": len(rows), "instances": len(source["annotations"]), "groups": len({i["group_id"] for i in images}), "duplicate_hash_groups": sum(count > 1 for count in Counter(i["sha256"] for i in images).values()), "empty_instances": sum(p["stats"]["empty_instances"] for i in images for p in i["passes"]), "passes_with_overlapping_instances": sum(p["stats"]["overlap_area"] > 0 for i in images for p in i["passes"]), "adjacent_date_candidate_pairs": len(candidates), "year_counts": dict(sorted(Counter(i["capture_date"][:4] for i in images).items())), "instrument_counts": dict(sorted(Counter(i["instrument"] for i in images).items()))}
    manifest = {"version": 1, "annotations_path": str(annotations_path.resolve()), "image_root": str(image_root.resolve()), "annotation_sha256": sha256_file(annotations_path), "images": images, "summary": summary, "grouping": {"date_gap_days": date_gap_days, "same_date_across_instruments": True, "exact_hash": True}, "near_duplicate_candidates": candidates, "near_duplicate_review": "Adjacent dates grouped conservatively by default. Approximate similarity aids visual audit and is not a duplicate label.", "extra_groups": extra_groups or []}
    manifest["manifest_hash"] = manifest_digest(manifest)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if manifest.get("version") != 1:
        raise ValueError("Unsupported manifest version")
    if manifest.get("manifest_hash") != manifest_digest(manifest):
        raise ValueError("Manifest hash mismatch")
    return manifest
