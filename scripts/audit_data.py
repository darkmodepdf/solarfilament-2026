"""Build the official-data manifest/folds and reproducible visual audit sheets.

Run from the repository root after installing the package, for example:
python scripts/audit_data.py --data-root data/raw/MAGFiLO_1.0_Kaggle_2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from solarfilament.data import build_manifest, load_manifest, rasterize_pass, sha256_file
from solarfilament.splits import make_splits, validate_splits


def render_audit(manifest: dict, output: Path, count: int = 20):
    images = manifest["images"]
    selected = []
    # Include crowded, sparse, tiny and extensive annotations before a fixed
    # random sample. Each sheet pairs the unaltered image and COCO mask overlay.
    orderings = [sorted(images, key=lambda i: max(p["stats"]["instance_count"] for p in i["passes"]), reverse=True), sorted(images, key=lambda i: min(p["stats"]["area_min"] for p in i["passes"])), sorted(images, key=lambda i: max(p["stats"]["foreground_area"] for p in i["passes"]), reverse=True)]
    for ordering in orderings:
        for item in ordering[:3]:
            if item not in selected:
                selected.append(item)
    remaining = [i for i in images if i not in selected]
    random.Random(2026).shuffle(remaining)
    selected.extend(remaining[:max(0, count - len(selected))])
    selected = selected[:count]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for sheet_index in range((len(selected) + 4) // 5):
        sheet_items = selected[sheet_index * 5:(sheet_index + 1) * 5]
        canvas = Image.new("RGB", (768, 416 * len(sheet_items)), "white")
        for row_index, item in enumerate(sheet_items):
            pass_record = max(item["passes"], key=lambda p: p["stats"]["instance_count"])
            raw = Image.open(Path(manifest["image_root"]) / item["file_name"]).convert("RGB")
            overlay = np.asarray(raw).copy()
            for index, mask in enumerate(rasterize_pass(pass_record, item["height"], item["width"])):
                color = np.asarray([(index * 71 + 30) % 255, (index * 43 + 180) % 255, (index * 101 + 80) % 255])
                overlay[mask] = (overlay[mask] * .35 + color * .65).astype(np.uint8)
            offset = row_index * 416
            canvas.paste(raw.resize((384, 384)), (0, offset))
            canvas.paste(Image.fromarray(overlay).resize((384, 384)), (384, offset))
            caption = f"{item['physical_id']} | pass {pass_record['annotation_pass_id']} | n={len(pass_record['annotations'])}"
            ImageDraw.Draw(canvas).text((5, offset + 388), caption, fill="black")
            records.append({"physical_id": item["physical_id"], "annotation_pass_id": pass_record["annotation_pass_id"], "stats": pass_record["stats"]})
        canvas.save(output / f"annotations-{sheet_index + 1:02d}.jpg", quality=92)
    candidates = manifest.get("near_duplicate_candidates", [])[:10]
    lookup = {i["physical_id"]: i for i in images}
    for start in range(0, len(candidates), 5):
        selected_pairs = candidates[start:start + 5]
        canvas = Image.new("RGB", (768, 416 * len(selected_pairs)), "white")
        for row_index, pair in enumerate(selected_pairs):
            for column, key in enumerate(("first", "second")):
                item = lookup[pair[key]]
                raw = Image.open(Path(manifest["image_root"]) / item["file_name"]).convert("RGB")
                canvas.paste(raw.resize((384, 384)), (column * 384, row_index * 416))
                ImageDraw.Draw(canvas).text((column * 384 + 5, row_index * 416 + 388), item["physical_id"], fill="black")
        canvas.save(output / f"adjacent-dates-{start // 5 + 1:02d}.jpg", quality=92)
    (output / "selected-images.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reuse-manifest", action="store_true")
    parser.add_argument("--extra-groups", type=Path, help="JSON list of reviewed near-duplicate physical-id groups")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if args.reuse_manifest:
        manifest = load_manifest(manifest_path)
    else:
        overrides = json.loads(args.extra_groups.read_text()) if args.extra_groups else None
        manifest = build_manifest(args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json", args.data_root / "train/train_images", manifest_path, extra_groups=overrides)
    splits = make_splits(manifest, n_folds=args.folds, seed=args.seed)
    validate_splits(manifest, splits)
    (args.output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    test_images = []
    for path in sorted((args.data_root / "test/test_images").glob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            with Image.open(path) as opened:
                width, height = opened.size
            test_images.append({"physical_id": path.stem, "file_name": path.name, "sha256": sha256_file(path), "width": width, "height": height})
    (args.output / "test-images.json").write_text(json.dumps(test_images, indent=2), encoding="utf-8")
    render_audit(manifest, args.output / "audit")
    print(json.dumps({"train": manifest["summary"], "test_images": len(test_images), "manifest_hash": manifest["manifest_hash"], "split_hash": splits["split_hash"], "visual_audit": str(args.output / "audit")}, indent=2))


if __name__ == "__main__":
    main()
