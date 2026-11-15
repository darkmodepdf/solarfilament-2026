"""Full-image probability blending before instance construction."""
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .common import sha256_file, write_json
from .model import build_model
from .postprocess import instances_from_probability


def preprocess(image, mode="raw"):
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("Expected grayscale H-alpha image")
    gray = gray / 255.0
    if mode == "disk_percentile":
        disk = gray > 0.05
        if disk.any():
            low, high = np.percentile(gray[disk], (1, 99))
            gray = np.clip((gray - low) / max(float(high - low), 0.01), 0, 1)
    elif mode != "raw":
        raise ValueError(f"Unknown normalization: {mode}")
    rgb = np.repeat(gray[None], 3, axis=0)
    return (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )[:, None, None]


def tile_starts(length, size, overlap):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Invalid tile size/overlap")
    if length <= size:
        return [0]
    stride = max(1, int(round(size * (1 - overlap))))
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


@torch.inference_mode()
def predict_probability(model, image, config, device="cpu"):
    model.eval()
    device = torch.device(device)
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape
    size = int(config.get("crop_size", 768))
    if size % 32:
        raise ValueError("U-Net crop size must be divisible by 32")
    x = preprocess(image, config.get("normalization", "raw"))
    x = np.pad(x, ((0, 0), (0, max(0, size-height)), (0, max(0, size-width))), mode="edge")
    total = np.zeros(x.shape[1:], np.float32)
    weights = np.zeros_like(total)
    window = np.maximum(np.outer(np.hanning(size), np.hanning(size)), 1e-3).astype(np.float32)
    overlap = config.get("tile_overlap", 0.25)
    positions = [(y, z) for y in tile_starts(x.shape[1], size, overlap)
                 for z in tile_starts(x.shape[2], size, overlap)]
    batch_size = int(config.get("inference_batch_size", config.get("batch_size", 2)))
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start+batch_size]
        batch = torch.from_numpy(np.stack([x[:, y:y+size, z:z+size] for y,z in batch_positions])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda" and config.get("amp", True)):
            probability = model(batch).float().sigmoid()
            if config.get("tta", False):
                probability += model(batch.flip(-1)).float().sigmoid().flip(-1)
                probability += model(batch.flip(-2)).float().sigmoid().flip(-2)
                probability += model(batch.flip((-1, -2))).float().sigmoid().flip((-1, -2))
                probability /= 4
        for (y, z), pred in zip(batch_positions, probability[:, 0].cpu().numpy()):
            total[y:y+size, z:z+size] += pred * window
            weights[y:y+size, z:z+size] += window
    if not np.all(weights > 0):
        raise RuntimeError("Uncovered inference pixels")
    return (total / weights)[:height, :width]


def load_checkpoint_model(checkpoint_path, device="cpu"):
    # Only load checkpoints produced by this trusted local pipeline.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def run_prediction(checkpoint_path, image_dir, output_dir, overrides=None):
    from .submission import write_submission_stream, validate_submission
    overall_start=time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    config = {**checkpoint["config"], **(overrides or {})}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    if not files:
        raise ValueError("No test images found")
    processed, timings = [], {}
    def predictions():
        for path in files:
            start = time.monotonic()
            with Image.open(path) as source:
                gray = np.asarray(source.convert("L"))
            probability = predict_probability(model, gray, config, device)
            masks = instances_from_probability(
                probability, config["threshold"], config["min_area"], config.get("low_threshold")
            )
            processed.append(path.stem)
            print(f"predict {len(processed)}/{len(files)} {path.stem}: {len(masks)} instances", flush=True)
            yield path.stem, masks
            timings[path.stem] = time.monotonic() - start
    path = output_dir / "submission.csv"
    write_submission_stream(predictions(), path, expected_images=[p.stem for p in files])
    write_json(output_dir / "processed_images.json", processed)
    result = validate_submission(path, expected_images=[p.stem for p in files], processed_images=processed)
    write_json(output_dir / "inference.json", {
        "config": config, "checkpoint_sha256": sha256_file(checkpoint_path),
        "submission_sha256": sha256_file(path), "seconds_per_image": timings,
        "end_to_end_seconds": time.monotonic()-overall_start,
        "device": device, "validation": result,
        "max_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
    })
    return path
