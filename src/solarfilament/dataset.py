"""Index-seeded sampling makes data progress recoverable without worker RNG state."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .data import rasterize_pass
from .inference import preprocess


class CropDataset(Dataset):
    def __init__(self, manifest, images, config, length):
        self.root = Path(manifest["image_root"])
        self.images = images
        self.config = config
        self.length = length
        if not images:
            raise ValueError("Training split is empty")

    def __len__(self):
        return self.length

    @lru_cache(maxsize=16)
    def image(self, index):
        with Image.open(self.root / self.images[index]["file_name"]) as source:
            return np.asarray(source.convert("L"))

    @lru_cache(maxsize=32)
    def target(self, index, pass_index):
        item = self.images[index]
        target = np.zeros((item["height"], item["width"]), np.float32)
        passes = item["passes"] if pass_index == -1 else [item["passes"][pass_index]]
        for record in passes:
            foreground = np.zeros_like(target, dtype=bool)
            for mask in rasterize_pass(record, item["height"], item["width"]):
                foreground |= mask
            target += foreground
        return target.astype(np.uint8) if pass_index != -1 else (target / len(passes)).astype(np.float16)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], int(index)]))
        # Uniform physical image, then complete pass: annotation-rich images receive no extra weight.
        image_index = int(rng.integers(len(self.images)))
        item = self.images[image_index]
        pass_index = -1 if self.config.get("annotation_mode") == "soft_consensus" else int(rng.integers(len(item["passes"])))
        image, target = self.image(image_index), self.target(image_index, pass_index)
        size = int(self.config["crop_size"])
        height, width = image.shape
        if height < size or width < size:
            padding = ((0, max(0, size-height)), (0, max(0, size-width)))
            image, target = np.pad(image, padding, mode="edge"), np.pad(target, padding)
            height, width = image.shape
        if rng.random() < self.config.get("foreground_probability", 0.5) and np.any(target > 0):
            foreground = np.flatnonzero(target > 0)
            y, x = np.unravel_index(rng.choice(foreground), target.shape)
            y, x = y - int(rng.integers(size)), x - int(rng.integers(size))
        else:
            # Uniform disk-centered crop, with bounded rejection for corrupted/empty disks.
            y, x = int(rng.integers(height)), int(rng.integers(width))
            for _ in range(20):
                if image[y, x] > 12:
                    break
                y, x = int(rng.integers(height)), int(rng.integers(width))
            y, x = y-size//2, x-size//2
        y, x = int(np.clip(y, 0, height-size)), int(np.clip(x, 0, width-size))
        mode=self.config.get("normalization","raw")
        # Raw normalization is pointwise; crop first to avoid processing 12M values per sample.
        image = (preprocess(image[y:y+size,x:x+size],mode) if mode=="raw" else
                 preprocess(image,mode)[:,y:y+size,x:x+size])
        target = target[y:y+size, x:x+size]
        turns = int(rng.integers(4))
        image, target = np.rot90(image, turns, axes=(-2,-1)), np.rot90(target, turns)
        for axis in (-1,-2):
            if rng.random() < 0.5:
                image, target = np.flip(image, axis), np.flip(target, axis)
        image = image * rng.uniform(0.95,1.05) + rng.uniform(-0.03,0.03)
        if rng.random() < 0.15:
            image += rng.normal(0,0.015,image.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(target[None], dtype=np.float32))
