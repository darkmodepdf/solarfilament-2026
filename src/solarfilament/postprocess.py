"""Deterministic disjoint instances at native resolution."""
import numpy as np
from scipy import ndimage
from collections.abc import Sequence


class ComponentMasks(Sequence):
    """Represent disjoint masks using one label image, regardless of object count."""
    def __init__(self, labels, ids):
        self.labels, self.ids = labels, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,index):
        if isinstance(index,slice):
            return ComponentMasks(self.labels,self.ids[index])
        return self.labels == self.ids[index]


def instances_from_probability(probability, threshold=0.45, min_area=32, low_threshold=None):
    probability = np.asarray(probability)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Expected a finite 2D probability map")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probabilities must lie in [0,1]")
    if not 0 <= threshold <= 1 or min_area < 0:
        raise ValueError("Invalid threshold/minimum area")
    structure = np.ones((3, 3), dtype=bool)
    foreground = probability >= threshold
    if low_threshold is not None:
        if not 0 <= low_threshold <= threshold:
            raise ValueError("Low threshold must not exceed high threshold")
        foreground = ndimage.binary_propagation(
            foreground, structure=structure, mask=probability >= low_threshold
        )
    labels, count = ndimage.label(foreground, structure)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    ids=[label for label in range(1,count+1) if areas[label]>=max(1,min_area)]
    return ComponentMasks(labels,ids) if ids else []
