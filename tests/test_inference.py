import numpy as np
import pytest
import torch

from solarfilament.inference import predict_probability, tile_starts
from solarfilament.postprocess import instances_from_probability


class Constant(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x[:, :1]) + 0.7


@pytest.mark.parametrize("shape", [(37, 83), (32, 32), (11, 19), (96, 101)])
def test_tiled_blending_has_exact_shape_no_seams(shape):
    result = predict_probability(Constant(), np.zeros(shape, np.uint8), {"crop_size":32,"batch_size":3})
    assert result.shape == shape
    assert np.allclose(result, torch.tensor(0.7).sigmoid().item(), atol=1e-6)


def test_hysteresis_requires_connected_high_seed():
    prob = np.zeros((20, 20), np.float32)
    prob[2:4, 2:8] = 0.4
    prob[2, 2] = 0.9
    prob[15:17, 15:17] = 0.4
    masks = instances_from_probability(prob, 0.7, 0, 0.3)
    assert len(masks) == 1 and masks[0].sum() == 12


def test_threshold_min_area_and_invalid_maps():
    p = np.zeros((8,8),np.float32)
    p[1,1]=1
    assert instances_from_probability(p, min_area=2) == []
    with pytest.raises(ValueError):
        instances_from_probability(p, low_threshold=0.9)
    with pytest.raises(ValueError):
        tile_starts(32, 32, 1)
