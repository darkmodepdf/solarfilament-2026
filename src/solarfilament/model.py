"""Pinned SMP U-Net; inference never downloads initialization weights."""
import torch
import segmentation_models_pytorch as smp


def build_model(config, pretrained=True):
    return smp.Unet(
        encoder_name=config.get("encoder", "resnet34"),
        encoder_weights=config.get("encoder_weights", "imagenet") if pretrained else None,
        in_channels=3,
        classes=1,
    )


def segmentation_loss(logits, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target.float())
    probability = logits.float().sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probability * target).sum(dims) + 1) / (
        probability.sum(dims) + target.sum(dims) + 1
    )
    return 0.5 * bce + 0.5 * (1 - dice.mean())
