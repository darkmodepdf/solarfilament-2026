# Sources, pretrained initialization and release notes

Verified 15 November 2026 against primary source code, package documentation and the downloaded competition assets. This file records provenance; it does not assert permission where a source does not provide it.

| Component | Version/source | Confirmed terms or status |
|---|---|---|
| Segmentation Models PyTorch | [v0.5.0 source/license](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/LICENSE) | MIT code license; keep the notice if redistributing covered code |
| torchvision ResNet implementation | [v0.22.1 license](https://github.com/pytorch/vision/blob/v0.22.1/LICENSE) | BSD-3-Clause code license |
| ImageNet ResNet34 initializer | [`smp-hub/resnet34.imagenet`](https://huggingface.co/smp-hub/resnet34.imagenet/tree/7a57b34f723329ff020b3f8bc41771163c519d0c) | Public model; pinned card declares `license: other`, without a more specific grant in that card |
| Original initializer URL | [ResNet34 original weights](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | Identified by the pinned SMP model card and legacy loader settings |
| Competition images/annotations | [official competition](https://www.kaggle.com/competitions/filament-segmentation-2026/data) | Use for this competition under its rules; do not publish the data in the repository |
| Organizer self-evaluation notebook | [official notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) | Retrieved and hashed for exact parity checks; redistribution license not identified |
| Report template | [official Overleaf share](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe) | Convenience template for this competition; its notice does not establish general redistribution or reuse rights |

`segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)` is the actual builder. SMP 0.5.0's [ResNet settings](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/segmentation_models_pytorch/encoders/resnet.py) pin the Hugging Face repository to revision `7a57b34f723329ff020b3f8bc41771163c519d0c`. Its [loader](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/segmentation_models_pytorch/encoders/__init__.py) first requests `config.json` and `model.safetensors` from that revision and falls back to the original PyTorch URL if that fetch fails. Record the route actually used and the downloaded file SHA-256 from the runtime before freezing training provenance. Final inference sets `encoder_weights=None` and loads the released trained checkpoint.

The [torchvision pretrained-model notice](https://github.com/pytorch/vision/blob/v0.22.1/README.md#pre-trained-model-license) distinguishes model terms from source-code terms. Consequently, neither the SMP MIT license nor torchvision's BSD license is described as the weight license here. The [competition-specific rules, Winner License section](https://www.kaggle.com/competitions/filament-segmentation-2026/rules) explicitly exempt input data and pretrained models with incompatible licenses from the required open-source grant. Their inclusion therefore does not itself establish an eligibility blocker. Original pretrained assets retain their existing terms; this project's license must not purport to relicense those assets. Preserve provenance and any applicable upstream distribution conditions when packaging trained weights. The source snapshot of this exception is `docs/evidence/competition-pages-2026-11-15.json`.

Primary method references:

- Ronneberger, Fischer and Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), 2015. We use an existing implementation; no new U-Net architecture is claimed.
- He, Zhang, Ren and Sun, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015 preprint / CVPR 2016. The encoder is the 34-layer residual architecture.
- [SMP 0.5.0 U-Net documentation](https://smp.readthedocs.io/en/v0.5.0/models.html#unet) specifies the library constructor and defaults used by our code.
- Ahmadzadeh et al., [MAGFiLO dataset paper](https://doi.org/10.1038/s41597-024-03876-y), Scientific Data, 2024. Only the competition-provided subset of segmentation supervision is consumed.

License/model-card evidence is retained under `docs/evidence/`. Competition assets and the untouched official template archive are local provenance materials, not an assertion that they may all be bundled into a public code release. Public release packaging must distinguish original project code from these sources and preserve the notices of any redistributed third-party code.
