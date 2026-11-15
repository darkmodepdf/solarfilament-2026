# Solar filament instance segmentation

A reproducible entry for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026): a ResNet34 U-Net predicts native-scale foreground probabilities, overlapping tiles are blended into a complete solar image, and connected components become disjoint filament instances.

**Status, 15 November 2026:** the data audit, grouped splits, official-metric parity checks, training/recovery implementation and export validation are implemented. GPU execution is being evaluated separately. No score, accepted prediction, final submission selection or completed organizer form is claimed by this README. The technical report is a draft until measured results and final assets are inserted.

The main design choices are deliberate. Thin structures retain their original pixel scale through 768-pixel crops. A physical observation is sampled uniformly, followed by one complete annotation pass, so an image is not overweighted merely because more people labeled it. Whole-image Panoptic Quality (PQ), rather than crop Dice, chooses checkpoints. See [the method explanation](#understanding-and-defending-the-method) and [data audit](docs/data-audit.md).

## Data and validation contract

Only the competition images and segmentation polygons are used for competition supervision. The downloaded training JSON includes `spine` and other auxiliary fields; the manifest explicitly retains only instance `id` and `segmentation`. Categories are treated as one foreground class. No spine, chirality, bounding-box supervision, outside MAGFiLO annotation or test annotation enters training.

The official download contains 707 physical training images, 1,154 complete annotation passes, 8,199 instances and 180 test images; all are 2048 × 2048. Five original annotation passes contain overlapping instance masks; those labels are retained unchanged. Date/hash grouping conservatively joins same-day and adjacent-day observations across instruments. The resulting 502 groups form folds of 236, 236 and 235 physical images; fold 0 is the development holdout. This is a three-way split, **not a claim that three fold models were trained**.

Frozen manifest SHA-256: `fa74669c215519f416419935b3a0432ce0685eb1bc92c9f0caf9f1f540e81d0e`.

Frozen split SHA-256: `ba393d2c0b05df0627d17cc16eea95de45336449859c9ac1c09e51435135bcc5`.

The [official-metric notes](docs/official-metric.md) explain the organizer's exact pooled, per-annotation-pass calculation, including its all-qualifying-pairs matching behavior. Our exported instances are disjoint; duplicate-mask behavior is tested for parity rather than used to inflate scores. Synthetic and deliberately perturbed training-mask tests verify scorer behavior and mask encoding; they do not measure a trained model.

## Run locally or on a GPU host

Use Python 3.11 or newer. On a fresh environment, install a matching PyTorch/torchvision build for the host, then the pinned requirements. The local reference pair is torch 2.7.1 and torchvision 0.22.1; the Kaggle notebook preserves its preinstalled working CUDA pair and saves the actual environment freeze instead of replacing it indiscriminately.

```text
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
kaggle competitions download -c filament-segmentation-2026 -p data/raw
```

Extract the official archive to `data/raw`, retaining its `MAGFiLO_1.0_Kaggle_2026` directory. Do not commit competition images or annotations to the public repository.

```text
python -m solarfilament.cli prepare --data-root data/raw/MAGFiLO_1.0_Kaggle_2026 --output-dir runs/data
python -m pytest -q
python scripts/preflight.py --output-dir runs/preflight --require-gpu
python -m solarfilament.cli train --manifest runs/data/manifest.json --splits runs/data/splits.json --config configs/baseline.yaml --output-dir runs/baseline --max-steps 1000
```

For two NVIDIA GPUs, explicitly launch one process per GPU:

```text
torchrun --standalone --nproc_per_node=2 -m solarfilament.cli train --manifest runs/data/manifest.json --splits runs/data/splits.json --config configs/baseline.yaml --output-dir runs/baseline --max-steps 1000
```

The effective batch remains eight: on two GPUs, batch two per device uses two accumulation microsteps. Recovery uses `--resume runs/baseline/last.pt`; retain the original data/split identity, GPU count and training recipe. The step cap limits this run while the configured 4,000-step cosine schedule remains unchanged for a faithful continuation. Inspect restored step and learning rate in the logs.

Prepare audit images separately with `python scripts/audit_data.py --data-root data/raw/MAGFiLO_1.0_Kaggle_2026`. The script writes hash manifests, fold summaries and reproducible raw/mask contact sheets.

## Kaggle execution

The [packager](scripts/package_kaggle.py) generates a private training notebook with an allowlisted, checksummed source snapshot and the official competition attached. Rebuild it after source changes. It creates and validates a CSV; it does not upload a competition submission or complete the organizer form. The generated notebook may embed the locally obtained reference scorer for tests, so it is excluded from the public repository. The public [inference notebook](notebooks/inference.ipynb) contains no vendor source or data.

```text
python scripts/package_kaggle.py --output build/kaggle --max-steps 1000
kaggle kernels push -p build/kaggle --timeout 36000 --accelerator NvidiaTeslaT4
kaggle kernels status vishwasbollam/solarfilament-annotation-aware-baseline
kaggle kernels logs vishwasbollam/solarfilament-annotation-aware-baseline
kaggle kernels output vishwasbollam/solarfilament-annotation-aware-baseline -p runs/kaggle-baseline
```

The private notebook checks actual GPU allocation, RAM/VRAM, disk and packages; tests train/save/restart/resume before the long run; and counts setup time within a ten-hour budget with a one-hour final reserve. Session success and durable outputs must be confirmed from the run, not inferred from notebook creation. [Execution and continuation details](docs/kaggle-execution.md) cover the quota and deadline checks.

## Freeze settings and reproduce predictions

After a completed run has saved `best.pt` and its matching full-image validation probabilities:

```text
python -m solarfilament.cli tune --manifest runs/data/manifest.json --splits runs/data/splits.json --cache-dir runs/baseline/validation --output runs/selected.json --fold 0
python -m solarfilament.cli predict --checkpoint runs/baseline/best.pt --image-dir data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images --output-dir runs/final --config runs/selected.json
python -m solarfilament.cli validate --submission runs/final/submission.csv --image-dir data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images --processed runs/final/processed_images.json
```

Prediction loads the supplied checkpoint with encoder initialization disabled, so inference does not fetch pretrained weights. The [inference notebook](notebooks/inference.ipynb) loads the frozen released checkpoint and settings without retraining, then compares the resulting instances with the submitted artifact. Released weights, its executed reproduction evidence, public repository commit and acceptance receipts remain pending until generated and checked.

Every output row contains `filament_id,segmentation_rle`, with an exact image stem and unique object suffix. The validator checks COCO compressed RLE, dimensions, nonempty masks, unique IDs and disjoint pixels. A sidecar proves all expected images were processed, including zero-detection images, without inventing empty rows. `runs/final/inference.json` records configuration, checkpoint/CSV hashes, per-image time and peak GPU memory. File validity is separate from Kaggle acceptance.

## Understanding and defending the method

**Annotation disagreement.** Different annotators sometimes divide one visible structure differently. Concatenating all their instances would create artificial duplicates. Sampling one complete pass preserves an annotator's internally consistent interpretation; evaluation compares the same prediction against untouched passes. It does not establish which annotator is correct.

**Native-scale detail.** Downsampling an entire 2048-pixel disk can erase a narrow filament. Crops keep the pixel resolution while limiting VRAM. Their cost is reduced global context. At inference, a nonzero Hann window blends overlapping probabilities before any object is extracted, preventing tile boundaries from becoming object boundaries by construction.

**Pixel loss versus instance score.** Equal-weight binary cross-entropy and soft Dice train the foreground map. PQ also penalizes missed/extra instances and bad overlaps, so better semantic Dice alone does not justify a checkpoint. The pipeline selects using the actual organizer computation on complete held-out images.

**Instance construction.** Eight-connected components are a transparent baseline. A high/low threshold candidate retains low-confidence pixels connected to a confident seed; it can recover faint extensions but can also merge neighbors. Threshold, minimum area and optional reconstruction are chosen from cached validation probabilities. Defaults are starting settings, not proven optimal values. No global morphological closing or fixed disk-edge removal is applied.

**Remaining limits.** A semantic model has no explicit object identity and may merge adjacent filaments or fragment a long one. One development holdout supports tuning but does not provide an unbiased estimate after repeated selection. Three folds are available for confirmation if time permits. A winning score, novel architecture, completed ablation, confidence interval or GPU throughput must not be claimed without its saved evidence.

## Release evidence and attribution

The [eligibility record](docs/eligibility.md) separates the prediction and organizer deadlines. The [report draft](docs/report-draft.md) and [LaTeX source](report/report.tex) use the retrieved official template; author details, results and final examples remain visibly pending until confirmed. The template limits the main report to four pages, excluding Acknowledgment and References.

See [third-party sources and weights](docs/sources-and-licenses.md). SMP code is MIT and torchvision code is BSD-3-Clause, but these code licenses do not automatically license pretrained weights. The public ImageNet encoder's pinned model card declares `license: other`. The competition's Winner License clause explicitly exempts incompatible pretrained models from its open-source grant; original pretrained assets retain their terms and are not relicensed by this project. The organizer's scorer notebook is retained locally for parity; its redistribution license has not been established. Retrieve it as documented if omitted from a public release.

AI assistance was used in implementation, testing and documentation. The participant remains responsible for reviewing the code, explaining these choices, checking the numerical results and reproducing the final entry. Repository publication, released weights, four-page report, explicit final prediction selection and organizer-form receipt are all required for the final entry; training completion alone is insufficient.
