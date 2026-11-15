# Annotation-aware native-scale U-Net for solar filament instance segmentation

**Working draft — not a final submission.** Author name **Vishwas Bollam** and Kaggle username `vishwasbollam` were confirmed from the authenticated Kaggle profile. Affiliation and contact email remain unconfirmed and are omitted from the typeset draft. Reported model quality, compute measurements, final weights, submission ID and acceptance receipts are pending. No blank below represents a measured zero.

The actual organizer template was retrieved from its [Overleaf share](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe). `report/report.tex` preserves its fixed subtitle, introductory text and acknowledgments. The limit is four pages **excluding Acknowledgment and References**. Its main source uses `acmart[sigconf]`. A three-page draft PDF was compiled locally with Tectonic 0.17.0 and all pages rendered and inspected; long-line overflows were corrected. This is a layout check, not a final report or final page-limit certification once results and figures are added. Compiler, bundle, source/PDF hashes and nonfatal warning details are in [the build record](../report/build-record.json). Template archive SHA-256: `7c725e69d30d4cbe9e194ef34f1da214ce962817bd1b728b2efec77708f1bff3`.

## Abstract

We implement a reproducible solar filament instance-segmentation pipeline using an existing ResNet34 U-Net, native-resolution crops and full-image probability blending. Training samples physical observations uniformly and then selects one complete annotation pass. Same-day and adjacent-day observations remain in the same validation group. We reproduce the organizer's pooled per-annotation-pass PQ calculation, use it for checkpoint selection, and tune disjoint instance construction on cached held-out probabilities. **The measured validation result, selected post-processing settings, runtime and accepted submission score must be inserted after verified execution.**

## Data and method

The official competition download contains 707 physical training observations represented by 1,154 annotation-pass image records and 8,199 instance masks, plus 180 test images. All images are 2048 × 2048. Polygons are rasterized by pycocotools 2.0.10. The manifest retains only each instance identifier and segmentation coordinates; supplied spines and other auxiliary annotations are excluded. All categories become one semantic foreground class. The five passes containing overlapping ground-truth instances remain unchanged for evaluation.

We group images by identical image hash and by same/adjacent calendar dates across instruments, joining temporal chains transitively. Inspection of adjacent-date examples showed persistent evolving structures. The resulting 502 groups have at most five images each. A deterministic allocator balances year, instrument and filament-density strata into folds of 236, 236 and 235 images. Fold 0 is the development holdout; the current baseline trains on the other 471 physical images. Three available folds do not imply three completed training runs. Manifest and split hashes identify the exact assignment.

Each training sample first chooses a physical image uniformly, then one complete annotation pass uniformly. Its instance masks are combined into a binary foreground target for semantic training only. This avoids weighting observations by annotator count or pretending that multiple annotators' instances coexist. With probability 0.5, a crop contains a randomly selected foreground pixel; otherwise the center is sampled from the bright solar disk using an intensity-based rejection rule. Coordinates are clamped to the image boundary. The latter strategy is disk-centered sampling, not uniform sampling over all possible crop origins.

Grayscale intensities are divided by 255, repeated across three channels and standardized with ImageNet channel means `(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`. Native 768 × 768 crops preserve narrow structures. Augmentation consists of quarter-turn rotations, horizontal/vertical flips, a multiplicative factor in `[0.95,1.05]`, an additive shift in `[-0.03,0.03]`, and Gaussian noise of standard deviation 0.015 with probability 0.15. Intensity adjustments are applied after channel standardization in the current implementation. No blur is implemented in the baseline.

The network is SMP 0.5.0's U-Net with a ResNet34 encoder, five encoder stages, skip connections and one output logit channel. Its default decoder channels are 256, 128, 64, 32 and 16, with batch normalization and nearest-neighbor interpolation. The encoder starts from public ImageNet weights; the decoder is initialized by the library. This is an existing architecture, not an architectural novelty. The pinned initializer repository/revision and distinct code/weight licensing status are recorded in [sources-and-licenses.md](sources-and-licenses.md).

The loss is half pixelwise binary cross-entropy and half soft Dice loss, with smoothing constant one and Dice calculated per crop before averaging. AdamW uses an initial learning rate of 0.0003 and weight decay 0.0001. The configured cosine schedule spans 4,000 optimizer steps. The initial run cap is 1,000 steps, not a completed-run claim. Effective batch is eight; on two GPUs with batch two per device this requires two accumulation microsteps. FP16 uses loss scaling, gradient norm clipping at one, and one explicit DDP process per GPU. Sample indices are disjoint across ranks and map deterministically to image, pass and augmentation, supporting exact data-progress recovery.

Inference processes overlapping native-scale tiles, targeting 25% overlap. Final border tiles are anchored to the image boundary and may overlap more. Tile probabilities are blended by a two-dimensional Hann window clamped to a minimum of 0.001; probabilities are stitched before instance extraction. Eight-connected components define disjoint instances. The initial settings are threshold 0.45 and minimum area 32 native pixels. Optional high/low threshold reconstruction retains low-threshold components connected to a high-threshold seed. No global morphological closing or fixed edge-ring suppression is used. Test-time augmentation is disabled in the baseline.

## Evaluation and results to populate

The organizer notebook is byte-pinned and its original functions are executed in parity tests. It counts all mask pairs with float32 IoU strictly greater than 0.5; no Hungarian or other one-to-one assignment is performed. TP, FP, FN and matched IoU sums are pooled over annotation passes. Predictions remain disjoint to provide an honest instance construction. Whole-image official PQ chooses checkpoints; semantic training loss is not used as a substitute for the objective.

After checkpoint selection, the cached foreground probabilities support 32 post-processing candidates: thresholds `{0.35,0.45,0.55,0.65}`, minimum areas `{0,32,128,256}`, and either no reconstruction or a low threshold 0.15 below the high threshold. The best development PQ selects the settings. Any reported improvement after this search is a development-set result, not an independent confirmation result. Confirmation on an untouched grouped split and group-level uncertainty estimates require additional actual runs.

| Required evidence | Current state |
|---|---|
| Selected checkpoint step/hash | Pending completed GPU run |
| Baseline development PQ / SQ / RQ / TP / FP / FN | Pending measured evaluation |
| Post-processing selected values and PQ difference | Pending cached validation sweep |
| Matched IoU/Dice distribution; split/merge examples | Pending saved validation predictions |
| Confirmation fold or group-bootstrap interval | Not executed; do not imply otherwise |
| Training/evaluation/inference time and peak memory | Pending named-hardware measurements |
| 180-image export validation and clean reproduction | Pending model inference |
| Kaggle accepted score and submission ID | Pending receipt |

The final report should show a representative ordinary case and at least one failure using input, untouched GT and actual prediction overlays. Neither ground-truth-only audit sheets nor training loss curves should be labeled as model segmentation results. The current expected limitations are merged neighboring structures, fragmented long filaments, faint-object recall and annotation ambiguity; their observed frequency must be measured before ranking them.

## Reproducibility

Every run records configuration, actual package versions, source identity, data/split hashes, initialization provenance and device information. Full recovery checkpoints include model, optimizer, schedule, scaler, per-rank random state, optimizer step and sample position. The clock starts before setup; training reserves time for evaluation, saving and inference. Output size is checked against Kaggle storage capacity. Resume is checked before long training, and continuation requires matching data, split, world size and recipe.

Prediction requires only supplied test H-alpha images, the frozen checkpoint and selected configuration. Encoder initialization is disabled when loading the final checkpoint. The submission contains exact image stems and unique suffixes with compressed COCO RLE counts. Validators reject empty or overlapping masks and check mask dimensions and round-trip identity. A separate processed-image manifest accounts for every image, including zero detections.

Before finalizing: supply the immutable repository/weight links and hashes, confirm a fresh inference run reproduces decoded masks, insert actual hardware/metrics, confirm author/contact details, remove all pending markers, compile the official-template report and check the page limit, preserve the accepted prediction/final-selection receipts, and complete the organizer form. AI assistance in code and documentation should be disclosed accurately while the participant explains and takes responsibility for the method.
