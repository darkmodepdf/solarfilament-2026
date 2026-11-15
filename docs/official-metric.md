# Organizer metric and submission contract

Inspected 15 November 2026. The organizer's public [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) was retrieved using the installed Kaggle CLI:

```text
kaggle kernels pull azimahmadzadeh/self-evaluation-notebook -p vendor/official --metadata
```

The downloaded notebook SHA-256 is `ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801`; Kaggle notebook ID is `130222308`. The metadata does not expose a notebook version number; the byte hash is the source pin. The original notebook and metadata are preserved locally under `vendor/official/`. It is an organizer reference, not our original code. Its redistribution license was not identified in the download; confirm that license before including the notebook in a public source release. The independent implementation in `src/solarfilament/metrics.py` follows the inspected computation.

## Exact inspected behavior

- Decode each compressed COCO mask as a separate 2048 × 2048 layer.
- Split a ground-truth filament ID at the first underscore to get an annotator-image ID, then split that ID at the first hyphen to get the physical image. Predictions use the image stem before the first underscore. Evaluate each complete annotation pass separately against the same image predictions.
- Calculate pairwise IoU in float32; qualifying matches have IoU **strictly greater than 0.5**.
- Count **every qualifying pair** as a true positive. There is **no Hungarian assignment, greedy assignment, deduplication, or one-to-one constraint**. Independent overlapping masks remain independent. Our metric reproduces this behavior; our submission writer nevertheless requires disjoint predictions for a defensible instance construction.
- A prediction column without a qualifying pair is one false positive. A GT row without a qualifying pair is one false negative.
- Pool TP, FP, FN and matched IoU sums across all annotation passes, then divide `sum(IoU)/(TP + 0.5*FP + 0.5*FN)`. Do not average per-image PQ for model selection. Per-image and macro values are separate diagnostics.
- The core scorer returns 0 when the denominator is zero. For a supplied empty-GT row it counts all predictions as FP; for an empty-prediction row it counts all GT as FN.
- The CSV wrapper discovers annotation passes from GT filament rows only. Thus genuinely empty-GT passes and prediction-only images are not represented by that wrapper. `evaluate_passes` mirrors this default and explicitly reports excluded empty passes. `include_empty_gt=True` is an additional diagnostic, not wrapper-equivalent scoring.
- Split/merge plots use any positive IoU, independently of the 0.5 PQ threshold. IoU/Dice histograms likewise include all positive-overlap pairs.

## Parity verification

`tests/test_metric_rle_submission.py` checks the source hash, extracts the original functions unchanged from notebook cells, and executes them on the same masks as our implementation. Cases cover perfect predictions, missing/extra objects, exactly-0.5 splits, merges, empty images and overlapping predictions. A full-size CSV fixture exercises original ID parsing, per-annotator duplication and pooled aggregation. Tests require pandas and torch, as used by the reference. The original notebook is not redistributed without a confirmed license. When it is absent these specific parity checks explicitly skip; a passing suite with skipped parity checks does not establish metric parity. Download the pinned reference using the command above to run them.

Synthetic parity is not validation performance. An additional competition-data fixture checks 20 supplied training polygons against `COCO.annToMask` exactly, and scores deliberately shifted/partly omitted training masks against every pass of one multiply annotated physical image using both the original CSV wrapper and our implementation. This is a scorer/rasterizer check, not model-performance evidence.

Local verification on 15 November 2026: `python -m pytest tests/test_metric_rle_submission.py -q` completed with **19 passed** in 9.74 seconds, including the attached-data parity fixture and a weak-reference regression check proving that metrics/export retain at most two generated dense component masks. The 321 warnings were upstream pycocotools/NumPy deprecation messages. No real-data model performance is implied by this result.

## Submission contract

The [official submission specification](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/organizers) requires exactly `filament_id,segmentation_rle`, one row per detected filament, IDs `<exact-image-stem>_<unique-suffix>`, and compressed COCO RLE counts without a serialized size field. All masks are 2048 × 2048. No extra empty-mask row represents a zero-detection image.

`write_submission` requires all image keys, including `[]` for zero detections; preserves a processed-image coverage sidecar; rejects empty, overlapping, nonbinary and wrong-sized masks; and checks the asymmetric, column-major RLE round trip. `validate_submission` needs the expected and processed image manifests, rejects malformed compressed streams before native decoding, and checks IDs, uniqueness, coverage, nonempty masks, dimensions and overlap. These checks establish file integrity, not competition acceptance.
