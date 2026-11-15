# Solar filament decisions

## 2026-11-15 — Native-scale semantic baseline first
Problem: fresh project with a live deadline and no local NVIDIA GPU.
Chosen: the supplied execution plan: ResNet34 U-Net, annotation-pass sampling, date/hash grouped holdout, official-scoring parity, full-image tiled inference and cached postprocessing sweeps. Use the user's authorized free Kaggle compute.
Alternatives: detector/refiner cascade and large instance models have greater engineering and runtime uncertainty; defer until measured errors justify them. Do not reuse a full-training public checkpoint as held-out evidence.
Effect: get a verified baseline before spending compute on extensions.

## 2026-11-15 — Reproduce the official scorer exactly
Problem: organizer implementation counts all pairs with IoU strictly greater than 0.5 and pools across annotation passes; a generic one-to-one PQ library would differ.
Chosen: pin the downloaded notebook and compare a local implementation against its executed functions, retaining diagnostic conventional metrics separately. Produce disjoint predictions and never exploit overlapping duplicate masks.
Alternative: Hungarian assignment would change competition semantics.
Effect: model selection measures the competition objective without gaming it.

## 2026-11-15 — Group adjacent observation dates conservatively
Problem: same-day grouping alone leaves evolving solar structures on adjacent days across train/validation. Inspection of ten candidate pairs confirmed persistent structures.
Chosen: default `date_gap_days=1`; union same-day and adjacent-calendar-date images across instruments, transitively, plus exact image hashes. Preserve every annotation pass in its physical observation's fold. This refines the grouping in the baseline decision above.
Alternatives: same-day-only grouping leaves a measured leakage risk; selecting approximate duplicates by thumbnail threshold is unreliable because solar disks share background geometry.
Rationale: all 179 adjacent-date pairs can be grouped without giant clusters (502 groups; largest five images). Three folds remain balanced at 236/236/235 physical images.
Effect: validation is more conservative while retaining all 707 permitted physical training observations. See [data audit](../docs/data-audit.md).
