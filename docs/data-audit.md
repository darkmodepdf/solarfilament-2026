# Competition data audit — 15 November 2026

Downloaded only `filament-segmentation-2026` competition files with the authenticated Kaggle CLI. Archive size: 703,574,877 bytes; SHA-256: `56d8cd2927859fa7df72bd28ad1586f53c5814df6f81db02e755223e430657df`. The archive contains 887 images and one training annotation JSON. No external MAGFiLO ground truth was obtained.

| Audited item | Count |
|---|---:|
| Physical training observations | 707 |
| Independent annotation-pass image records | 1,154 |
| Original instance annotations | 8,199 |
| Supplied test observations | 180 |
| Exact duplicate training image hashes | 0 |
| Empty rasterized original instances | 0 |
| Passes containing overlapping original instances | 5 |
| Adjacent-calendar-date observation pairs | 179 |

Every supplied image is 2048 × 2048. The six instrument suffixes are Bh, Ch, Lh, Mh, Th and Uh. Training years span 2011–2022. Counts come from the downloaded competition data and COCO polygon rasterization, not from participant documentation.

The input JSON contains `spine` and other fields beyond segmentation. The manifest retains only each instance's `id` and original `segmentation`. Its area statistics are computed from permitted polygons. All categories are foreground. Each physical observation contains separate, complete annotation passes, preserving original overlapping instances and annotator disagreements. Empty observations would remain represented; no fake instance is introduced.

The initial same-date/hash grouping produced 656 groups. Visual inspection of the ten most similar adjacent-date pairs showed persistent evolving structures, including the 2016-02-25/26, 2016-10-12/13 and 2011-03-25/26 pairs. The final default conservatively unions **all** adjacent-calendar-date observations across instruments (`date_gap_days=1`) as well as exact image hashes. Temporal chains are transitive; this produces 502 groups, largest size five. This deliberately avoids choosing leakage boundaries from a thumbnail similarity threshold. Approximate high-pass similarity is retained only as an audit aid.

The deterministic three-fold allocator balances physical-image count, year, instrument and low/medium/high filament density. Every pass inherits its physical observation's fold. Group, capture-date and exact-hash isolation are asserted. Split and manifest hashes are written to `artifacts/splits.json` and `artifacts/manifest.json`; fresh Kaggle preparation uses the same default temporal grouping.

Twenty varied raw/overlay pairs were rendered and inspected in `artifacts/audit/annotations-01.jpg` through `annotations-04.jpg`, with selected IDs and pass IDs in `selected-images.json`. Samples include crowded disks, thin/faint filaments, extensive structures and limb cases. COCO masks align with the source orientation; no systematic flip or coordinate offset was observed at contact-sheet scale. Some faint extensions and grouping decisions are visibly ambiguous. Original validation labels remain unchanged. This inspection is not a claim of exhaustive pixel-by-pixel label correctness.

The ten adjacent-date pair comparisons are in `artifacts/audit/adjacent-dates-01.jpg` and `adjacent-dates-02.jpg`. All 179 adjacent pairs are grouped, including those not individually inspected. `artifacts/test-images.json` records exact expected test stems, dimensions and image hashes for coverage checks.

Reproduce the audit with `python scripts/audit_data.py --data-root data/raw/MAGFiLO_1.0_Kaggle_2026`. The CLI `solarfilament prepare` builds the same manifest/folds without visualization. Data/split unit checks cover COCO orientation, disconnected polygons within one instance, pass separation, prohibited-field exclusion, empty passes, overlapping labels, transitive temporal/hash groups, dimension errors, tampering, deterministic fold balance and leakage detection. No trained-model quality is implied by these checks.
