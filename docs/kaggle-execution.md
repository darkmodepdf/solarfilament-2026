# Kaggle execution

The packager produces a **private** self-contained notebook with an allowlisted source archive,
its SHA-256 digest, and file checksums. Data, credentials, unrelated files and model weights are
not embedded. Build again after any source/config/test change; the notebook is a source snapshot.

```powershell
python scripts/package_kaggle.py --output build/kaggle --max-steps 1000
kaggle quota --csv
kaggle kernels push -p build/kaggle --timeout 36000 --accelerator NvidiaTeslaT4
kaggle kernels status vishwasbollam/solarfilament-annotation-aware-baseline
kaggle kernels logs vishwasbollam/solarfilament-annotation-aware-baseline
```

Push starts Save & Run All. A launch is not proof of training success. Inspect logs and the
runtime preflight for allocated GPUs, actual CUDA/PyTorch versions, memory and elapsed time.
The notebook preserves the preinstalled torch/torchvision CUDA builds using pip constraints,
pins additions from requirements files, records the resulting package freeze, runs tests,
prepares actual data, tests four-step GPU recovery (stop at step 2, restart, resume to 4),
tests synthetic tiled inference, and tests two production-sized training steps with one
native-resolution validation image. All scientific work runs in a fresh Python interpreter
after dependency installation; the persistent notebook must not reuse preloaded NumPy modules.
It then runs the baseline and exports a validated fallback before tuning held-out cached
probabilities. Tuning has a deadline reserving 1.5 times the measured inference/validation
duration plus five minutes; a failed or unfinished tune retains the checked fallback.
No competition submission,
public publication, final-selection action or organizer form is performed by this notebook.

The default baseline stops at optimizer step 1000 but keeps the 4000-step learning-rate recipe
so it can continue faithfully. With two T4 GPUs it launches two DDP processes. Total budget is
10 hours starting before extraction/install, including a configured one-hour final reserve.
Hardware runtime, output availability and throughput still need confirmation from the live run.

`--fold 0`, `--fold 1`, or `--fold 2` selects a member of the same frozen split for an independent
run; both training and tuning receive that fold. Initial fallback runs use `--total-hours 4`,
with one hour reserved for final work. The runtime inspects actual devices: an accepted L4
request was allocated T4x2 by Kaggle, so requested metadata is never used as hardware evidence.
All scientific work after package installation runs in a fresh interpreter to avoid stale
NumPy imports in the persistent notebook. A production-size two-step smoke also tests a real
complete-image validation and cache/checkpoint provenance before the long run. A checked
fallback CSV is saved before deadline-bounded tuning.

After successful completion, save durable outputs locally:

```powershell
kaggle kernels output vishwasbollam/solarfilament-annotation-aware-baseline -p runs/kaggle-baseline
```

Validate the downloaded CSV and preserve the exact source digest, checkpoint checksums,
package freeze, split/manifest hashes, settings and Kaggle version. Freeze the selected
post-processing parameters. Re-run inference to verify decoded-mask equivalence before
claiming reproducibility. The submitted report must use actual measured outcomes.

A continuation uses the completed notebook output as a read-only input and copies its
checkpoint and matching best validation cache into writable `/kaggle/working`:

```powershell
python scripts/package_kaggle.py --output build/kaggle-continue --max-steps 4000 --slug solarfilament-annotation-aware-continuation --title "Solarfilament Annotation Aware Continuation" --resume-kernel vishwasbollam/solarfilament-annotation-aware-baseline
```

Use the same data, split, world size and recipe on resume. Check the logged restored step/LR.
Estimate the remaining training + validation + inference duration from the first run before
launching. Recheck free quota and the earlier organizer deadline; calendar time overrides the
nominal ten-hour run budget. Do not start the continuation if it jeopardizes final materials.

For an isolated GPU smoke run, add `--smoke-only` and use a distinct slug/title. It still
prepares data and tests recovery but does not train the competition model or create predictions.
