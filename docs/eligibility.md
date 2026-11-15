# Eligibility and execution record

Verified 15 November 2026 from authenticated Kaggle CLI 2.2.4 and official pages.
Raw primary-source snapshots are retained locally in `docs/evidence/`; the public sources are linked below.

## Cutoffs and access

| Item | Observed value | UTC | IST |
|---|---|---|---|
| Kaggle prediction deadline | API `deadline` 2026-11-16T11:59:59 | 16 Nov 11:59:59 | 16 Nov 17:29:59 |
| Organizer report and solution deadline | Important Dates: 15 Nov 11:59 pm **CDT** | Literal CDT (UTC−05): 16 Nov 04:59 | 16 Nov 10:29 |

The organizer writes CDT although US Central civil time in November is normally CST.
Do not silently replace their stated zone: use the **earlier literal CDT cutoff** for planning.
The prediction API deadline does **not** extend the organizer form deadline.
The installed Kaggle SDK `DateTimeSerializer` removes the incoming `Z` suffix before creating
Python datetimes, explaining the timezone-free CLI display; its serializer adds `Z` again.
The API prediction deadline is therefore recorded in UTC. Confirm the displayed timeline
before final upload; this is not a promise that future submissions will be accepted.

- Account: `vishwasbollam`; `userHasEntered: true`.
- At inspection: zero lifetime submissions; zero today; five remaining today.
- Rules permit at most five submissions daily and up to two final selections.
- Prize conflict-of-interest and personal eligibility cannot be inferred from account access.
- Actual prediction acceptance and explicit final selection still require successful receipts.

Sources: [official Important Dates](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates),
[rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules),
`kaggle -W competitions list -s filament-segmentation-2026 --format json`, and
`kaggle -W competitions submission-limits filament-segmentation-2026 --json`.

## Required final materials

- Four-page PDF using the [official Overleaf template](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe).
- Public Git repository, versioned requirements, and an end-to-end notebook.
- Trained model weights stored in the repository and loaded by the notebook without retraining.
- Complete the [organizer Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform?usp=dialog).
- Repository stays publicly accessible from competition close through winner announcement.
- Quantitative criteria carry 70%; method, morphology, and code quality carry 30%.
- Code, report, and form remain incomplete until checked and receipt saved. The form's live
  accepting-responses state was not verified by the text-fetch tool (login/browser needed).

## Permitted inputs and understanding

The official rules allow reasonably accessible public external models/tools. The evaluation
page restricts model inference to the supplied test H-alpha images and prohibits other
ground-truth metadata. Use only competition segmentation masks as supervision; do not use
MAGFiLO spines, chirality, other public labels, test labels, or neighboring test observations.
ImageNet initialization is public external pretraining; preserve its provenance and license.
The [organizer clarification](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396)
was retrieved through the discussion API and is retained in the evidence directory.

AI-assisted development is allowed, but the participant must understand, explain, justify,
and reproduce the solution. This project must include a readable explanation and a measured
report; generated claims are not substitutes for experiments. Source and model licenses must
support the winning-entry obligations. Competition data is for noncommercial participation
and academic use and should not be committed into a public repository.

## Free compute

`kaggle -W quota --format json` showed 30.00 GPU hours available, zero used, and a refresh
timestamp of 2026-11-21T00:00:00. This is a point-in-time observation; recheck before launch.
The [official notebook documentation](https://www.kaggle.com/docs/notebooks) currently states
12-hour CPU/GPU sessions and 20 GB of saved `/kaggle/working` output. Keep a configurable
10-hour total budget measured before setup and at least one hour for final evaluation,
inference, and saves. Early fallback inference should finish well before this limit.

The [official CLI metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
maps `NvidiaTeslaT4` to T4×2. Actual allocation and available VRAM are checked inside the run;
two GPUs require explicit DDP, with one process per GPU and no pooled-VRAM assumption.
No paid compute is authorized or configured.
