"""Check local release artifacts without claiming publication or prize eligibility."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_release(root: Path, report: Path, *, require_git: bool = True) -> dict:
    root = root.resolve()
    report = report if report.is_absolute() else root / report
    required = [root / name for name in (
        "README.md", "requirements.txt", "notebooks/inference.ipynb",
        "weights/final.pt", "weights/provenance.json", "submissions/final.csv",
        "submissions/final.csv.manifest.json", "artifacts/test-images.json",
        "src/solarfilament/inference.py", "src/solarfilament/submission.py",
    )] + [report]
    required = list(dict.fromkeys(required + sorted((root / "src/solarfilament").glob("*.py"))))
    checks, details = {}, {}
    missing = [str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
               for path in required if not path.is_file() or path.stat().st_size == 0]
    checks["required_artifacts_present"] = not missing
    details["missing_artifacts"] = missing
    provenance_path = root / "weights/provenance.json"
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            for field, relative in (("weights_sha256", "weights/final.pt"),
                                    ("submission_sha256", "submissions/final.csv")):
                path = root / relative
                checks[field] = path.is_file() and sha256(path) == provenance.get(field)
            checks["provenance_has_run_identity"] = all(
                key in provenance for key in ("data_hash", "split_hash", "global_step", "config", "versions")
            )
            details["provenance"] = provenance
        except (ValueError, TypeError, OSError) as error:
            checks["provenance_parse"] = False
            details["provenance_error"] = str(error)
    manifest_path = root / "artifacts/test-images.json"
    coverage_path = root / "submissions/final.csv.manifest.json"
    submission = root / "submissions/final.csv"
    if all(path.is_file() for path in (manifest_path, coverage_path, submission)):
        try:
            sys.path.insert(0, str(root / "src"))
            from solarfilament.submission import validate_submission
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = [record["physical_id"] for record in manifest]
            checks["test_image_count"] = len(expected) == 180 and len(set(expected)) == 180
            checks["test_image_dimensions"] = all((r["height"], r["width"]) == (2048, 2048) for r in manifest)
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            details["submission_validation"] = validate_submission(submission, expected, coverage["processed_images"])
            checks["submission_valid"] = True
            checks["coverage_sidecar_checksum"] = coverage.get("sha256") == sha256(submission)
        except (ImportError, ValueError, TypeError, KeyError, OSError) as error:
            checks["submission_valid"] = False
            details["submission_error"] = str(error)
    notebook = root / "notebooks/inference.ipynb"
    if notebook.is_file():
        try:
            content = json.loads(notebook.read_text(encoding="utf-8"))
            code_cells = ["".join(c["source"]) for c in content["cells"] if c["cell_type"] == "code"]
            for index, code in enumerate(code_cells):
                compile(code, f"inference.ipynb:{index}", "exec")
            checks["notebook_code_compiles"] = bool(code_cells)
            checks["notebook_outputs_cleared"] = all(
                not c.get("outputs") and c.get("execution_count") is None
                for c in content["cells"] if c["cell_type"] == "code"
            )
            details["notebook_code_cells"] = len(code_cells)
        except (ValueError, TypeError, KeyError, SyntaxError) as error:
            checks["notebook_code_compiles"] = False
            details["notebook_error"] = str(error)
    if report.is_file():
        with report.open("rb") as handle:
            checks["report_pdf_header"] = handle.read(5) == b"%PDF-"
        try:
            from pypdf import PdfReader
            details["report_total_pages"] = len(PdfReader(report).pages)
        except ImportError:
            details["report_page_count_status"] = "pypdf is unavailable; count pages during manual report review"
        except Exception as error:
            checks["report_pdf_parse"] = False
            details["report_pdf_error"] = str(error)
    try:
        tracked_result = subprocess.run(["git", "ls-files", "-z"], cwd=root, text=True,
                                        capture_output=True, check=True)
        tracked = set(tracked_result.stdout.split("\0")) - {""}
        untracked = [path.relative_to(root).as_posix() for path in required
                     if path.is_relative_to(root) and path.relative_to(root).as_posix() not in tracked]
        outside = [str(path) for path in required if not path.is_relative_to(root)]
        checks["required_files_tracked_in_git"] = not untracked and not outside
        details["untracked_required_files"] = untracked
        details["required_files_outside_repository"] = outside
        forbidden = sorted(path for path in tracked if (
            path.startswith("data/") or Path(path).name in {".env", "kaggle.json"}
            or path in {"vendor/official/self-evaluation-notebook.ipynb", "vendor/official/kernel-metadata.json",
                        "notebooks/train_and_submit.ipynb"}
        ))
        checks["no_forbidden_release_paths"] = not forbidden
        details["forbidden_tracked_paths"] = forbidden
        ignored_result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=root,
            input="\n".join(path.relative_to(root).as_posix() for path in required if path.is_relative_to(root)) + "\n",
            text=True, capture_output=True)
        details["required_paths_matching_ignore_rules"] = ignored_result.stdout.splitlines()
        # A forced tracked weight is distributable, but keep ignore matches visible.
        details["git_note"] = "Tracking checks do not establish that a remote repository is publicly accessible."
    except (OSError, subprocess.CalledProcessError) as error:
        checks["required_files_tracked_in_git"] = not require_git
        details["git_error"] = str(error)
    return {
        "ready_for_manual_release_review": bool(checks) and all(checks.values()),
        "competition_completion_verified": False,
        "checks": checks, "details": details,
        "manual_gates": [
            "Confirm the organizer template and four pages of report content; references may add pages.",
            "Execute inference.ipynb with released weights and verify reproduction.json, not just notebook syntax.",
            "Check the public repository and weight links without privileged access.",
            "Verify Kaggle acceptance and final selection, then preserve submission IDs and receipts.",
            "Complete the organizer form before its separate cutoff and preserve the receipt.",
            "Review licenses, author/AI-assistance disclosures and participant eligibility.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=Path("report/final.pdf"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-no-git", action="store_true", help="Artifact-only local inspection, not a publication gate")
    args = parser.parse_args()
    result = check_release(args.root, args.report, require_git=not args.allow_no_git)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["ready_for_manual_release_review"] else 1)


if __name__ == "__main__":
    main()
