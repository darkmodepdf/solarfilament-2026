"""Record real runtime resources before setup/training; never print credentials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.request


def collect_preflight(output_dir: str | Path, *, session_start: float | None = None,
                      total_hours: float = 10, reserve_minutes: float = 60,
                      require_gpu: bool = False) -> dict:
    start = session_start if session_start is not None else time.time()
    if total_hours <= 0 or reserve_minutes < 30 or reserve_minutes >= total_hours * 60:
        raise ValueError("Budget must be positive with >=30 minutes reserved and time left to train")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "scipy", "pycocotools",
                 "segmentation-models-pytorch", "timm", "pandas", "PyYAML", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(out)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "session_start_unix": start, "elapsed_seconds": max(0, time.time()-start),
              "total_budget_seconds": total_hours*3600, "reserve_seconds": reserve_minutes*60,
              "training_stop_unix": start+total_hours*3600-reserve_minutes*60,
              "completion_deadline_unix": start+total_hours*3600,
              "platform": platform.platform(), "python": platform.python_version(),
              "cpu_count": os.cpu_count(), "packages": packages,
              "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
              "output_dir": str(out.resolve()), "gpus": [],
              "documented_kaggle_session_limit_hours": 12,
              "documented_kaggle_saved_output_limit_gb": 20,
              "limits_source": "https://www.kaggle.com/docs/notebooks",
              "limits_checked_on": "2026-11-15"}
    try:
        import psutil
        ram = psutil.virtual_memory()
        report.update(ram_total_bytes=ram.total, ram_available_bytes=ram.available)
    except ImportError:
        if Path("/proc/meminfo").exists():
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            report["ram_total_bytes"] = int(mem["MemTotal"].split()[0])*1024
            report["ram_available_bytes"] = int(mem["MemAvailable"].split()[0])*1024
    try:
        import torch
        report["cuda_runtime"] = torch.version.cuda
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["gpus"].append({"index": index, "name": props.name,
                                   "total_memory_bytes": props.total_memory,
                                   "capability": [props.major, props.minor]})
    except (ImportError, RuntimeError) as exc:
        report["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                                "--format=csv"], capture_output=True, text=True, timeout=30)
        report["nvidia_smi"] = probe.stdout.strip()
    try:
        with urllib.request.urlopen("https://pypi.org/simple/pycocotools/", timeout=10) as response:
            report["internet_available"] = response.status == 200
    except Exception as exc:
        report["internet_available"] = False
        report["internet_probe_error"] = type(exc).__name__
    report["checks"] = {
        "gpu_available": bool(report["gpus"]),
        "budget_remaining": time.time() < report["training_stop_unix"],
        "free_disk_at_least_5gb": disk.free >= 5*1024**3,
    }
    (out / "preflight.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if require_gpu and not report["checks"]["gpu_available"]:
        raise RuntimeError("No CUDA GPU allocated; refusing to start the long training job")
    if not report["checks"]["budget_remaining"] or not report["checks"]["free_disk_at_least_5gb"]:
        raise RuntimeError("Insufficient remaining session budget or disk space")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/preflight")
    parser.add_argument("--session-start", type=float)
    parser.add_argument("--total-hours", type=float, default=10)
    parser.add_argument("--reserve-minutes", type=float, default=60)
    parser.add_argument("--require-gpu", action="store_true")
    collect_preflight(**vars(parser.parse_args()))
