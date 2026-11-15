"""Build a private, self-contained Kaggle notebook without uploading or launching it."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile


def cell(source: str, markdown: bool = False) -> dict:
    result = {"cell_type": "markdown" if markdown else "code", "metadata": {},
              "source": textwrap.dedent(source).strip()+"\n"}
    if not markdown:
        result.update(execution_count=None, outputs=[])
    return result


def package(root: Path, output: Path, *, owner: str, slug: str, title: str,
            max_steps: int = 1000, total_hours: float = 10, smoke_only: bool = False, fold: int = 0,
            resume_kernel: str | None = None, machine_shape: str = "NvidiaTeslaT4") -> dict:
    if max_steps < 1 or not 1 < total_hours <= 10:
        raise ValueError("max_steps must be positive and total_hours must be >1 and <=10")
    if fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1 or 2 for the fixed three-fold split")
    root, output = root.resolve(), output.resolve()
    files = []
    for pattern in ("src/**/*.py", "configs/*.yaml", "configs/*.json", "tests/*.py",
                    "vendor/official/*", "requirements*.txt", "pyproject.toml"):
        files.extend(p for p in root.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    files.append(root / "scripts/preflight.py")
    for required in ("src/solarfilament/cli.py", "src/solarfilament/training.py",
                     "configs/baseline.yaml", "requirements.txt", "scripts/preflight.py"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"Build not ready: missing {required}")
    hashes, archive = {}, io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(set(files)):
            if path.stat().st_size > 8*1024**2:
                raise ValueError(f"Unexpected large source artifact: {path}")
            name = path.relative_to(root).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, content)
    payload = archive.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    settings = {"max_steps": max_steps, "total_hours": total_hours, "smoke_only": smoke_only, "fold": fold,
                "resume_kernel": resume_kernel, "source_sha256": digest}
    cells = [cell("""
        # Solar Filament — reproducible training and prediction

        Uses only official competition images and segmentation annotations. One complete
        annotation pass is sampled for each physical-image crop. All physical-image/date
        groups remain together during validation. The pinned official evaluator is tested
        before training. The source archive below is an exact, checksummed build of the
        accompanying repository; no private credentials or competition data are embedded.

        A successful run produces checkpoints, validation evidence, selected parameters,
        and a validated CSV under `/kaggle/working/solarfilament/runs`. It does not publish
        code, select a final submission, or complete the organizer form automatically.
        For two allocated GPUs it starts one DDP worker per GPU. Setup, smoke testing,
        training, evaluation and inference share the same session clock.
    """, True), cell(f"""
        import time
        SESSION_START = time.time()  # Starts before extraction, installation, or imports.
        import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
        SETTINGS = json.loads({json.dumps(json.dumps(settings))})
        ROOT = pathlib.Path('/kaggle/working/solarfilament')
        ROOT.mkdir(parents=True, exist_ok=True)
        encoded = {encoded!r}
        archive = base64.b64decode(encoded)
        assert hashlib.sha256(archive).hexdigest() == SETTINGS['source_sha256']
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for member in bundle.infolist():
                target = (ROOT / member.filename).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise ValueError('Unsafe archive path')
            bundle.extractall(ROOT)
        (ROOT / 'source-manifest.json').write_text(json.dumps({hashes!r}, indent=2))
        os.chdir(ROOT)
        os.environ['PYTHONPATH'] = str(ROOT / 'src')
        sys.path.insert(0, str(ROOT / 'src'))
        (ROOT / 'runs').mkdir(exist_ok=True)
        (ROOT / 'runs/session.json').write_text(json.dumps({{'session_start_unix': SESSION_START, **SETTINGS}}, indent=2))
        def run(args):
            remaining = SESSION_START + SETTINGS['total_hours'] * 3600 - time.time()
            if remaining <= 0:
                raise RuntimeError('Total session budget exhausted')
            print('+', ' '.join(map(str,args)), flush=True)
            subprocess.run(list(map(str,args)), check=True, timeout=remaining, cwd=ROOT)
        run([sys.executable, 'scripts/preflight.py', '--output-dir', 'runs/preflight-before-setup',
             '--session-start', SESSION_START, '--total-hours', SETTINGS['total_hours'], '--require-gpu'])
    """), cell(r"""
        # Preserve the platform's working CUDA/PyTorch stack. Pin every added dependency.
        import importlib.metadata
        constraints = []
        for package in ('torch', 'torchvision'):
            constraints.append(f'{package}=={importlib.metadata.version(package)}')
        pathlib.Path('runtime-constraints.txt').write_text('\n'.join(constraints)+'\n')
        requirements = []
        for filename in ('requirements.txt', 'requirements-dev.txt'):
            for line in pathlib.Path(filename).read_text().splitlines():
                line = line.strip()
                if not line or line.startswith(('#', '-')):
                    continue
                if line.split('==')[0].lower() not in ('torch','torchvision'):
                    requirements.append(line)
        run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--prefer-binary',
             '--constraint', 'runtime-constraints.txt', *requirements])
        run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', '.'])
        with open('runs/environment-freeze.txt', 'w') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, stdout=stream)
        run([sys.executable, 'scripts/preflight.py', '--output-dir', 'runs/preflight',
             '--session-start', SESSION_START, '--total-hours', SETTINGS['total_hours'], '--require-gpu'])
        run([sys.executable, '-m', 'pytest', '-q'])
    """), cell("""
        import torch, yaml
        matches = list(pathlib.Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one official competition data root, found {len(matches)}')
        DATA_ROOT = matches[0].parent.parent
        run([sys.executable, '-m', 'solarfilament.cli', 'prepare', '--data-root', DATA_ROOT,
             '--output-dir', 'runs/data'])
        GPU_COUNT = torch.cuda.device_count()
        if GPU_COUNT < 1:
            raise RuntimeError('No GPU available')
        TRAIN_PREFIX = ([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                         f'--nproc_per_node={GPU_COUNT}', '-m', 'solarfilament.cli']
                        if GPU_COUNT > 1 else [sys.executable, '-m', 'solarfilament.cli'])
        COMMON_TRAIN = ['train', '--manifest', 'runs/data/manifest.json',
                        '--splits', 'runs/data/splits.json', '--session-start', SESSION_START,
                        '--total-hours', SETTINGS['total_hours']]
        config = yaml.safe_load(pathlib.Path('configs/baseline.yaml').read_text())
        config['fold'] = SETTINGS['fold']
        pathlib.Path('configs/baseline.yaml').write_text(yaml.safe_dump(config))
        smoke = {**config, 'encoder_weights': None, 'crop_size': 128, 'batch_size': 1,
                 'effective_batch_size': max(2, GPU_COUNT), 'workers': 0,
                 'max_steps': 4, 'skip_validation': True, 'validation_limit': 1}
        pathlib.Path('runs/smoke.yaml').write_text(yaml.safe_dump(smoke))
        run([*TRAIN_PREFIX, *COMMON_TRAIN, '--config', 'runs/smoke.yaml',
             '--output-dir', 'runs/smoke', '--max-steps', 2])
        assert pathlib.Path('runs/smoke/last.pt').is_file(), 'Recovery checkpoint missing'
        run([*TRAIN_PREFIX, *COMMON_TRAIN, '--config', 'runs/smoke.yaml',
             '--output-dir', 'runs/smoke', '--max-steps', 4, '--resume', 'runs/smoke/last.pt'])
        from solarfilament.inference import load_checkpoint_model, predict_probability
        import numpy as np
        smoke_model, smoke_checkpoint = load_checkpoint_model('runs/smoke/last.pt', 'cuda:0')
        assert smoke_checkpoint['global_step'] == 4, 'Resume did not restore and advance the expected step'
        synthetic = np.full((128,128), 128, dtype=np.uint8)
        smoke_probability = predict_probability(smoke_model, synthetic, smoke_checkpoint['config'], 'cuda:0')
        assert smoke_probability.shape == synthetic.shape and np.isfinite(smoke_probability).all()
        del smoke_model, smoke_checkpoint, smoke_probability
        # One cleanup between distinct runs, never per training step.
        torch.cuda.empty_cache()
        print('GPU train/save/restart/resume and synthetic tiled inference passed.')
        # Prove the real crop, batch, initialization and one native full-image validation.
        production_smoke = {**config, 'skip_validation': False, 'validation_limit': 1,
                            'validation_every': 2}
        pathlib.Path('runs/production-smoke.yaml').write_text(yaml.safe_dump(production_smoke))
        run([*TRAIN_PREFIX, *COMMON_TRAIN, '--config', 'runs/production-smoke.yaml',
             '--output-dir', 'runs/production-smoke', '--max-steps', 2])
        assert pathlib.Path('runs/production-smoke/best.pt').is_file()
        assert pathlib.Path('runs/production-smoke/validation/provenance.json').is_file()
        record_initialization_assets()
        print('Actual training crop/batch and native-resolution PQ validation passed.')
    """), cell("""
        # Main baseline; training itself reserves time for saving/evaluation/inference.
        if not SETTINGS['smoke_only']:
            main_args = [*TRAIN_PREFIX, *COMMON_TRAIN, '--config', 'configs/baseline.yaml',
                         '--output-dir', 'runs/baseline', '--max-steps', SETTINGS['max_steps']]
            if SETTINGS['resume_kernel']:
                candidates = list(pathlib.Path('/kaggle/input').rglob('baseline/last.pt'))
                if len(candidates) != 1:
                    raise RuntimeError(f'Expected one attached baseline recovery checkpoint, found {len(candidates)}')
                # Carry the previous best checkpoint AND its matching probability cache.
                # Otherwise a continuation without a new best would have no exportable best.pt.
                import shutil
                shutil.copytree(candidates[0].parent, pathlib.Path('runs/baseline'), dirs_exist_ok=True)
                main_args += ['--resume', candidates[0]]
            run(main_args)
            assert pathlib.Path('runs/baseline/best.pt').is_file(), 'No PQ-selected checkpoint was saved'
            # Export a checked fallback before spending any time on threshold tuning.
            fallback_started = time.time()
            run([sys.executable, '-m', 'solarfilament.cli', 'predict',
                 '--checkpoint', 'runs/baseline/best.pt', '--image-dir', DATA_ROOT/'test/test_images',
                 '--output-dir', 'runs/fallback'])
            run([sys.executable, '-m', 'solarfilament.cli', 'validate',
                 '--submission', 'runs/fallback/submission.csv', '--image-dir', DATA_ROOT/'test/test_images',
                 '--processed', 'runs/fallback/processed_images.json'])
            import shutil
            shutil.copytree('runs/fallback', 'runs/final', dirs_exist_ok=True)
            fallback_seconds = time.time()-fallback_started
            selection = {'chosen': 'fallback', 'fallback_seconds': fallback_seconds}
            total_deadline = SESSION_START + SETTINGS['total_hours']*3600
            tuning_deadline = total_deadline - 1.5*fallback_seconds - 300
            selection['tuning_deadline_unix'] = tuning_deadline
            if time.time() < tuning_deadline:
                try:
                    run([sys.executable, '-m', 'solarfilament.cli', 'tune',
                         '--manifest', 'runs/data/manifest.json', '--splits', 'runs/data/splits.json',
                         '--cache-dir', 'runs/baseline/validation', '--output', 'runs/selected.json',
                         '--deadline', tuning_deadline, '--fold', SETTINGS['fold']])
                    if pathlib.Path('runs/selected.json').is_file() and time.time()+1.5*fallback_seconds+120 < total_deadline:
                        run([sys.executable, '-m', 'solarfilament.cli', 'predict',
                             '--checkpoint', 'runs/baseline/best.pt', '--image-dir', DATA_ROOT/'test/test_images',
                             '--output-dir', 'runs/candidate-final', '--config', 'runs/selected.json'])
                        run([sys.executable, '-m', 'solarfilament.cli', 'validate',
                             '--submission', 'runs/candidate-final/submission.csv', '--image-dir', DATA_ROOT/'test/test_images',
                             '--processed', 'runs/candidate-final/processed_images.json'])
                        shutil.copytree('runs/candidate-final', 'runs/final', dirs_exist_ok=True)
                        selection['chosen'] = 'tuned'
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    # Preserve the already validated final fallback; document failed tuning.
                    selection['tuning_or_candidate_error'] = str(exc)
                    print('Keeping validated fallback:', exc, flush=True)
            else:
                selection['tuning_skipped_reason'] = 'Insufficient time after measured fallback inference'
            pathlib.Path('runs/final-selection.json').write_text(json.dumps(selection, indent=2))
            record_initialization_assets()
            print('Submission file validated. Download and check it before uploading to the competition.')
        total_bytes = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        summary = {'elapsed_seconds': time.time()-SESSION_START, 'gpu_count': GPU_COUNT,
                   'physical_gpu_hours': GPU_COUNT*(time.time()-SESSION_START)/3600,
                   'saved_output_bytes': total_bytes, 'smoke_only': SETTINGS['smoke_only']}
        pathlib.Path('runs/completion.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        if total_bytes > 19*1024**3:
            raise RuntimeError('Outputs approach Kaggle saved-output limit; trim caches before declaring success')
    """)]
    # IPython may preload NumPy before pip replaces it. A fresh interpreter is mandatory
    # for scientific imports after setup: reusing the kernel mixes old binary extensions
    # with new package Python files (observed on the first Kaggle smoke run).
    runner_prelude = textwrap.dedent("""
        import hashlib, json, os, pathlib, subprocess, sys, time
        ROOT = pathlib.Path.cwd()
        SETTINGS = json.loads(pathlib.Path('runs/session.json').read_text())
        SESSION_START = SETTINGS['session_start_unix']
        def run(args):
            remaining = SESSION_START + SETTINGS['total_hours']*3600 - time.time()
            if remaining <= 0:
                raise RuntimeError('Total session budget exhausted')
            print('+', ' '.join(map(str,args)), flush=True)
            subprocess.run(list(map(str,args)), check=True, timeout=remaining, cwd=ROOT)
        def record_initialization_assets():
            provenance = []
            for cache_root in (pathlib.Path.home()/'.cache/torch/hub/checkpoints',
                               pathlib.Path.home()/'.cache/huggingface/hub/models--smp-hub--resnet34.imagenet/snapshots'):
                if cache_root.exists():
                    for asset in sorted(cache_root.rglob('*')):
                        if asset.is_file() and asset.suffix.lower() in {'.pt','.pth','.bin','.safetensors','.json'}:
                            digest = hashlib.sha256()
                            with asset.open('rb') as stream:
                                for chunk in iter(lambda: stream.read(1024*1024), b''):
                                    digest.update(chunk)
                            provenance.append({'path': str(asset), 'bytes': asset.stat().st_size,
                                               'sha256': digest.hexdigest()})
            pathlib.Path('runs/initialization-assets.json').write_text(json.dumps(provenance, indent=2))
            print('Initialization assets:', json.dumps(provenance), flush=True)
    """).strip()+"\n"
    runner_source = runner_prelude + cells[3]["source"] + "\n" + cells[4]["source"]
    compile(runner_source, "run_kaggle.py", "exec")
    runner_digest = hashlib.sha256(runner_source.encode("utf-8")).hexdigest()
    cells[3] = cell("# Write the scientific workflow for a clean interpreter after installation.\n"
                    + "runner_source = " + repr(runner_source) + "\n"
                    + f"assert hashlib.sha256(runner_source.encode('utf-8')).hexdigest() == {runner_digest!r}\n"
                    + f"SETTINGS['runner_sha256'] = {runner_digest!r}\n"
                    + "pathlib.Path('runs/session.json').write_text(json.dumps({'session_start_unix': SESSION_START, **SETTINGS}, indent=2))\n"
                    + "pathlib.Path('scripts/run_kaggle.py').write_text(runner_source, encoding='utf-8')\n")
    cells[4] = cell("# Avoid scientific imports in the notebook's potentially stale package state.\n"
                    + "run([sys.executable, 'scripts/run_kaggle.py'])\n")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python", "version": "3.11"}},
                "nbformat": 4, "nbformat_minor": 5}
    for index, entry in enumerate(cells):
        entry["id"] = f"solarfilament-{index:02d}"
        if entry["cell_type"] == "code":
            compile(entry["source"], f"notebook-cell-{index}", "exec")
    metadata = {"id": f"{owner}/{slug}", "title": title, "code_file": "train_and_submit.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "machine_shape": machine_shape, "competition_sources": ["filament-segmentation-2026"],
                "dataset_sources": [], "model_sources": [],
                "kernel_sources": [resume_kernel] if resume_kernel else []}
    output.mkdir(parents=True, exist_ok=True)
    notebook_text = json.dumps(notebook, indent=1)+"\n"
    (output / "train_and_submit.ipynb").write_text(notebook_text, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    (output / "source-manifest.json").write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    result = {"output": str(output), "source_sha256": digest, "runner_sha256": runner_digest,
              "notebook_sha256": hashlib.sha256(notebook_text.encode("utf-8")).hexdigest(), "source_files": len(hashes),
              "archive_bytes": len(payload), "kernel": metadata["id"], "settings": settings}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("build/kaggle"))
    parser.add_argument("--owner", default="vishwasbollam")
    parser.add_argument("--slug", default="solarfilament-annotation-aware-baseline")
    parser.add_argument("--title", default="Solarfilament Annotation Aware Baseline")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--total-hours", type=float, default=10)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume-kernel")
    parser.add_argument("--machine-shape", default="NvidiaTeslaT4")
    package(**vars(parser.parse_args()))
