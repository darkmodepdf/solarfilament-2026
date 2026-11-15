"""Reproducible command line entry points, also used by the Kaggle notebook."""
import argparse
from pathlib import Path
import json
import yaml

from .common import read_json, write_json


def load_config(path):
    result=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "selected" in result:
        result=result["selected"]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    prepare=commands.add_parser("prepare")
    prepare.add_argument("--data-root",required=True)
    prepare.add_argument("--output-dir",required=True)
    prepare.add_argument("--n-folds",type=int,default=3)
    prepare.add_argument("--seed",type=int,default=2026)
    for name in ("train","tune"):
        child=commands.add_parser(name)
        child.add_argument("--manifest",required=True)
        child.add_argument("--splits",required=True)
        if name=="train":
            child.add_argument("--config",required=True)
            child.add_argument("--output-dir",required=True)
            child.add_argument("--resume")
            child.add_argument("--max-steps",type=int)
            child.add_argument("--total-hours",type=float)
            child.add_argument("--session-start",type=float)
        else:
            child.add_argument("--cache-dir",required=True)
            child.add_argument("--output",required=True)
            child.add_argument("--fold",type=int,default=0)
            child.add_argument("--deadline",type=float)
            child.add_argument("--checkpoint")
    predict=commands.add_parser("predict")
    predict.add_argument("--checkpoint",required=True)
    predict.add_argument("--image-dir",required=True)
    predict.add_argument("--output-dir",required=True)
    predict.add_argument("--config")
    validate=commands.add_parser("validate")
    validate.add_argument("--submission",required=True)
    validate.add_argument("--image-dir",required=True)
    validate.add_argument("--processed",required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        from .data import build_manifest
        from .splits import make_splits
        root=Path(args.data_root)
        candidates=list(root.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
        if len(candidates)!=1:
            raise ValueError("Need exactly one official training annotation file under data-root")
        annotation=candidates[0]
        output=Path(args.output_dir)
        output.mkdir(parents=True,exist_ok=True)
        manifest=build_manifest(annotation,annotation.parent/"train_images",output/"manifest.json")
        splits=make_splits(manifest,n_folds=args.n_folds,seed=args.seed)
        write_json(output/"splits.json",splits)
        print(json.dumps({"images":len(manifest["images"]),"manifest_hash":manifest["manifest_hash"],"split_hash":splits["split_hash"]}),flush=True)
    elif args.command in ("train","tune"):
        from .splits import validate_splits
        manifest,splits=read_json(args.manifest),read_json(args.splits)
        validate_splits(manifest,splits)
        if args.command=="train":
            from .training import train
            config=load_config(args.config)
            if args.total_hours is not None: config["total_hours"]=args.total_hours
            train(manifest,splits,config,args.output_dir,args.resume,args.max_steps,args.session_start)
        else:
            from .evaluation import tune_cached
            print(tune_cached(manifest,splits,args.cache_dir,args.output,args.fold,args.deadline,args.checkpoint))
    elif args.command=="predict":
        from .inference import run_prediction
        print(run_prediction(args.checkpoint,args.image_dir,args.output_dir,load_config(args.config) if args.config else None))
    else:
        from .submission import validate_submission
        expected=[p.stem for p in Path(args.image_dir).iterdir() if p.suffix.lower() in {".jpeg",".jpg",".png"}]
        processed=read_json(args.processed)
        if isinstance(processed,dict): processed=processed["processed_images"]
        print(json.dumps(validate_submission(args.submission,expected,processed),indent=2))


if __name__=="__main__":
    main()
