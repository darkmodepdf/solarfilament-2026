"""Whole-image original-pass evaluation and cached probability ablations."""
from pathlib import Path
import time
import numpy as np
from PIL import Image

from .common import write_json, read_json, sha256_file
from .data import rasterize_pass
from .metrics import evaluate_passes
from .postprocess import instances_from_probability
from .inference import predict_probability


def validation_images(manifest, splits, fold):
    images = [item for item in manifest["images"] if splits["assignments"][item["physical_id"]] == fold]
    if not images:
        raise ValueError("Validation fold is empty")
    return images


def evaluate_model(model, manifest, images, config, device, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    def records():
        for i,item in enumerate(images):
            with Image.open(Path(manifest["image_root"])/item["file_name"]) as source:
                gray = np.asarray(source.convert("L"))
            probability = predict_probability(model, gray, config, device)
            np.save(cache_dir / (item["physical_id"] + ".npy"), probability.astype(np.float32))
            masks = instances_from_probability(probability, config["threshold"], config["min_area"], config.get("low_threshold"))
            for annotation_pass in item["passes"]:
                yield {"image_id":item["physical_id"],"annotation_pass_id":annotation_pass["annotation_pass_id"],
                       "gt_masks":rasterize_pass(annotation_pass,item["height"],item["width"]),"pred_masks":masks}
            if (i+1)%20 == 0:
                print(f"validate {i+1}/{len(images)}",flush=True)
    return evaluate_passes(records())


def tune_cached(manifest, splits, cache_dir, output, fold=0, deadline=None, checkpoint=None):
    images = validation_images(manifest,splits,fold)
    cache_dir=Path(cache_dir)
    provenance=read_json(cache_dir/"provenance.json")
    checkpoint=Path(checkpoint) if checkpoint else cache_dir.parent/"best.pt"
    if (provenance["split_hash"]!=splits["split_hash"] or
            provenance["data_hash"]!=manifest["manifest_hash"] or
            provenance["checkpoint_sha256"]!=sha256_file(checkpoint)):
        raise ValueError("Probability cache does not belong to this checkpoint/data/split")
    missing=[item["physical_id"] for item in images if not (cache_dir/(item["physical_id"]+".npy")).exists()]
    if missing:
        raise ValueError(f"Missing cached validation probabilities: {missing[:5]}")
    # Cache instances and ground truth per image only: bounded RAM even at native scale.
    from .metrics import panoptic_quality, aggregate_scores
    candidates=[{"threshold":threshold,"min_area":area,"low_threshold":low}
                for threshold in (0.35,0.45,0.55,0.65)
                for area in (0,32,128,256)
                for low in (None, max(0.15,threshold-0.15))]
    # Finish complete-fold candidates one at a time; never select on a partial fold.
    baseline={"threshold":0.45,"min_area":32,"low_threshold":None}
    candidates.remove(baseline)
    candidates.insert(0,baseline)
    rows=[]
    for candidate in candidates:
        scores=[]
        for item in images:
            if deadline is not None and time.time()>=deadline:
                if not rows:
                    raise TimeoutError("No complete tuning candidate before deadline; retain fallback")
                return rows[0]["config"]
            probability=np.load(cache_dir/(item["physical_id"]+".npy"))
            pred=instances_from_probability(probability,**candidate)
            for record in item["passes"]:
                masks=rasterize_pass(record,item["height"],item["width"])
                if masks:
                    scores.append(panoptic_quality(masks,pred))
        rows.append({"config":candidate,**aggregate_scores(scores)})
        rows.sort(key=lambda row:(-row["pq"],row["config"]["low_threshold"] is not None,row["config"]["min_area"]))
        write_json(output,{"selected":rows[0]["config"],"split_hash":splits["split_hash"],
                           "cache_dir":str(cache_dir),"provenance":provenance,"completed_candidates":len(rows),"results":rows})
        print(f"tune {len(rows)}/{len(candidates)} best_PQ={rows[0]['pq']:.6f}",flush=True)
    return rows[0]["config"]
