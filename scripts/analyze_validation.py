"""Score frozen validation probabilities and render worst/random review panels."""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion

from solarfilament.common import read_json,write_json
from solarfilament.cli import load_config
from solarfilament.data import rasterize_pass
from solarfilament.evaluation import validation_images
from solarfilament.metrics import evaluate_passes
from solarfilament.postprocess import instances_from_probability


def overlay(gray,masks):
    rgb=np.repeat(gray[...,None],3,axis=2).astype(np.float32)
    palette=np.array([[255,80,80],[70,235,120],[80,150,255],[255,190,40],[220,90,255],[60,220,220]],np.float32)
    for i,mask in enumerate(masks):
        rgb[mask]=0.5*rgb[mask]+0.5*palette[i%len(palette)]
        border=mask & ~binary_erosion(mask)
        rgb[border]=palette[i%len(palette)]
    return Image.fromarray(np.clip(rgb,0,255).astype(np.uint8))


def analyze(manifest,splits,cache_dir,config,output_dir,image_root=None,fold=0):
    manifest,splits=read_json(manifest),read_json(splits)
    config=load_config(config)
    cache_dir,output_dir=Path(cache_dir),Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    root=Path(image_root or manifest["image_root"])
    items=validation_images(manifest,splits,fold)
    def records():
        for item in items:
            probability=np.load(cache_dir/(item["physical_id"]+".npy"))
            pred=instances_from_probability(probability,config["threshold"],config["min_area"],config.get("low_threshold"))
            for annotation_pass in item["passes"]:
                yield {"image_id":item["physical_id"],"annotation_pass_id":annotation_pass["annotation_pass_id"],
                       "gt_masks":rasterize_pass(annotation_pass,item["height"],item["width"]),"pred_masks":pred}
    scores=evaluate_passes(records())
    write_json(output_dir/"metrics.json",scores)
    order=sorted(items,key=lambda item:(scores["per_image"].get(item["physical_id"],{}).get("pq",0),item["physical_id"]))
    worst=order[:30]
    rest=[item for item in items if item not in worst]
    random.Random(2026).shuffle(rest)
    chosen=worst+rest[:20]
    with (output_dir/"review.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["physical_id","sample","pq","fp","fn","split_count","merge_count","dominant_error_manual"])
        writer.writeheader()
        for index,item in enumerate(chosen):
            s=scores["per_image"].get(item["physical_id"],{})
            writer.writerow({"physical_id":item["physical_id"],"sample":"worst" if index<30 else "random",
                             **{key:s.get(key,0) for key in ("pq","fp","fn","split_count","merge_count")},
                             "dominant_error_manual":"unreviewed"})
    panels=[]
    for item in chosen:
        with Image.open(root/item["file_name"]) as source:
            gray=np.asarray(source.convert("L"))
        pred=instances_from_probability(np.load(cache_dir/(item["physical_id"]+".npy")),config["threshold"],config["min_area"],config.get("low_threshold"))
        gt=rasterize_pass(item["passes"][0],item["height"],item["width"])
        panel=Image.new("RGB",(1152,416),"white")
        for column,view in enumerate((Image.fromarray(gray).convert("RGB"),overlay(gray,gt),overlay(gray,pred))):
            panel.paste(view.resize((384,384)),(column*384,32))
        label=f"{item['physical_id']} | PQ={scores['per_image'].get(item['physical_id'],{}).get('pq',0):.3f} | raw / GT pass {item['passes'][0]['annotation_pass_id']} / prediction"
        ImageDraw.Draw(panel).text((6,8),label,fill="black")
        panel.save(output_dir/(item["physical_id"]+".png"))
        panels.append(panel)
    for start in range(0,len(panels),5):
        sheet=Image.new("RGB",(1152,416*len(panels[start:start+5])),"white")
        for row,panel in enumerate(panels[start:start+5]): sheet.paste(panel,(0,row*416))
        sheet.save(output_dir/f"review-{start//5+1:02d}.jpg",quality=90)
    print(json.dumps({"pq":scores["pq"],"sq":scores["sq"],"rq":scores["rq"],"review_images":len(chosen)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("manifest","splits","cache-dir","config","output-dir"):
        parser.add_argument("--"+field,required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--fold",type=int,default=0)
    analyze(**vars(parser.parse_args()))
