import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from solarfilament.data import build_manifest
from solarfilament.splits import make_splits
from solarfilament import training


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.Dropout2d(0.3),torch.nn.Conv2d(4,1,1))

    def forward(self,x):
        return self.layers(x)


def test_interrupted_training_restores_optimizer_scheduler_rng_and_data(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(training,"build_model",lambda config,pretrained: TinyModel())
    image_dir=tmp_path/"images"
    image_dir.mkdir()
    rows=[];annotations=[]
    for i in range(6):
        stem=f"20110{i+1}01110000Ch"
        Image.fromarray(np.random.default_rng(i).integers(0,255,(64,64),dtype=np.uint8)).save(image_dir/(stem+".jpeg"))
        rows.append({"id":f"pass-{stem}","file_name":stem+".jpeg","width":64,"height":64})
        annotations.append({"id":str(i),"image_id":f"pass-{stem}","segmentation":[[4,4,20,4,20,20,4,20]]})
    source=tmp_path/"annotations.json"
    source.write_text(json.dumps({"images":rows,"annotations":annotations}),encoding="utf-8")
    manifest=build_manifest(source,image_dir)
    splits=make_splits(manifest,n_folds=2)
    config={"seed":2026,"encoder":"tiny","encoder_weights":None,"crop_size":32,"batch_size":2,
            "effective_batch_size":4,"workers":0,"learning_rate":0.001,"weight_decay":0.0001,
            "max_steps":4,"validation_every":4,"checkpoint_minutes":12,"total_hours":1,"reserve_minutes":1,
            "fold":0,"annotation_mode":"sampled_pass","normalization":"raw","amp":False,"skip_validation":True}
    training.train(manifest,splits,config,tmp_path/"uninterrupted")
    training.train(manifest,splits,config,tmp_path/"resumed",max_steps=2)
    training.train(manifest,splits,config,tmp_path/"resumed",resume=tmp_path/"resumed/last.pt")
    a=torch.load(tmp_path/"uninterrupted/last.pt",weights_only=False)
    b=torch.load(tmp_path/"resumed/last.pt",weights_only=False)
    assert a["global_step"]==b["global_step"]==4
    assert a["data_position"]==b["data_position"]==16
    assert a["scheduler"]==b["scheduler"]
    for key in a["model"]:
        assert torch.equal(a["model"][key],b["model"][key]),key
    for key,state in a["optimizer"]["state"].items():
        for field,value in state.items():
            assert torch.equal(value,b["optimizer"]["state"][key][field])
