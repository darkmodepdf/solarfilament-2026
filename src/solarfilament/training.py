"""Budgeted single-GPU/DDP training with complete restartable checkpoints."""
import contextlib
import importlib.metadata
import math
import os
import random
import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .common import write_json, sha256_file
from .data import manifest_digest
from .dataset import CropDataset
from .evaluation import evaluate_model, validation_images
from .model import build_model, segmentation_loss
from .splits import validate_splits


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, payload):
    path=Path(path)
    temporary=path.with_suffix(".tmp")
    torch.save(payload,temporary)
    os.replace(temporary,path)


def remove_run_cache(path, run_root):
    path,root=Path(path).resolve(),Path(run_root).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Refusing cache cleanup outside run directory")
    if path.exists():
        shutil.rmtree(path)


def train(manifest,splits,config,output_dir,resume=None,max_steps=None,session_start=None):
    validate_splits(manifest,splits)
    started=float(session_start if session_start is not None else time.time())
    rank,world=int(os.environ.get("RANK",0)),int(os.environ.get("WORLD_SIZE",1))
    local_rank=int(os.environ.get("LOCAL_RANK",0))
    cuda=torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local_rank)
    device=torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if world>1:
        if not cuda:
            raise ValueError("DDP training requires CUDA in this pipeline")
        dist.init_process_group("nccl",timeout=timedelta(hours=1))
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    seed=int(config["seed"])
    random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank)
    if cuda:
        torch.cuda.manual_seed_all(seed+rank)
    effective=int(config["effective_batch_size"])
    batch=int(config["batch_size"])
    if effective%(batch*world):
        raise ValueError("Effective batch must be divisible by per-device batch times world size")
    accumulation=effective//(batch*world)
    if accumulation<1:
        raise ValueError("Per-step batch exceeds effective batch")
    data_hash=manifest_digest(manifest)
    if data_hash != manifest["manifest_hash"]:
        raise ValueError("Manifest identity mismatch")
    train_images=[item for item in manifest["images"] if splits["assignments"][item["physical_id"]]!=config["fold"]]
    val_images=validation_images(manifest,splits,config["fold"])
    if config.get("validation_limit"):
        val_images=val_images[:int(config["validation_limit"])]
    model=build_model(config,pretrained=resume is None).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    schedule_steps=int(config["max_steps"])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:0.5*(1+math.cos(math.pi*min(step,schedule_steps)/schedule_steps)))
    scaler=torch.amp.GradScaler("cuda",enabled=cuda and config.get("amp",True))
    step,best_pq=0,-1.0
    if resume:
        state=torch.load(resume,map_location="cpu",weights_only=False)
        if state["data_hash"]!=data_hash or state["split_hash"]!=splits["split_hash"]:
            raise ValueError("Resume data/split identity mismatch")
        for key in ("encoder","crop_size","batch_size","effective_batch_size","learning_rate","weight_decay","fold","normalization","annotation_mode","seed","max_steps","amp","foreground_probability"):
            if state["config"].get(key)!=config.get(key):
                raise ValueError(f"Resume config mismatch: {key}")
        if state["world_size"]!=world:
            raise ValueError("Exact recovery requires the original GPU count")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step,best_pq=state["global_step"],state["best_pq"]
        restore_rng(state["rng_states"][rank])
        if rank==0:
            print(f"RESUMED step={step} lr={optimizer.param_groups[0]['lr']:.9g} data_position={state['data_position']}",flush=True)
    if world>1:
        model=DistributedDataParallel(model,device_ids=[local_rank])
    unwrapped=model.module if world>1 else model
    stop_step=min(schedule_steps,int(max_steps) if max_steps is not None else schedule_steps)
    dataset=CropDataset(manifest,train_images,config,schedule_steps*effective)
    # Unique global sample index, disjoint ranks, stable across restart and worker count.
    indices=range(step*effective+rank,schedule_steps*effective,world)
    loader=DataLoader(dataset,batch_size=batch,sampler=indices,num_workers=int(config.get("workers",2)),
                      pin_memory=cuda,persistent_workers=False,drop_last=True,
                      generator=torch.Generator().manual_seed(seed+rank+100000))
    batches=iter(loader)
    versions={name:importlib.metadata.version(name) for name in ("torch","torchvision","numpy","segmentation-models-pytorch","pycocotools")}
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit="uncommitted-or-packaged-source"
    if rank==0:
        write_json(output_dir/"run.json",{"config":config,"versions":versions,"data_hash":data_hash,"split_hash":splits["split_hash"],
            "commit":commit,"world_size":world,"session_start":started,"device":str(device),
            "gpu":torch.cuda.get_device_name(device) if cuda else "CPU", "initialization":config.get("encoder_weights")})
    last_save=time.monotonic()
    save_seconds=0.0
    validation_seconds=0.0
    step_seconds=[]

    def save(name):
        nonlocal save_seconds,last_save
        begin=time.monotonic()
        states=[None]*world if rank==0 else None
        if world>1:
            dist.gather_object(rng_state(),states,dst=0)
        else:
            states=[rng_state()]
        if rank==0:
            atomic_checkpoint(output_dir/name,{"model":unwrapped.state_dict(),"optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"rng_states":states,"global_step":step,
                "data_position":step*effective,"best_pq":best_pq,"config":config,"data_hash":data_hash,
                "split_hash":splits["split_hash"],"versions":versions,"world_size":world,"commit":commit})
        if world>1:
            dist.barrier()
        save_seconds=time.monotonic()-begin
        last_save=time.monotonic()

    while step<stop_step:
        # Reserve validation plus two saves on top of the user-configured export reserve.
        reserve=max(float(config["reserve_minutes"])*60,validation_seconds+2*save_seconds+300)
        exhausted=time.time()-started >= float(config["total_hours"])*3600-reserve
        flag=torch.tensor(int(exhausted),device=device)
        if world>1:
            dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if flag.item():
            if rank==0: print("BUDGET_STOP: export reserve reached",flush=True)
            break
        begin=time.monotonic()
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum=0.0
        for microstep in range(accumulation):
            images,targets=next(batches)
            images,targets=images.to(device,non_blocking=True),targets.to(device,non_blocking=True)
            sync=model.no_sync() if world>1 and microstep<accumulation-1 else contextlib.nullcontext()
            with sync:
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=cuda and config.get("amp",True)):
                    loss=segmentation_loss(model(images),targets)/accumulation
                scaler.scale(loss).backward()
                loss_sum+=float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        old_scale=scaler.get_scale()
        scaler.step(optimizer);scaler.update()
        if scaler.get_scale()>=old_scale:
            scheduler.step()
        step+=1
        if cuda: torch.cuda.synchronize(device)
        step_seconds.append(max(time.monotonic()-begin,1e-6))
        if rank==0 and (step%20==0 or step<=3):
            stats={"step":step,"loss":loss_sum,"lr":optimizer.param_groups[0]["lr"],"seconds":step_seconds[-1],
                   "examples_per_second":effective/step_seconds[-1],"peak_vram_bytes":torch.cuda.max_memory_allocated(device) if cuda else 0}
            print(stats,flush=True)
            with (output_dir/"training.jsonl").open("a",encoding="utf-8") as handle:
                import json
                handle.write(json.dumps(stats)+"\n")
        validate=(step%int(config["validation_every"])==0 or step==stop_step) and not config.get("skip_validation",False)
        if validate:
            begin=time.monotonic()
            improved=False
            if rank==0:
                temporary=output_dir/"validation_pending"
                remove_run_cache(temporary,output_dir)
                metrics=evaluate_model(unwrapped,manifest,val_images,config,device,temporary)
                write_json(output_dir/f"metrics-step-{step}.json",metrics)
                print(f"VALIDATION step={step} PQ={metrics['pq']:.6f} SQ={metrics['sq']:.6f} RQ={metrics['rq']:.6f}",flush=True)
                improved=metrics["pq"]>best_pq
                if improved:
                    best_pq=metrics["pq"]
                    remove_run_cache(output_dir/"validation",output_dir)
                    temporary.rename(output_dir/"validation")
                    write_json(output_dir/"best_metrics.json",{"step":step,**metrics})
                else:
                    remove_run_cache(temporary,output_dir)
            if world>1:
                state=[best_pq,improved]
                dist.broadcast_object_list(state,src=0)
                best_pq,improved=state
            validation_seconds=time.monotonic()-begin
            if improved:
                save("best.pt")
                if rank==0:
                    # Written last: interrupted publication fails closed at tuning.
                    write_json(output_dir/"validation/provenance.json",{
                        "global_step":step,"data_hash":data_hash,"split_hash":splits["split_hash"],
                        "config":config,"checkpoint_sha256":sha256_file(output_dir/"best.pt")})
        checkpoint_due=time.monotonic()-last_save>=float(config["checkpoint_minutes"])*60
        flag=torch.tensor(int(checkpoint_due),device=device)
        if world>1: dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if flag.item(): save("last.pt")
    save("last.pt")
    if rank==0:
        write_json(output_dir/"completion.json",{"global_step":step,"best_pq":best_pq,"elapsed_seconds":time.time()-started,
            "mean_step_seconds":float(np.mean(step_seconds)) if step_seconds else None,"validation_seconds":validation_seconds,
            "save_seconds":save_seconds,"last_checkpoint_sha256":sha256_file(output_dir/"last.pt"),
            "gpu_hours":world*(time.time()-started)/3600 if cuda else 0})
    if world>1: dist.destroy_process_group()
    return step
