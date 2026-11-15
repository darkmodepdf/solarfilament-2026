"""Export inference-only weights and immutable provenance from a completed run."""
import argparse
import json
from pathlib import Path
import hashlib
import torch


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def export(checkpoint,inference,submission,destination):
    checkpoint,inference,submission,destination=map(Path,(checkpoint,inference,submission,destination))
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    evidence=json.loads(inference.read_text(encoding="utf-8"))
    if evidence["checkpoint_sha256"]!=sha256(checkpoint) or evidence["submission_sha256"]!=sha256(submission):
        raise ValueError("Inference receipt does not match the checkpoint and submission")
    config=evidence["config"]
    destination.mkdir(parents=True,exist_ok=True)
    inference={"model":state["model"],"config":config,"global_step":state["global_step"],
               "data_hash":state["data_hash"],"split_hash":state["split_hash"],"versions":state["versions"]}
    weights=destination/"final.pt"
    torch.save(inference,weights)
    # GitHub's ordinary per-file limit is100MiB; full optimizer checkpoints stay in durable run output.
    if weights.stat().st_size>=100*1024**2:
        raise ValueError("Inference weights exceed GitHub's ordinary-file limit; use a verified repository storage method")
    provenance={"checkpoint_sha256":sha256(checkpoint),"weights_sha256":sha256(weights),
                "submission_sha256":sha256(submission),"data_hash":state["data_hash"],
                "split_hash":state["split_hash"],"global_step":state["global_step"],
                "config":config,"versions":state["versions"],"weight_bytes":weights.stat().st_size}
    (destination/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","inference","submission"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--destination",default="weights")
    export(**vars(parser.parse_args()))
