import time
import numpy as np
import pytest
from solarfilament.common import write_json,sha256_file
from solarfilament.evaluation import tune_cached


def test_tuning_rejects_wrong_checkpoint_and_retains_only_complete_candidates(tmp_path):
    cache=tmp_path/"validation"
    cache.mkdir()
    checkpoint=tmp_path/"best.pt"
    checkpoint.write_bytes(b"test-only checkpoint identity")
    manifest={"manifest_hash":"data1","images":[{"physical_id":"one","height":16,"width":16,
        "passes":[{"annotation_pass_id":"a","annotations":[{"id":"1","segmentation":[[2,2,7,2,7,7,2,7]]}]}]}]}
    splits={"assignments":{"one":0},"split_hash":"split1"}
    np.save(cache/"one.npy",np.zeros((16,16),np.float32))
    write_json(cache/"provenance.json",{"data_hash":"data1","split_hash":"split1","checkpoint_sha256":"wrong"})
    with pytest.raises(ValueError,match="does not belong"):
        tune_cached(manifest,splits,cache,tmp_path/"selection.json")
    write_json(cache/"provenance.json",{"data_hash":"data1","split_hash":"split1","checkpoint_sha256":sha256_file(checkpoint)})
    with pytest.raises(TimeoutError,match="No complete"):
        tune_cached(manifest,splits,cache,tmp_path/"selection.json",deadline=time.time()-1)
    assert not (tmp_path/"selection.json").exists()
    chosen=tune_cached(manifest,splits,cache,tmp_path/"selection.json")
    assert chosen["threshold"] in (0.35,0.45,0.55,0.65)
