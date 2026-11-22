import numpy as np
from src.data.analysis import generate_eda

def test_eda_creates_real_artifact_files_in_requested_directory(tmp_path):
    rng=np.random.default_rng(3)
    data={"train":(rng.random((14,2304),dtype=np.float32),np.arange(14)%7),
          "val":(rng.random((7,2304),dtype=np.float32),np.arange(7)),
          "test":(rng.random((7,2304),dtype=np.float32),np.arange(7))}
    out=generate_eda(data,{"invalid_count":2},tmp_path/"eda")
    assert (out/"statistics.json").is_file()
    assert (out/"class_distribution.csv").is_file()
    assert all((out/"graphs"/name).is_file() for name in
               ("class_distribution.png","pixel_distribution.png","class_examples.png"))
