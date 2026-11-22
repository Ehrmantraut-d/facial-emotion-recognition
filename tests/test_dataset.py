from types import SimpleNamespace
import pytest
from src.data.dataset import acquire_dataset

def test_download_failure_preserves_cli_message(tmp_path, monkeypatch):
    import src.data.dataset as dataset
    monkeypatch.setattr(dataset,"RAW_DIR",tmp_path)
    monkeypatch.setattr(dataset.shutil,"which",lambda _:"kaggle")
    monkeypatch.setattr(dataset.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=1,stderr="",stdout="Kaggle credentials missing"))
    with pytest.raises(RuntimeError,match="Kaggle credentials missing"):
        acquire_dataset()
