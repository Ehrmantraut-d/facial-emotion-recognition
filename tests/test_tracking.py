from src.tracking import new_run, save_json

def test_runs_are_unique_and_json_saved(tmp_path, monkeypatch):
    import src.tracking as tracking
    monkeypatch.setattr(tracking,"RUNS_DIR",tmp_path)
    a=tracking.new_run("quick"); b=tracking.new_run("quick")
    assert a != b
    tracking.save_json(a/"metrics.json",{"ok":True})
    assert (a/"metrics.json").exists() and not (b/"metrics.json").exists()
