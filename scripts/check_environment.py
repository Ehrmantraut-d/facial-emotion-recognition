"""Capture local tool, dependency, GUI, and webcam diagnostics."""
from __future__ import annotations
import argparse, importlib.util, json, os, platform, sys, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import ENVIRONMENT_DIR

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-camera",action="store_true",help="Skip webcam probing (useful in CI or headless systems).")
    args=parser.parse_args()
    names = {"numpy":"numpy", "pandas":"pandas", "scikit_learn":"sklearn", "opencv":"cv2", "mediapipe":"mediapipe", "pillow":"PIL", "matplotlib":"matplotlib", "joblib":"joblib", "jupyter":"jupyter", "pytest":"pytest", "tkinter":"tkinter"}
    checks = {label: bool(importlib.util.find_spec(module)) for label,module in names.items()}
    camera = {"checked": False, "available_indices": []}
    if checks["opencv"] and not args.skip_camera:
        import cv2
        camera["checked"] = True
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None and frame.size: camera["available_indices"].append(i)
            cap.release()
    home=Path.home()
    kaggle_config = home/".kaggle"/"kaggle.json"
    conda_candidates=(home/"anaconda3"/"Scripts"/"conda.exe", home/"anaconda3"/"bin"/"conda",
                      home/"miniconda3"/"Scripts"/"conda.exe", home/"miniconda3"/"bin"/"conda",
                      home/"miniforge3"/"Scripts"/"conda.exe", home/"miniforge3"/"bin"/"conda",
                      home/"mambaforge"/"Scripts"/"conda.exe", home/"mambaforge"/"bin"/"conda")
    conda_available=bool(os.environ.get("CONDA_PREFIX")) or shutil.which("conda") is not None or any(p.is_file() for p in conda_candidates)
    report = {"python":sys.version, "python_executable":sys.executable, "platform":platform.platform(),
              "conda_available": conda_available,
              "kaggle_cli_available": shutil.which("kaggle") is not None or importlib.util.find_spec("kaggle") is not None,
              "kaggle_credentials_configured": kaggle_config.is_file() or bool(os.environ.get("KAGGLE_API_TOKEN")) or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")),
              "dependencies":checks,"webcam":camera,"notes":[]}
    if sys.version_info[:2] != (3,11): report["notes"].append("Reproducible target is Python 3.11; active interpreter differs.")
    if not report["conda_available"]: report["notes"].append("Conda was not detected; create the environment on a machine with Conda.")
    ENVIRONMENT_DIR.mkdir(parents=True, exist_ok=True)
    path=ENVIRONMENT_DIR/"environment_check.json"; path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); print(f"Saved {path}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
