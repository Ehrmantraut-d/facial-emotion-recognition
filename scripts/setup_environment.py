"""Create a project-local isolated venv; never modifies global/base Python."""
import subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
venv=root/".venv"
if not venv.exists(): subprocess.run([sys.executable,"-m","venv",str(venv)],check=True)
python=venv/"Scripts"/"python.exe" if sys.platform=="win32" else venv/"bin"/"python"
subprocess.run([str(python),"-m","pip","install","--upgrade","pip"],check=True)
subprocess.run([str(python),"-m","pip","install","-r",str(root/"requirements.txt")],check=True)
subprocess.run([str(python),"-m","ipykernel","install","--user","--name","facial_emotion_ai_venv","--display-name","Python (facial_emotion_ai, venv)"],check=True)
print(f"Environment ready: {python}")
