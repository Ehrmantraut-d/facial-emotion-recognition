#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CONDA="$(command -v conda || true)"

if [ -z "$CONDA" ]; then
  for candidate in \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/mambaforge/bin/conda" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda"
  do
    if [ -x "$candidate" ]; then
      CONDA="$candidate"
      break
    fi
  done
fi

if [ -n "$CONDA" ]; then
  exec "$CONDA" run --no-capture-output -n facial_emotion_ai python "$ROOT/main.py"
fi

if [ -x "$ROOT/.venv/bin/python" ]; then
  exec "$ROOT/.venv/bin/python" "$ROOT/main.py"
fi

printf '%s\n' "Conda environment 'facial_emotion_ai' was not found." \
  "Create it with: conda env create -f environment.yml" \
  "Or install a local venv with: python3 scripts/setup_environment.py" >&2
exit 1
