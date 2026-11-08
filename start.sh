#!/usr/bin/env bash
# One-click launcher for the TrialMatch-LLM browser app (Linux/macOS).
set -e
cd "$(dirname "$0")"
if [ -z "$TM_ADAPTER" ]; then
  if [ -d runs/qwen15_lora/best_adapter ]; then TM_ADAPTER=runs/qwen15_lora/best_adapter
  elif [ -f adapter.txt ]; then TM_ADAPTER=$(head -n1 adapter.txt); fi
fi
export TM_ADAPTER
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  if command -v nvidia-smi >/dev/null; then pip install torch; else pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
echo "[run] adapter = $TM_ADAPTER ; app at http://127.0.0.1:${TM_PORT:-7860}"
python app/app.py
