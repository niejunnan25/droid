#!/usr/bin/env bash
set -euo pipefail

# Launch vLLM on the remote box via SSH with a configurable port.

PORT="8002"
REMOTE_HOST="${REMOTE_HOST:-njn@162.105.195.74}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      if [[ -z "$PORT" ]]; then
        echo "Error: --port requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --host)
      REMOTE_HOST="${2:-}"
      if [[ -z "$REMOTE_HOST" ]]; then
        echo "Error: --host requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# Remote script runs conda env, exports CUDA libs, and starts vLLM.
read -r -d '' REMOTE_SCRIPT <<'EOF'
set -euo pipefail

cd worksapce/vllm

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Fallback if conda command is not on PATH.
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  eval "$(conda shell.bash hook)"
else
  echo "conda not found on remote host." >&2
  exit 1
fi

conda activate vllm

export LIBRARY_PATH="$HOME/libcuda_fix:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$HOME/libcuda_fix:${LD_LIBRARY_PATH:-}"

python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-VL-4B-Instruct" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max_model_len 4096 \
  --host 0.0.0.0 \
  --port "__PORT__"
EOF

# Inject the chosen port into the remote script.
REMOTE_SCRIPT="${REMOTE_SCRIPT//__PORT__/$PORT}"

ssh -t "$REMOTE_HOST" "bash -lc $(printf '%q' "$REMOTE_SCRIPT")"
