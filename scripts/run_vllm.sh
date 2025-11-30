#!/usr/bin/env bash
set -euo pipefail

# Launch vLLM on the remote box via SSH with a configurable port.

# bash scripts/run_vllm.sh --port 8002 --gpu_id 0

PORT="8002"
GPU_ID=""
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
    --gpu_id)
      GPU_ID="${2:-}"
      if [[ -z "$GPU_ID" ]]; then
        echo "Error: --gpu_id requires a value" >&2
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
# 注意：read -d '' 在 heredoc 结束时返回 1，因此需容忍非零返回。
read -r -d '' REMOTE_SCRIPT <<'EOF' || true
set -euo pipefail

cd workspace/vllm

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

# Proxy for huggingface download (remote host runs clash at 127.0.0.1:66535)
export http_proxy="http://127.0.0.1:65535"
export https_proxy="http://127.0.0.1:65535"

GPU_ID="__GPU_ID__"
if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

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
REMOTE_SCRIPT="${REMOTE_SCRIPT//__GPU_ID__/$GPU_ID}"

ssh -t "$REMOTE_HOST" "bash -lc $(printf '%q' "$REMOTE_SCRIPT")"
