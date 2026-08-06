#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_ROOT="/workspace/persistent/ComfyUI"
COMFY_PYTHON="/workspace/comfyui-rocm-env/bin/python"
LOG_DIR="$HERE/logs"
COMFY_LOG="$LOG_DIR/comfyui.log"
GENERATION_LOG="$LOG_DIR/generation.log"

mkdir -p "$LOG_DIR"
export FLUX_MEMORY_PROFILE="DynamicVRAM (10 GB headroom), CPU text encoder, CPU tiled VAE"

comfy_pid=""
cleanup() {
  if [[ -n "$comfy_pid" ]] && kill -0 "$comfy_pid" 2>/dev/null; then
    kill "$comfy_pid" 2>/dev/null || true
    wait "$comfy_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! curl --silent --fail --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null; then
  cd "$COMFY_ROOT"
  "$COMFY_PYTHON" main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --disable-auto-launch \
    --disable-all-custom-nodes \
    --disable-manager-ui \
    --enable-dynamic-vram \
    --vram-headroom 10 \
    --cpu-vae \
    --reserve-vram 6 \
    --disable-pinned-memory \
    --use-pytorch-cross-attention \
    --preview-method none \
    --verbose INFO "$COMFY_LOG" \
    --log-stdout &
  comfy_pid=$!

  for _ in $(seq 1 180); do
    if curl --silent --fail --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null; then
      break
    fi
    if ! kill -0 "$comfy_pid" 2>/dev/null; then
      echo "ComfyUI exited during startup; inspect $COMFY_LOG" >&2
      wait "$comfy_pid"
    fi
    sleep 1
  done
fi

if ! curl --silent --fail --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null; then
  echo "ComfyUI did not become ready; inspect $COMFY_LOG" >&2
  exit 1
fi

cd "$HERE"
if (($#)); then
  case_dirs=("$@")
else
  case_dirs=(
    delta-formation
    wave-interference
    mitosis
    endocytosis-exocytosis
  )
fi

skip_args=(--skip-existing)
if [[ "${FLUX_FORCE:-0}" == "1" ]]; then
  skip_args=()
fi

"$COMFY_PYTHON" run_flux2_demo.py \
  "${skip_args[@]}" \
  --log-file "$GENERATION_LOG" \
  "${case_dirs[@]}"
