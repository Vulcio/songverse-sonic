#!/bin/sh
# Starts Songverse Sonic in its container: the device from the image (cpu, or cuda when the GPU is
# really there), the models in DEMUX_PRELOAD_MODELS downloaded once into the volume, then the API.
set -eu

# The image's device: "cpu", or "cu130" for the CUDA image - the API calls it "cuda", and only uses
# it when PyTorch sees a GPU (else the CPU, as the LXC installer falls back).
case "${DEMUX_DEVICE:-cpu}" in
  cu*) DEMUX_DEVICE=cuda ;;
esac
if [ "$DEMUX_DEVICE" = "cuda" ] && ! python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "songverse-sonic: no usable GPU (is the NVIDIA container toolkit set up?); using the CPU" >&2
  DEMUX_DEVICE=cpu
fi
export DEMUX_DEVICE
export DEMUX_ALLOWED_DEVICES="${DEMUX_ALLOWED_DEVICES:-$([ "$DEMUX_DEVICE" = cpu ] && echo cpu || echo "cpu,$DEMUX_DEVICE")}"

for model in ${DEMUX_PRELOAD_MODELS:-}; do
  python - "$model" <<'PY' || echo "songverse-sonic: couldn't download $model; Demucs will retry on first use" >&2
import sys
from demucs.pretrained import get_model
get_model(sys.argv[1])
print(f"songverse-sonic: model {sys.argv[1]} ready")
PY
done

set -- main:app --host "${DEMUX_BIND:-0.0.0.0}" --port "${DEMUX_PORT:-8000}"
if [ "${DEMUX_TRUST_PROXY_HEADERS:-false}" = "true" ]; then
  set -- "$@" --proxy-headers --forwarded-allow-ips "${DEMUX_FORWARDED_ALLOW_IPS:-127.0.0.1}"
else
  set -- "$@" --no-proxy-headers
fi
exec uvicorn "$@"
