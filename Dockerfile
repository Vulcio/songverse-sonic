# Songverse Sonic: stem separation (Demucs) and music analysis, as a container.
# The same app/ as the Proxmox LXC; only how it's installed differs.
#   docker build -t songverse-sonic .                                  # CPU
#   docker build -t songverse-sonic:cuda --build-arg TORCH=cu130 .     # NVIDIA GPU
# The base image; a mirror of it where Docker Hub is out of reach.
ARG BASE=python:3.12-slim-bookworm
FROM ${BASE}

# PyTorch's wheel index: cpu, or a CUDA build (cu130) whose wheels carry their own CUDA runtime.
ARG TORCH=cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 tzdata curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/songverse-sonic
COPY app/requirements.txt ./
RUN pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/${TORCH}" torch torchaudio \
 && pip install --no-cache-dir -r requirements.txt

COPY app/ ./
COPY docker/entrypoint.sh /usr/local/bin/sonic-entrypoint
RUN chmod 0755 /usr/local/bin/sonic-entrypoint \
 && useradd --system --uid 10001 --home /var/lib/demucs-api --shell /usr/sbin/nologin sonic \
 && mkdir -p /var/lib/demucs-api/jobs /var/lib/demucs-api/torch \
 && chown -R sonic:sonic /var/lib/demucs-api

# Jobs, their database and the downloaded models live in the volume.
ENV DEMUX_DATA_DIR=/var/lib/demucs-api \
    TORCH_HOME=/var/lib/demucs-api/torch \
    DEMUX_BIND=0.0.0.0 \
    DEMUX_PORT=8000 \
    DEMUX_DEVICE=${TORCH} \
    PYTHONUNBUFFERED=1
VOLUME /var/lib/demucs-api
USER sonic
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fsS "http://127.0.0.1:${DEMUX_PORT}/api/v1/health" >/dev/null || exit 1
ENTRYPOINT ["sonic-entrypoint"]
