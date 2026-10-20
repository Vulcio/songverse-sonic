# Songverse Sonic

Stem separation (**Demucs 4.1.0**) and music analysis behind a FastAPI REST API: the server
[Songverse](https://github.com/bricedupuy/Songverse) sends recordings to, and anyone else can call.
(It was called *demucs-proxmox* until 1.6; its API, settings and the LXC's names are unchanged.)

Two ways to deploy it, the same app either way:

- **[Docker](#deploy-with-docker)**: an image for the CPU and one for NVIDIA GPUs, on any server.
- **[Proxmox LXC](#install-on-proxmox)**: a Community-Scripts-style installer for a dedicated **Debian 13 LXC**.

A church, a team or an organisation can run its own and connect it in Songverse's settings.

It is designed for an orchestrated workflow such as:

```text
URL -> File -> Demucs fast pass -> R2
                     |
                     +-> scheduled nightly HQ pass -> R2 -> signed webhook
```

## Features

- Debian 13 unprivileged LXC
- Default / Advanced guided creation
- Configurable CT ID, vCPU, RAM, swap, storage, bridge, VLAN and IP addressing
- Demucs 4.1.0
- Guided model choice with optional model pre-download
- CPU backend by default
- Optional/experimental NVIDIA CUDA and Intel XPU installation paths with runtime verification and CPU fallback
- FastAPI REST API + built-in OpenAPI UI
- API-key authentication
- Persistent SQLite job and webhook state
- One Demucs separation at a time
- Fast-job priority over scheduled HQ work
- `htdemucs` now + `htdemucs_ft` nightly upgrade workflow
- Signed webhooks with retry persistence
- Direct file upload
- Remote HTTP/HTTPS audio input
- Optional direct Cloudflare R2/S3 upload
- Music analysis (tempo, first downbeat, meter, key, sections) on every job after the fast pass
- Automatic job cleanup
- Reverse-proxy-aware public URLs and trusted forwarding configuration
- Docker images (CPU and CUDA) with a health check, or a Proxmox LXC with a systemd service and an `update` command

Full endpoint documentation is in **[API.md](API.md)**.

## Repository layout

```text
Dockerfile                       The Docker image (CPU, or CUDA with TORCH=cu130)
docker/entrypoint.sh             The image's start-up: device, model download, uvicorn
docker-compose.yml, .env.example Running the image with Docker Compose
ct/demucs-api.sh                 Proxmox LXC creator + in-LXC updater
install/demucs-api-install.sh    First-install script
app/main.py                      REST API, scheduler, worker and webhooks
app/analysis.py                  Music analysis (tempo, key, sections)
tests/                           pytest suite
app/requirements.txt             Python dependencies
app/VERSION                      API version
json/demucs-api.json             Community Scripts-style metadata
API.md                           API and reverse-proxy documentation
```

## Deploy with Docker

Images are published to GitHub's registry from every push to `main`:

| Image | For |
| --- | --- |
| `ghcr.io/bricedupuy/songverse-sonic:latest` | CPU (any x86-64 server) |
| `ghcr.io/bricedupuy/songverse-sonic:cuda` | NVIDIA GPU, with the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |

Each is also tagged with the API's version (`1.6.0`, `1.6.0-cuda`) and the commit (`sha-…`).

```bash
docker run -d --name sonic -p 8000:8000 \
  -e DEMUX_API_KEY="$(openssl rand -hex 32)" \
  -v sonic-data:/var/lib/demucs-api \
  ghcr.io/bricedupuy/songverse-sonic:latest
curl http://localhost:8000/api/v1/health
```

Or with Docker Compose: copy `docker-compose.yml` and `.env.example` (as `.env`), set
`DEMUX_API_KEY`, then `docker compose up -d` (CPU) or `docker compose --profile gpu up -d` (GPU).

- **Settings** are the same `DEMUX_*` variables as the LXC's (see [Configuration](#configuration)).
  The key can come from a file instead (`DEMUX_API_KEY_FILE`, for Docker secrets).
- **Models** download into the volume on first use; `DEMUX_PRELOAD_MODELS=htdemucs htdemucs_ft`
  fetches them when the container starts instead.
- **Data** (jobs, their results, the models) lives in the `/var/lib/demucs-api` volume; the
  container runs as a non-root user (uid 10001).
- **The GPU image** uses the GPU when PyTorch sees one, else says so in its log and uses the CPU.
- **HTTPS:** put it behind a reverse proxy (see [Reverse proxy](#reverse-proxy)), with
  `DEMUX_PUBLIC_BASE_URL`, `DEMUX_TRUST_PROXY_HEADERS=true` and the proxy's address in
  `DEMUX_FORWARDED_ALLOW_IPS`.
- **Updating:** `docker pull` the image and recreate the container; the volume keeps the jobs.

To build it yourself: `docker build -t songverse-sonic .`, or `--build-arg TORCH=cu130` for CUDA
(`BASE=` picks another Python base image, such as a registry mirror).

## Install on Proxmox

Run on the Proxmox VE host as root:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/bricedupuy/songverse-sonic/main/ct/demucs-api.sh)"
```

Default installation:

```text
Debian 13
Unprivileged LXC
6 vCPU
8 GB RAM
2 GB swap
24 GB disk
DHCP / vmbr0
CPU compute
htdemucs default model
htdemucs pre-downloaded
htdemucs_ft HQ model
HQ window 02:00-07:00 Europe/Paris
```

Advanced setup adds model selection, model preloading, compute backend, PyTorch wheel index and HQ scheduling settings in addition to the container/network options.

## GPU support

The guided installer offers:

```text
CPU
Auto-detect
NVIDIA CUDA
Intel XPU
```

CPU is the production-safe default.

GPU passthrough in LXC depends on the Proxmox host, driver versions and device permissions. The installer performs basic device passthrough and then verifies the accelerator **inside the container** with PyTorch. If CUDA/XPU is not actually usable, the service falls back to CPU instead of leaving a broken API.

`Auto-detect` currently selects CUDA when `/dev/nvidia0` exists; otherwise it selects CPU. It deliberately does not auto-select Intel XPU because older Intel iGPUs such as UHD 630 expose `/dev/dri` but are not suitable PyTorch XPU targets.

## Fast + nightly HQ processing

A job can ask for an immediate fast result and a scheduled HQ upgrade:

```json
{
  "source_url": "http://path.to/song.wav",
  "fast": {
    "model": "htdemucs"
  },
  "hq": {
    "enabled": true,
    "model": "htdemucs_ft"
  },
  "callback_url": "https://app.example.com/hooks/demucs",
  "callback_secret": "SHARED_SECRET"
}
```

The worker processes `htdemucs` immediately. The original source is retained and the `htdemucs_ft` pass becomes eligible during the configured HQ window.

The external service can either poll `GET /api/v1/jobs/{id}` or receive signed events such as:

```text
fast.completed
hq.started
hq.completed
hq.failed
```

## Music analysis

As the last step of the fast pass, the server analyses the recording and adds an optional `analysis` block to the job and to webhook payloads (see [API.md](API.md#music-analysis)). It runs before the fast stage is reported completed, uses the separated stems, has a time budget, and never fails a job.

Settings in `/etc/demucs-api/demucs-api.env`:

```ini
ANALYSIS_ENABLED=true            # false turns it off
ANALYSIS_BACKEND=librosa         # or allin1 (opt-in, see below)
ANALYSIS_TIMEOUT_SECONDS=60      # budget; remaining steps are skipped once exceeded
```

### Libraries and licences

This server may be used commercially and its client Songverse is AGPL-3.0, so only permissively licensed code without non-commercial model files is used by default.

| Component | Licence | Used |
| --- | --- | --- |
| librosa | ISC | **Default backend**: beat tracking, chroma, segmentation. No trained model files. |
| soundfile / numpy | BSD-3 | Audio I/O and maths. |
| Key detection | own code | Krumhansl-Schmuckler profile correlation on librosa chroma (published algorithm, no model). |
| madmom | code BSD-3, **model files CC BY-NC-SA** | **Not used** (non-commercial models). |
| Essentia | AGPL-3.0 | **Not used.** |
| All-In-One (`allin1`) | code MIT; **weights' licence is not stated** in the repository; trained on the Harmonix Set; depends on madmom and NATTEN | **Opt-in only**, off by default. |

How the default works: tempo from a harmonic comb over the autocorrelation of the drum accents (kick and snare-body flux, so hi-hats do not look like double time), with a log-normal prior toward ~92 bpm to choose the octave (double/half time is not decidable from periodicity alone: songs counted faster than ~140 are folded to half, and the confidence drops near the boundaries); beats tracked on drums plus the mix's low band so a drumless intro still has a grid; the meter (3 or 4 beats per bar) and downbeat phase from kick accents plus chord changes (chroma novelty) with a prior toward 4/4, and `first_beat` as the start of bar 1, extended back over the intro to where the music starts;
`beats`/`downbeats` are the tracked beats themselves (sub-frame refined, trailing ring-out dropped), and `tempo.bpm` is their median interval so the fields always agree; `intro_free` compares how periodic the onset envelope is before vs after the first sustained downbeat, and is only answered for intros of 8+ beats (otherwise left out); key from the bass+other stems (or `no_vocals`); sections from clustering chroma/MFCC features of the mix, snapped to bars, with labels guessed from repetition, loudness and whether the vocal stem is active. These are heuristics, not trained models: sections in particular are rough.

All-In-One would give better downbeats and functional section labels. Because the licence of its weights is unspecified and it needs madmom and NATTEN (heavy to install), it is not installed or enabled by default. If you accept that after checking the licence yourself, install `allin1` in the venv and set `ANALYSIS_BACKEND=allin1`; the librosa backend still fills in anything it does not return, and any failure falls back to librosa. That path reuses the fast stems only when the fast model is `htdemucs` with four wav stems (otherwise it separates again, which takes minutes); it has not been exercised in this repository's tests.

## Cloudflare R2

R2 is optional. When enabled, fast and HQ results are uploaded separately:

```text
demucs/JOB_ID/fast/...
demucs/JOB_ID/hq/...
```

Configure `/etc/demucs-api/demucs-api.env`:

```ini
DEMUX_R2_ENABLED=true
DEMUX_R2_ENDPOINT=https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com
DEMUX_R2_ACCESS_KEY_ID=...
DEMUX_R2_SECRET_ACCESS_KEY=...
DEMUX_R2_BUCKET=your-bucket
DEMUX_R2_PREFIX=demucs
```

Use an R2 token limited to only the required bucket.

## Reverse proxy

Do not expose LXC port 8000 directly to the Internet. Put TLS/authentication/rate controls at your reverse proxy and keep the API key enabled.

When the proxy is configured, set:

```ini
DEMUX_PUBLIC_BASE_URL=https://demucs.example.com
DEMUX_TRUST_PROXY_HEADERS=true
DEMUX_FORWARDED_ALLOW_IPS=10.0.0.20
```

Use the actual IP of the reverse proxy for `DEMUX_FORWARDED_ALLOW_IPS`, not `*`, unless port 8000 is separately restricted to the proxy.

For large direct uploads, raise the proxy request-body limit (the API default is 1024 MB). Demucs execution itself is asynchronous, so you do not need a multi-hour reverse-proxy request timeout while separation runs.

See [API.md](API.md) for Nginx/Caddy examples and security notes.

## API key

```bash
pct exec <CTID> -- cat /etc/demucs-api/api-key
```

## Configuration

Main file:

```text
/etc/demucs-api/demucs-api.env
```

Notable options:

```ini
DEMUX_DEFAULT_MODEL=htdemucs
DEMUX_HQ_DEFAULT_MODEL=htdemucs_ft
DEMUX_DEVICE=cpu
DEMUX_ALLOWED_DEVICES=cpu
DEMUX_CPU_JOBS=6
DEMUX_MAX_UPLOAD_MB=1024
DEMUX_JOB_TTL_HOURS=48
DEMUX_TIMEZONE=Europe/Paris
DEMUX_HQ_START_HOUR=2
DEMUX_HQ_END_HOUR=7
DEMUX_PUBLIC_BASE_URL=
DEMUX_TRUST_PROXY_HEADERS=false
DEMUX_FORWARDED_ALLOW_IPS=127.0.0.1
DEMUX_ALLOW_PRIVATE_SOURCE_URLS=true
DEMUX_ALLOW_PRIVATE_CALLBACK_URLS=false
ANALYSIS_ENABLED=true
```

Also: `DEMUX_API_KEY_FILE` (the key read from a file, when `DEMUX_API_KEY` is empty) and
`DEMUX_DEMUCS_BIN` (the `demucs` command; by default the one beside the Python running the API).

Restart after editing:

```bash
systemctl restart demucs-api
```

## Update

Inside the LXC:

```bash
update
```

The updater pulls the latest application files and requirements from the configured GitHub repository and restarts the service.

## Useful commands

```bash
systemctl status demucs-api
journalctl -u demucs-api -f
demucs-api-info
update
```

Interactive API documentation after install:

```text
http://LXC_IP:8000/docs
```

## Security notes

- Keep the API key private.
- Prefer HTTPS at the reverse proxy.
- Firewall port 8000 to your reverse proxy and trusted internal services.
- Remote source URLs are allowed to reach private addresses by default. Treat this as an SSRF-sensitive capability and give API credentials only to trusted callers.
- Private webhook URLs are disabled by default.
- Configure a per-integration webhook secret and verify `X-Demucs-Signature` on the receiving server.

## Community Scripts compatibility

The project follows the Community Scripts shape and user experience (`ct/`, `install/`, `json/`, guided defaults/advanced options, and an in-container `update` command) but remains self-contained and does not depend on their repository at runtime.

## License

MIT

## Tests

```bash
pip install pytest librosa soundfile fastapi httpx boto3 python-multipart
python -m pytest tests
```
