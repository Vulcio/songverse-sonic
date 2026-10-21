# Songverse Sonic - notes for Claude

The stem-separation server that [Songverse](https://github.com/Vulcio/Oneverse) calls: a FastAPI
service running Demucs, deployed as a Docker image or in a Debian LXC on Proxmox, with an optional music analysis of each recording
(tempo, beats and downbeats, meter, key, sections). Songverse is its main client; it is a separate
project with its own users, so its API stays general rather than shaped around Songverse.

## Layout

- `app/main.py`: the REST API, the job queue and scheduler, the worker, webhooks, R2/S3 upload.
- `app/analysis.py`: the music analysis (librosa by default; `allin1` optional and off).
- `tests/`: the pytest suite. `synth.py` makes test audio, so no recordings live in the repository.
- `ct/` and `install/`: the Proxmox host script and the in-LXC installer and updater, in the
  Community Scripts style. `json/demucs-api.json` is their metadata.
- `Dockerfile`, `docker/entrypoint.sh`, `docker-compose.yml`, `.env.example`: the Docker images (CPU,
  and CUDA with `--build-arg TORCH=cu130`), published to `ghcr.io/bricedupuy/songverse-sonic` by
  `.github/workflows/ci.yml` from `main` and tags.
- `API.md`: the API's documentation, the contract its clients rely on.

Both ways to deploy run the same `app/` with the same `DEMUX_*` settings: a new setting works in both
(the LXC's env file, the image's environment and `.env.example`), and a new system package goes in
both the installer and the `Dockerfile`.

## Before every push

```sh
pip install pytest librosa soundfile fastapi httpx boto3 python-multipart
python -m pytest tests
bash -n ct/demucs-api.sh install/demucs-api-install.sh docker/entrypoint.sh
docker build -t songverse-sonic:test .   # when the Dockerfile, docker/ or requirements change
```

All tests must pass. The install scripts can't run outside Proxmox: check their syntax, change them
carefully, and say in the commit what you couldn't run. CI runs the tests, builds both images and
checks the CPU one starts and answers `/api/v1/health`: watch it after pushing.

## The API is a contract

- Every change to what a client sends or gets is written in `API.md` in the same change, and bumps
  the version in all three places it lives: `app/VERSION`, `API.md`'s "API version", and
  `FastAPI(..., version=...)` in `app/main.py`. New fields are a minor version; removing or changing
  one is a major version and needs a very good reason.
- Fields are added, never repurposed. Optional parts stay optional, and a client that ignores a new
  field keeps working. The analysis never fails a job: when unsure, a field is left out rather than
  guessed (see "Music analysis" in `API.md`).
- **Songverse** reads it in `apps/api/src/stem-separation/demucs-client.ts` (types) and `analysis.ts`
  (what it keeps), and stands in for this server in its e2e suites (`e2e/lib/fake-demucs.mjs`). A
  change here that Songverse should use gets a Songverse issue, so its client and fake follow.

## How to work

- **An issue first,** here or in Songverse, referenced by the commits (`Refs #n`, `Closes #n`).
- **One commit per change,** made with git, with a message that says what changed and why (or a pull
  request, as the analysis work was).
- **Tests with the change:** new behaviour gets a test in `tests/`; analysis changes are checked
  against synthetic audio with known tempo, beats and key (`tests/synth.py`).
- **Licences:** Demucs is MIT; check any new model's or library's licence before adding it, and keep
  anything with an unclear licence optional and off by default, as `allin1` is.
- **Say only what you verified:** what you ran, on what hardware (CPU, CUDA, XPU). GPU paths are
  experimental; don't claim they work without running them.
