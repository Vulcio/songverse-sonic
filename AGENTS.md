# Instructions for coding agents

Read [CLAUDE.md](CLAUDE.md) before changing anything: it is this repository's rulebook for every
agent, whatever its name. In short:

- **Green before pushing:** `python -m pytest tests`, `bash -n` on the scripts in `ct/`, `install/`
  and `docker/`, and `docker build .` when the image changes. Never push with a test failing; watch CI.
- **Two ways to deploy, one app:** the Docker image and the Proxmox LXC run the same `app/` and
  `DEMUX_*` settings; change both together.
- **The API is a contract:** document every change in `API.md` and bump the version in `app/VERSION`,
  `API.md` and `app/main.py` together. Add fields; don't repurpose them.
- **Songverse is the main client:** a change it should use gets a Songverse issue, so its client
  (`apps/api/src/stem-separation/`) and its stand-in (`e2e/lib/fake-demucs.mjs`) follow.
- **An issue first, one commit per change,** with a message that says what and why.
- **Say only what you verified,** including on what hardware.
