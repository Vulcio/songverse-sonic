# Songverse Sonic API

API version: **1.10.0**

The service provides asynchronous Demucs stem separation with an optional two-stage workflow:

1. run `htdemucs` immediately for a fast result;
2. keep the original source audio;
3. schedule `htdemucs_ft` for the configured HQ processing window;
4. notify the caller by signed webhook when either stage completes;
5. optionally upload each stage directly to Cloudflare R2 / any S3-compatible store.

The built-in OpenAPI/Swagger UI is also available at `/docs` and `/redoc`.

## Authentication

All endpoints except `GET /api/v1/health` require the API key:

```http
X-API-Key: YOUR_API_KEY
```

Retrieve the key inside the LXC with:

```bash
cat /etc/demucs-api/api-key
```

Do not put the API key in query parameters.

## Base URL

On a LAN:

```text
http://DEMUX_LXC_IP:8000
```

Behind a reverse proxy, set `DEMUX_PUBLIC_BASE_URL` to the externally visible origin, for example:

```ini
DEMUX_PUBLIC_BASE_URL=https://demucs.example.com
```

Returned `download_url` values will then be absolute URLs using that origin.

## Models

Supported model identifiers:

| Model | Typical use |
| --- | --- |
| `htdemucs` | Default fast/general model |
| `htdemucs_ft` | Fine-tuned, slower HQ model |
| `htdemucs_6s` | Six-source model |
| `hdemucs_mmi` | Hybrid Demucs model |
| `mdx` | MDX model |
| `mdx_extra` | MDX Extra |
| `mdx_q` | Quantized MDX |
| `mdx_extra_q` | Quantized MDX Extra |

The default installer pre-downloads `htdemucs`. Other models are downloaded by Demucs when first used and cached under `/var/cache/demucs-api/torch`.

## Compute devices

The API can expose one or more of:

```text
cpu
cuda
xpu
```

Use `GET /api/v1/models` to see what the installed worker actually permits. CPU always remains available. CUDA/XPU are only enabled if the installer verifies that PyTorch can use the accelerator inside the LXC.

---

# Endpoints

## Health

```http
GET /api/v1/health
```

Authentication: **not required**.

Example response:

```json
{
  "status": "ok",
  "service": "songverse-sonic",
  "version": "1.10.0",
  "queue_depth": 0,
  "queued_fast": 0,
  "scheduled_hq": 3,
  "default_model": "htdemucs",
  "hq_default_model": "htdemucs_ft",
  "default_device": "cpu",
  "allowed_devices": ["cpu"],
  "timezone": "Europe/Paris",
  "hq_window": {
    "start_hour": 2,
    "end_hour": 7
  },
  "analysis": {
    "enabled": true, "backend": "librosa", "chords": "btc",
    "parts": ["rhythm", "key", "sections", "chords"], "analysis_only": true, "lyrics": true
  }
}
```

`service` and `version` (since 1.6) say what is answering and which API version it speaks, so a client
can check a server before using it (Songverse's **Test connection**). An older server leaves them out.

## Models and devices

```http
GET /api/v1/models
X-API-Key: YOUR_API_KEY
```

Example response:

```json
{
  "models": [
    "hdemucs_mmi",
    "htdemucs",
    "htdemucs_6s",
    "htdemucs_ft",
    "mdx",
    "mdx_extra",
    "mdx_extra_q",
    "mdx_q"
  ],
  "default": "htdemucs",
  "hq_default": "htdemucs_ft",
  "devices": ["cpu"],
  "default_device": "cpu",
  "analysis": { "enabled": true, "backend": "librosa", "chords": "btc" }
}
```

## Queue

```http
GET /api/v1/queue
X-API-Key: YOUR_API_KEY
```

Returns jobs that are currently waiting or processing.

Fast jobs have priority over HQ jobs. The worker does not interrupt an HQ separation already in progress, but it will not start another HQ job while fast work is waiting.

---

# Create jobs

## Upload an audio file

```http
POST /api/v1/jobs
Content-Type: multipart/form-data
X-API-Key: YOUR_API_KEY
```

Multipart fields:

| Field | Required | Default | Description |
| --- | --- | --- | --- |
| `file` | yes | | Input audio file |
| `model` | no | `htdemucs` | Fast-stage model |
| `device` | no | worker default | Fast-stage compute device |
| `two_stems` | no | | e.g. `vocals` |
| `hq_enabled` | no | `false` | Schedule an HQ second pass |
| `hq_model` | no | `htdemucs_ft` | HQ model |
| `hq_device` | no | fast device | HQ compute device |
| `hq_not_before` | no | next HQ window | ISO-8601 datetime |
| `callback_url` | no | | Webhook endpoint |
| `callback_secret` | no | | HMAC signing secret |
| `separate` | no | `true` | Since 1.8: `false` for an **analysis-only job** (see below) |
| `lyrics` | no | | Since 1.9: the song's words, to find its own sections (see [Lyrics](#lyrics)) |

Example: fast now and HQ tonight:

```bash
curl -X POST 'https://demucs.example.com/api/v1/jobs' \
  -H 'X-API-Key: YOUR_API_KEY' \
  -F 'file=@song.wav' \
  -F 'model=htdemucs' \
  -F 'hq_enabled=true' \
  -F 'hq_model=htdemucs_ft' \
  -F 'callback_url=https://app.example.com/hooks/demucs' \
  -F 'callback_secret=YOUR_WEBHOOK_SECRET'
```

### Analysis-only jobs

Since 1.8, `separate=false` (a form field here, `"separate": false` in a URL job) analyses the
recording as it is, without separating it: for recordings already separated elsewhere, or separated
before the server could find something (chords, since 1.7). The job goes through the same states and
webhooks: its `fast` stage completes with no `files`, carrying the `analysis` as a separation's does,
and `fast.completed` is sent; `hq` is always disabled (asking for it is refused, 400). It waits behind
the separations already queued, before any HQ pass. Refused (400) when the server's analysis is
turned off: `analysis.analysis_only` in `/health` says whether it takes them. The analysis uses the
mix only, so it can differ a little from a separation's, which listens to the drums and bass apart.
A job's `separate` field says which kind it is (`true` for every job before 1.8).

### Lyrics

Since 1.9, a job can carry the song's words (`lyrics`, a form field here, a top-level string in a
URL job), as LRC, ChordPro or plain text: Sonic tells which from the text. They are written in the
order they are sung, a repeated chorus written again (ChordPro's `{chorus}` recall counts as one).
Each block is a **pass**: what follows a heading (`{c: Chorus}`, `{start_of_verse: Verse 1}`,
`Verse 2`, `[Bridge]`, `Refrain :`) up to the next one or a blank line, or, in an LRC, the lines up to a
blank line. A block may hold chords (`[G]` inline, or chord lines) and no words (an intro, a solo).
At most 64 KB (400 otherwise); they are read by the job's analysis and deleted, never kept with the job.

With them, the analysis places those passes on the recording's bars instead of guessing how many
sections there are and what each one is: each section in `sections` then says which pass it is
(`hint`) and the name its heading gave it (`name`), see [Music analysis](#music-analysis). The
recording may sing a pass once more than the text does, leave one out, or play a stretch the text
doesn't have (an intro, an instrumental): such a stretch is a section without `hint`. When the
passes don't fit the recording, the lyrics are not used and the sections are found as without them.
`analysis.lyrics` in `/health` says whether the server reads them; an older server ignores the field.

## Submit a remote URL

This is useful when MeTube or another internal service exposes the completed WAV over HTTP.

```http
POST /api/v1/jobs/url
Content-Type: application/json
X-API-Key: YOUR_API_KEY
```

Example:

```json
{
  "source_url": "http://metube.internal/downloads/song.wav",
  "fast": {
    "model": "htdemucs",
    "device": "cpu"
  },
  "hq": {
    "enabled": true,
    "model": "htdemucs_ft"
  },
  "callback_url": "https://app.example.com/hooks/demucs",
  "callback_secret": "YOUR_WEBHOOK_SECRET"
}
```

`"separate": false` (top level, since 1.8) makes it an [analysis-only job](#analysis-only-jobs); `hq` is then left out.
`"lyrics": "…"` (top level, since 1.9) carries the song's [lyrics](#lyrics).

If `hq.not_before` is omitted, the API schedules the HQ pass for the next configured HQ window. With the defaults, that is the next time the local `Europe/Paris` clock enters **02:00-07:00**.

An explicit timestamp is also accepted:

```json
{
  "hq": {
    "enabled": true,
    "model": "htdemucs_ft",
    "not_before": "2026-10-01T02:00:00+02:00"
  }
}
```

### MeTube workflow

A typical orchestrator can implement:

```text
source URL
   -> MeTube download / WAV conversion
   -> obtain MeTube/internal WAV URL
   -> POST /api/v1/jobs/url
   -> receive fast.completed webhook
   -> use fast stems
   -> receive hq.completed webhook later
   -> switch application metadata to HQ stems
```

If R2 is enabled on the Demucs worker, the orchestrator never needs to proxy the WAV or returned stems through itself.

---

# Job status

```http
GET /api/v1/jobs/{job_id}
X-API-Key: YOUR_API_KEY
```

Example after the fast pass has completed:

```json
{
  "id": "06ba97c7-87f5-4a53-b208-f1ab34bf9fac",
  "status": "hq_scheduled",
  "source_name": "song.wav",
  "two_stems": null,
  "separate": true,
  "error": null,
  "fast": {
    "status": "completed",
    "model": "htdemucs",
    "device": "cpu",
    "completed_at": 1790851200.0,
    "files": [
      {
        "name": "vocals.wav",
        "download_url": "https://demucs.example.com/api/v1/jobs/06ba97c7-87f5-4a53-b208-f1ab34bf9fac/files/fast/vocals.wav"
      }
    ],
    "r2": null
  },
  "hq": {
    "enabled": true,
    "status": "scheduled",
    "model": "htdemucs_ft",
    "device": "cpu",
    "not_before_iso": "2026-10-01T02:00:00+02:00",
    "files": [],
    "r2": null
  },
  "analysis": {
    "tempo": { "bpm": 72.1, "confidence": 0.9 },
    "first_beat": 0.42,
    "time_signature": { "numerator": 4, "denominator": 4, "confidence": 0.8 },
    "beats": [0.42, 1.25, 2.08, 2.91, 3.74],
    "downbeats": [0.42, 3.74],
    "beats_confidence": 0.8,
    "intro_free": false,
    "key": { "name": "G", "confidence": 0.7 },
    "sections": [
      { "start": 0.0, "label": "intro" },
      { "start": 8.3, "label": "verse" },
      { "start": 40.1, "label": "chorus" }
    ]
  },
  "callback_configured": true
}
```

See [Music analysis](#music-analysis) for the `analysis` block.

Important overall `status` values include:

```text
queued
fast_processing
hq_scheduled
hq_processing
completed
fast_completed
failed
```

`fast_completed` means the fast result is still valid but the requested HQ upgrade failed.

---

# Music analysis

Since v1.4 a job carries an optional, top-level `analysis` block (next to `fast` and `hq`; `beats`, `downbeats`, `beats_confidence` and `intro_free` were added in v1.5; `chords`, `beat_chords` and `engines` in v1.7), in
`GET /api/v1/jobs/{id}` and in the `job` object of every webhook payload.

```json
"analysis": {
  "tempo": { "bpm": 72.1, "confidence": 0.9 },
  "first_beat": 0.42,
  "time_signature": { "numerator": 4, "denominator": 4, "confidence": 0.8 },
  "beats": [0.42, 1.25, 2.08, 2.91, 3.74],
  "downbeats": [0.42, 3.74],
  "beats_confidence": 0.8,
  "intro_free": false,
  "key": { "name": "G", "confidence": 0.7 },
  "sections": [
    { "start": 0.0, "label": "intro" },
    { "start": 8.3, "label": "verse", "hint": 0, "name": "Verse 1" },
    { "start": 40.1, "label": "chorus", "hint": 1, "name": "Chorus" }
  ],
  "chords": [
    { "start": 0.0, "end": 0.42, "label": "N" },
    { "start": 0.42, "end": 3.74, "root": 7, "quality": "maj", "label": "G" },
    { "start": 3.74, "end": 7.06, "root": 4, "quality": "min7", "label": "E:min7" },
    { "start": 7.06, "end": 10.38, "root": 0, "quality": "maj", "bass": 4, "label": "C/3" }
  ],
  "beat_chords": [1, 1, 1, 1, 2],
  "engines": { "rhythm": "librosa@1.10.0", "key": "librosa@1.10.0", "sections": "librosa-lyrics@1.10.0", "chords": "btc@1673d23f8f9a" }
}
```

**It is optional at every level.** Clients that ignore it are unaffected.

- `analysis` is `null` until the fast pass is done, and stays `null` when analysis is disabled
  (`ANALYSIS_ENABLED=false`), failed, or found nothing. A failed analysis never fails the job.
- Every field inside it may be absent: only what could be determined is returned.
- `beats` and `downbeats` come as a pair or not at all, and are never partial: an invalid or empty list
  is left out rather than trimmed. `tempo.bpm` is the median beat interval of `beats` (in BPM), and
  `first_beat` is the first downbeat, so the three always agree. When the intro is steady but has no
  detectable beats (a quiet pad, say), the grid is extended back to where the music starts.
- The analysis runs as the last step of the fast pass, so it is already present when
  `fast.status` becomes `completed` and in the `fast.completed` webhook. It is not recomputed after the HQ pass.

| Field | Meaning |
| --- | --- |
| `tempo.bpm` | Beats per minute, 20 to 400. |
| `first_beat` | Seconds from the start of the file to the first downbeat (bar 1, beat 1), 0 to 600. |
| `time_signature` | `numerator` 1 to 16, `denominator` one of 2, 4, 8, 16. Since 1.10 a song in 6/8 (two groups of three eighths a bar, which earlier versions gave as 3/4 or as twice as many bars of 3/4) is `6/8`: its `beats` are then the six eighths of each bar and `tempo.bpm` counts eighths (142 for a bar of 2.53 s). 12/8, and 6/8 heard at the dotted quarter, are not found yet. |
| `beats` | Every beat found, in seconds from the start of the file, strictly increasing, within the file's length, rounded to the millisecond. Since 1.8.1 each beat is where its hit starts (its attack), not where the onset peaks a few tens of ms later, so a click on them sounds with the recording. At the same beat level as `tempo.bpm` (not half or double time). Listed from `first_beat` onward. |
| `downbeats` | The first beat of each bar, in seconds. Always a subset of `beats`; the first one is `first_beat`. |
| `beats_confidence` | 0 to 1, for `beats` and `downbeats` together (low when the beats land on few onsets or the tempo octave is in doubt). Only present with `beats`. |
| `intro_free` | `true`: no steady beat before the first downbeat (a rubato, spoken or free opening), so `first_beat` is where the beat starts. `false`: the beat is steady before it (or there is no intro). Left out when unknown. |
| `key.name` | Letter `A`-`G`, optional `#` or `b`, then `m` for minor: `G`, `Bb`, `F#m`, `Ebm`. |
| `sections` | In time order: `start` (seconds) and a lowercase `label`. Labels used: `intro`, `verse`, `pre-chorus`, `chorus`, `bridge`, `inst`, `instrumental`, `solo`, `break`, `interlude`, `outro`, `tag`. Since 1.9, when the job's [lyrics](#lyrics) were used: `hint`, the 0-based pass of the lyrics the section is (two sections share one when the recording sings it once more), and `name`, that pass's heading as written (left out when it had none); a section without `hint` is a stretch the lyrics don't have. |
| `chords` | Since 1.7, only when the server has chords turned on (`analysis.chords` in `/health`). The recording's chords in time order, not overlapping, within the file: `start`, `end` (seconds), `label`. `"label": "N"` is a stretch with no chord (silence, drums alone) and has nothing else. Otherwise `root` and `bass` are pitch classes, 0 = C to 11 = B, so a client spells them in the song's key (`root: 1` is C# or Db); `bass` is left out when it is the root. `quality` is one of `maj`, `min`, `dim`, `aug`, `min6`, `maj6`, `min7`, `minmaj7`, `maj7`, `7`, `dim7`, `hdim7`, `sus2`, `sus4`. `label` is the chord in Harte's syntax (as `mir_eval` reads it), for display and comparison only: read `root`, `quality` and `bass`. All or nothing, like `beats`. |
| `beat_chords` | Since 1.7, only with `beats` and `chords`: for each entry of `beats`, the index in `chords` of the chord sounding there (the one heard longest until the next beat), or `null`. Same length as `beats`. |
| `engines` | Since 1.7: which engine found each part present (`rhythm` for tempo, meter, beats and the first beat; `key`; `sections`; `chords`), as `name@version` (a chord engine's version is the start of its checkpoint's sha256). Parts from different engines are never mixed within one part. `sections` is `librosa-lyrics@…` when they were placed from the job's lyrics (since 1.9). |
| `*.confidence` | 0 to 1 when available; omitted otherwise. `chords` has none: treat them as suggestions. |

Notes on quality: the default backend is heuristic. Tempo is chosen with a prior toward roughly
60-130 bpm because double/half time cannot be told apart from the audio alone: songs counted faster
than ~140 may be reported at half tempo, and `tempo.confidence` is lowered (below 0.5) when the choice
is a coin flip. `first_beat` is the start of bar 1 on the beat grid, extended back over a drumless
intro to where the music starts (unless the intro is `intro_free`, then it is the first steady downbeat). The meter is
3/4 vs 4/4 only; relative major/minor confusions are possible (reflected in a lower key confidence);
section labels are a rough guess and boundaries are approximate. Beat times have roughly 10-25 ms of timing resolution. Downbeat placement uses
kick accents and chord changes, so it can be a beat or a bar off on unusual material.

**Chords and bars (since 1.7).** When chords are on, chord changes also check the bar lines: chords
change on the first beat of a bar far more often than elsewhere, so when the changes clearly fall on
another beat of the bar than the downbeats found from the drums, `downbeats` and `first_beat` move
there (`beats` and `tempo` do not change). Section starts (after the first, at 0) are always one
of the reported `downbeats`, which follow the music when its tempo moves, so a client can cue or loop
a section on whole bars. Since 1.8.1 they are found at the bar lines where the music changes most,
preferring sections a multiple of 4 bars long, and `allin1`'s are placed on its downbeats too.
Since 1.9, with the job's [lyrics](#lyrics), each pass is placed where the music changes, where it
sounds like the other passes of the same kind (verses alike, choruses alike), where a vocal stem has
voice for a sung pass, on the bars whose chords are the pass's (when chords are on), with about the
same bars per line all through the song, and, from an LRC, where its first line is sung (the LRC's
clock may be offset or run up to 15 % fast or slow against the recording's: it is fitted first). They
are used only when same-kind passes really do sound alike, so a chart that doesn't match the
recording is left unused; one whose structure matches but whose names are swapped can't be told.
The chord engine knows the 14 qualities
above and no inversions yet (`bass` is reserved for engines that find them); it was trained on pop
and rock recordings, and may call a sus or add9 voicing a plain triad.

`GET /api/v1/health` and `GET /api/v1/models` report `analysis.enabled`, `analysis.backend`, (since 1.7)
`analysis.chords`: the chord engine, or `null` when chords are off, and (since 1.8) `analysis.parts`, the
parts it can find (`rhythm`, `key`, `sections`, `chords`), and `analysis.analysis_only`; since 1.9
`analysis.lyrics`, whether a job's lyrics are read.

**Time limits.** `ANALYSIS_TIMEOUT_SECONDS` is a budget the analysis checks between its steps (later
steps are skipped once it is spent). Since 1.8 the analysis also runs in a process of its own, stopped at
`ANALYSIS_HARD_LIMIT_SECONDS` (default three times the budget, at least 120 s): the job then completes
without an analysis, as when the analysis fails.

---

# HQ scheduling

## Add/reschedule an HQ pass

An HQ upgrade can be added after the original job was created:

```http
POST /api/v1/jobs/{job_id}/hq
Content-Type: application/json
X-API-Key: YOUR_API_KEY
```

```json
{
  "model": "htdemucs_ft",
  "device": "cpu",
  "not_before": "2026-10-01T02:00:00+02:00"
}
```

`not_before` may be omitted to use the next configured HQ window.

## Cancel a scheduled HQ pass

```http
DELETE /api/v1/jobs/{job_id}/hq
X-API-Key: YOUR_API_KEY
```

A currently running HQ pass cannot be cancelled through this endpoint.

## HQ window behavior

Default configuration:

```ini
DEMUX_TIMEZONE=Europe/Paris
DEMUX_HQ_START_HOUR=2
DEMUX_HQ_END_HOUR=7
```

HQ work only **starts** while the configured window is open. A separation already running at the end of the window is allowed to finish. Only one HQ job is fed into the worker at a time, preventing the whole nightly backlog from being pre-queued past the end of the window.

Fast work always receives a higher queue priority than waiting HQ work.

---

# Results

## Download a stem

```http
GET /api/v1/jobs/{job_id}/files/{quality}/{filename}
X-API-Key: YOUR_API_KEY
```

`quality` is either:

```text
fast
hq
```

Examples:

```text
/api/v1/jobs/JOB_ID/files/fast/vocals.wav
/api/v1/jobs/JOB_ID/files/hq/vocals.wav
```

## Logs

Fast-stage log:

```http
GET /api/v1/jobs/{job_id}/log/fast
```

HQ-stage log:

```http
GET /api/v1/jobs/{job_id}/log/hq
```

For compatibility, `/api/v1/jobs/{job_id}/log` returns the fast-stage log.

## Delete a job

```http
DELETE /api/v1/jobs/{job_id}
X-API-Key: YOUR_API_KEY
```

The input file, local outputs and notification records are removed. Objects already uploaded to R2 are **not** deleted automatically.

---

# Webhooks

When `callback_url` is supplied, the API can emit:

```text
fast.completed
fast.failed
hq.started
hq.completed
hq.failed
```

Example payload:

```json
{
  "event": "hq.completed",
  "job_id": "06ba97c7-87f5-4a53-b208-f1ab34bf9fac",
  "timestamp": "2026-10-01T02:18:43+02:00",
  "job": {
    "id": "06ba97c7-87f5-4a53-b208-f1ab34bf9fac",
    "status": "completed",
    "fast": {},
    "hq": {},
    "analysis": {}
  }
}
```

The `job` object is the same as in `GET /api/v1/jobs/{id}`, including `analysis`.

## Webhook signatures

If `callback_secret` was supplied, the raw request body is signed with HMAC-SHA256.

Header:

```http
X-Demucs-Signature: sha256=HEX_DIGEST
```

Verification pseudo-code:

```python
expected = hmac.new(secret.encode(), raw_request_body, hashlib.sha256).hexdigest()
valid = hmac.compare_digest(received_signature, f"sha256={expected}")
```

Always verify the signature against the **raw HTTP request body**, before JSON re-serialization.

## Retry policy

Webhook delivery is persisted in SQLite and retried independently from the Demucs processing queue.

Default attempts occur approximately:

```text
immediately
+30 seconds
+2 minutes
+10 minutes
+1 hour
```

The default maximum is five attempts. Polling `GET /api/v1/jobs/{job_id}` remains the authoritative fallback if every webhook attempt fails.

Inspect deliveries with:

```http
GET /api/v1/jobs/{job_id}/notifications
X-API-Key: YOUR_API_KEY
```

---

# Cloudflare R2 / S3-compatible output

Enable in `/etc/demucs-api/demucs-api.env`:

```ini
DEMUX_R2_ENABLED=true
DEMUX_R2_ENDPOINT=https://ACCOUNT_ID.r2.cloudflarestorage.com
DEMUX_R2_ACCESS_KEY_ID=...
DEMUX_R2_SECRET_ACCESS_KEY=...
DEMUX_R2_BUCKET=your-bucket
DEMUX_R2_PREFIX=demucs
```

Objects use separate fast/HQ prefixes:

```text
demucs/JOB_ID/fast/vocals.wav
demucs/JOB_ID/fast/drums.wav
demucs/JOB_ID/hq/vocals.wav
demucs/JOB_ID/hq/drums.wav
```

The job response returns bucket/object keys. Use credentials restricted to only the required bucket and operations.

---

# Reverse proxy deployment

The application should continue listening on the private LXC address/port, normally `0.0.0.0:8000`, while the reverse proxy handles TLS.

Recommended settings:

```ini
DEMUX_PUBLIC_BASE_URL=https://demucs.example.com
DEMUX_TRUST_PROXY_HEADERS=true
DEMUX_FORWARDED_ALLOW_IPS=10.0.0.20
```

`DEMUX_FORWARDED_ALLOW_IPS` should be the IP address of the reverse proxy that directly connects to the LXC. Do not use `*` unless the API port is separately firewalled so that **only** your trusted proxy can reach it.

After changing configuration:

```bash
systemctl restart demucs-api
```

## Nginx example

```nginx
server {
    listen 443 ssl http2;
    server_name demucs.example.com;

    client_max_body_size 1100M;

    location / {
        proxy_pass http://DEMUX_LXC_IP:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Only needed for long file uploads/downloads, not Demucs processing itself.
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

The separation itself is asynchronous, so the proxy does **not** need a multi-hour timeout while Demucs runs.

## Caddy example

```caddyfile
demucs.example.com {
    reverse_proxy DEMUX_LXC_IP:8000
}
```

Raise request-body limits in any middleware/WAF placed in front of Caddy if you intend to upload large WAV files through the public endpoint.

## Firewall recommendation

If the reverse proxy is on another server, restrict TCP/8000 on the LXC/Proxmox firewall so only:

- the reverse proxy; and
- explicitly trusted internal orchestrators

can reach it directly.

---

# Source URL security

Remote-source jobs intentionally support private URLs because the primary use case includes an internal MeTube server:

```ini
DEMUX_ALLOW_PRIVATE_SOURCE_URLS=true
```

This is an SSRF-sensitive feature. Keep the API key restricted to trusted services.

Private webhook destinations are disabled by default:

```ini
DEMUX_ALLOW_PRIVATE_CALLBACK_URLS=false
```

Set it to `true` only if your webhook receiver is also on the private network.

When private addresses are disabled (always for webhooks unless allowed, and for sources with
`DEMUX_ALLOW_PRIVATE_SOURCE_URLS=false`), every address is checked, not only the one submitted (since 1.6.1):

- each redirect a source download follows (5 at most) is checked again before it is fetched;
- the server connects to the address it checked (the name is resolved once, every address it resolves
  to must be public, TLS is still verified against the name), so a name can't be pointed at a private
  address after the check;
- behind an outbound proxy (`HTTPS_PROXY`), the proxy resolves names: Sonic checks the name of the first
  address and of every redirect, and the proxy's own rules do the rest.

Webhooks never follow redirects.

---

# Error codes

Common responses:

| HTTP | Meaning |
| ---: | --- |
| 202 | Job accepted/scheduled |
| 400 | Invalid model/device/URL/schedule |
| 401 | Missing or invalid API key |
| 404 | Job/file not found |
| 409 | Requested action conflicts with a running stage |
| 413 | Input exceeds configured maximum size |
| 422 | Invalid JSON/form schema |

Processing failures do not normally change the original submission response. They are reflected later in the job state and optional webhook event.
