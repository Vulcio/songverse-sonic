# RFC: Songverse Sonic — Modular Analysis Architecture

**Status:** Proposed, for review (SONIC-01, #4)  
**Focus:** Rhythm and beat tracking first  
**Baseline:** Songverse Sonic 1.6.0  
**Proposed first additive API release:** 1.7.0 (subject to implementation scope)  
**Repositories:** `Vulcio/songverse-sonic`, `Vulcio/Oneverse`

## 1. Summary

Evolve Songverse Sonic from a separation service with inline best-effort music analysis into a modular processing service supporting independent, persistent, versioned analysis runs.

The first advanced capability will be **rhythm analysis**, comparing the existing stem-aware librosa implementation with experimental neural beat/downbeat tracking, starting with Beat This!.

Chord recognition, song structure, and other capabilities will follow the same architecture later. They are not part of the first implementation milestone.

This proposal does **not** require an immediate major API version or a replacement of the current deployment architecture.

## 2. Motivation

Sonic currently performs analysis inside the fast separation workflow. The existing analyzer provides tempo, beats, downbeats, meter, first downbeat, free-intro information, key, and sections.

Current limitations:

- Analysis cannot be rerun independently against retained artifacts through the public API.
- Computationally expensive analysis can delay the `fast.completed` transition.
- `ANALYSIS_TIMEOUT_SECONDS` is an advisory budget, not a hard execution deadline.
- Model selection is effectively global rather than capability-specific.
- Individual analysis runs and their versions are not persisted separately.
- Results cannot be directly benchmarked against alternative models through a common interface.
- The architecture does not support isolated model environments cleanly.

Existing strengths must remain: Demucs separation, fast/HQ scheduling, persistent jobs, HMAC webhooks, R2/S3 integration, CPU deployments, and Proxmox support.

## 3. Goals

1. Separate analysis execution from stem separation.
2. Preserve the existing v1.6 job and webhook contracts.
3. Enable repeatable analysis on retained recordings/stems.
4. Define independent analyzer interfaces for rhythm and future capabilities.
5. Persist results, model provenance, inputs, failures, and execution metadata.
6. Support hard timeouts, crash recovery, and bounded resource usage.
7. Allow experimental neural models without requiring them on every installation.
8. Establish evidence-based model selection through reproducible benchmarks.
9. Preserve manual Songverse metadata as authoritative.

## 4. Non-goals

- Rewriting the service in another language.
- Replacing Demucs.
- Introducing distributed processing across machines immediately.
- Shipping chord recognition or structure models in the first milestone.
- Building Songverse's chord editor or playback UI inside Sonic.
- Enabling non-commercial or ambiguously licensed checkpoints by default.
- Automatically replacing existing Songverse beat maps with neural predictions.

## 5. Proposed components

### Sonic API

Retains FastAPI and existing authentication.

Responsibilities:

- Job submission and inspection.
- Analysis-run requests.
- Capability discovery.
- Artifact and result access.
- Webhook registration and delivery.

The API must not perform heavyweight inference inside request handlers.

### Job orchestrator

Coordinates independent stages and determines when existing public job states can transition.

Responsibilities:

- Persistent scheduling.
- Dependency resolution.
- Priority policies.
- Recovery after restart.
- Retry policies.
- Event generation.

### Separation worker

Runs Demucs using the existing model and device settings.

Fast separation continues to outrank pending HQ work. HQ processing retains its configured scheduling window.

### Analysis worker

Executes a requested analysis capability against recorded artifacts.

The first supported capability is `rhythm`; the current librosa analyzer is its baseline implementation. A future Beat This! adapter will be opt-in.

Workers run inference in supervised child processes or isolated containers so timeouts can actually terminate execution.

### Artifact manager

Tracks original audio, separation outputs, and analysis artifacts.

Coordinates local paths, optional R2/S3 locations, checksums, retention, and deletion.

## 6. Analyzer contract

Each analyzer advertises:

- Stable engine ID and implementation version.
- Capability, initially `rhythm`.
- Required input roles and supported formats.
- Supported CPU/GPU devices.
- Required checkpoint identity and checksum, when applicable.
- Availability and deployment requirements.
- Resource requirements and configurable execution limits.

Each execution accepts an immutable description of its selected input artifacts and configuration, and returns a normalized result or a structured failure.

The engine must not directly change public job status, dispatch webhooks, or update Songverse data.

An adapter may return incomplete findings. Missing predictions must remain missing rather than becoming fabricated defaults.

## 7. Persistent entities

### AudioArtifact

Represents a recording or derived audio resource.

Suggested fields:

- `id`, `job_id`, `role`, `origin`
- `storage_backend`, `storage_key`
- `sha256`, `duration_seconds`, `sample_rate`
- `source_time_offset_seconds`
- `created_at`, `expires_at`, `state`

Audio timestamps are always interpreted in the **original recording's time coordinates**. Derived stems require explicit alignment validation.

An artifact is immutable once marked ready.

### AnalysisRun

Represents one execution of one analyzer against specific input artifacts.

Suggested fields:

- `id`, `job_id`, `capability`
- `engine_id`, `engine_version`
- `model_id`, `model_checksum`
- `input_artifact_ids`, `input_digest`
- `settings_json`, `status`, `attempt_count`
- `requested_at`, `started_at`, `completed_at`
- `timeout_seconds`, `error_code`, `error_message`
- `result_id`, `runtime_metrics_json`

Proposed lifecycle:

`queued → running → succeeded | failed | timed_out | cancelled`

An interrupted `running` entry must be recovered or marked retryable according to a documented policy.

### AnalysisResult

Stores an immutable, validated output.

Suggested fields:

- `id`, `run_id`, `schema_version`
- `capability`, `result_json`
- `quality_json`, `created_at`

Multiple results for the same recording must be preserved.

### AnalysisProfile

Represents a named processing policy.

Initial profiles:

- `legacy`: current compatible behavior.
- `rhythm-baseline`: current librosa rhythm analyzer.
- `rhythm-experimental`: explicitly selected candidate engine.

Profiles determine requested capabilities; they must not hide the actual engine and model recorded in an analysis run.

## 8. Rhythm result contract

Introduce a normalized internal schema that can represent:

- Beat timestamps.
- Downbeat timestamps.
- Estimated global tempo.
- Optional local tempo map.
- Meter, when confidently determined.
- First confirmed downbeat.
- Free, steady, or unknown introduction.
- Rhythm quality flags.
- Model provenance.

Required validation:

- Timestamps are finite, nonnegative, strictly increasing, and within the recording.
- A published downbeat must align with a beat under a defined tolerance.
- BPM and time signature must not be inferred from missing evidence.
- Beat level and half/double-time ambiguity must be represented explicitly.
- Unknown confidence is `null`, not an invented number.
- No model receives an artificial confidence of 1.0 merely for completing inference.

Beat This! produces beat and downbeat events; it does not by itself establish an arbitrary musical time signature. Meter inference must therefore be a separate, testable process.

The existing public `analysis` representation remains available through a **legacy projection**:

- Maintain current field names, formats, and optionality.
- Preserve the existing consistent relationship among `tempo.bpm`, `beats`, `downbeats`, and `first_beat`.
- Continue rounding legacy timestamps to milliseconds.
- Do not mix beat positions from one engine with downbeats or tempo from another.
- Return `null` or omit uncertain fields rather than forcing a prediction.

## 9. Backward compatibility and events

Existing v1.6 consumers assume:

1. `fast.completed` means the fast result is available.
2. Available legacy analysis has already been stored.
3. The webhook payload contains that analysis.
4. Missing or failed analysis never makes a separation fail.

These semantics must remain intact for legacy requests.

The new internal lifecycle distinguishes:

1. Fast separation artifacts ready.
2. Legacy quick analysis finished or safely failed.
3. Legacy `fast.completed` published.
4. Optional advanced analysis completed independently.

New clients may subscribe to additive events such as `stems.ready` and `analysis.completed`. Those event types must **not** be sent to legacy webhook consumers unless explicitly opted in.

Existing signed webhook headers, payload structure, and event types remain supported.

The current `fast.status` should not become `completed` before the compatibility analysis gate has resolved. New clients may read a separate artifact-readiness field to access stems earlier.

Webhook deliveries remain at least once, with durable retry state and event IDs for idempotent consumers.

## 10. Proposed additive API

Retain all existing `/api/v1` endpoints.

Candidate additions:

- `GET /api/v1/analysis-capabilities`
- `POST /api/v1/jobs/{job_id}/analysis-runs`
- `GET /api/v1/jobs/{job_id}/analysis-runs`
- `GET /api/v1/jobs/{job_id}/analysis-runs/{run_id}`

The create endpoint requests a capability and engine against an eligible retained artifact.

It must support an idempotency key and return a structured error when the required source has expired.

The resulting schema, authentication rules, and event subscriptions require review before implementation.

Any additive public API change must update `API.md`, `app/VERSION`, and the FastAPI version together under repository policy.

## 11. Scheduling and resource policy

Start with one active scheduler per deployment.

Use SQLite transactional job claiming with a single-host deployment contract. An in-memory queue may be used only as a wake-up optimization; it must not be the sole source of execution state.

Suggested logical queues:

- Fast separation: highest separation priority.
- Legacy quick analysis: must resolve before legacy completion.
- Experimental analysis: independent, lower priority.
- HQ separation: retains the existing time-window rules.

Initially, limit analysis and separation concurrency explicitly. GPU memory must not be overcommitted by independently launching competing models.

Use process-level timeouts, cancellation, termination, and bounded retry rules. A failed or timed-out analysis must not invalidate successfully generated stems.

Distributed multi-host leasing is a future architectural decision, not implicitly supported by SQLite.

## 12. Retention and security

Analysis runs can be recreated only while the necessary input artifacts exist.

Define explicit policies for:

- Local source retention.
- Stems retained for reanalysis.
- Pending runs that depend on artifacts.
- Model checkpoint storage and checksums.
- Job deletion while processing.
- Local and remote artifact cleanup.
- Authenticated artifact access.
- Remote URL redirect and DNS validation.
- Per-job resource and upload limits.

Do not silently change the existing default allowing private source URLs, because current internal workflows rely on it. Provide explicit security guidance and enforce safe transport handling.

Keep model downloads out of the processing hot path where possible, and verify model artifacts before activation.

## 13. Experimental rhythm model

First experimental candidate: **Beat This!**

Compare at least:

- Sonic's existing librosa/stem-aware implementation.
- Beat This! `final0`.
- Beat This! `small0`.
- Where useful, original mix versus selected Demucs stems.

Both model variants must remain optional.

Before enabling distribution:

- Record code and checkpoint licenses.
- Record exact model versions and checksums.
- Review training-data provenance and commercial-use caveats.
- Validate CPU and available GPU execution.
- Check dependency compatibility with both deployment targets.

An engine may be present but unavailable if its required checkpoint or hardware is missing.

## 14. Evaluation

Build a reproducible benchmark corpus containing:

- Existing synthetic recordings.
- Real, permission-cleared recordings.
- Slow and fast tempos.
- Straight and swung rhythms.
- 3/4, 4/4, compound meter, and irregular meter.
- Drumless passages.
- Quiet introductions and rubato.
- Variable-tempo live performances.
- Pickup bars and ambiguous downbeats.

Evaluate beat and downbeat timing independently.

Measure beat F-measure, downbeat F-measure, continuity, tempo-octave errors, meter errors, median timing error, failure rate, processing latency, peak RAM, and GPU memory where applicable.

Report both conventional beat-tracking evaluation and full-track evaluation including introductions. Standard evaluation commonly omits beats before five seconds; our first-beat use case requires measuring them separately.

Prevent train/test leakage by recording dataset origins and avoiding model checkpoints trained on benchmark audio.

No automatic model-selection change is approved until the benchmark criteria and measurements have been reviewed.

## 15. Deployment and migration

Deliver incrementally:

1. Behavior-preserving module extraction.
2. Benchmark and security foundations.
3. Versioned result storage.
4. Independent analysis-run scheduling.
5. Experimental Beat This! adapter.
6. Shadow comparison and opt-in testing.
7. Future default selection after validation.

Keep the current CPU installation lightweight.

Ensure Docker and Proxmox LXC expose equivalent core capabilities; optional model runtimes may require separate installation profiles.

Document any hardware-specific capabilities and limitations.

## 16. Release criteria

The first architectural milestone is complete when:

- Existing tests and Songverse-compatible behavior still pass.
- Separation and analysis execute independently.
- Legacy completion/webhook semantics are retained.
- Analysis runs survive restarts.
- Retained audio supports independent reanalysis.
- Hard timeouts work.
- Invalid analysis cannot corrupt the legacy response.
- The baseline rhythm engine can be benchmarked.
- Unsupported hardware and missing models are reported honestly.
- Neither deployment method silently loses functionality.

**The first advanced-model milestone** is complete when Beat This! is integrated as an optional engine, evaluated against the baseline on reproducible data, and approved for the intended deployment/license conditions.

## 17. Open design questions

These require explicit resolution before implementation, without blocking initial drafting:

1. Should new event subscriptions be configured per job, per API client, or both?
2. How long should source audio be retained by default for reanalysis?
3. Should experimental workers run in subprocesses within the main container, or independently deployable containers?
4. What hardware/resource profiles should a self-hosted administrator be able to select?
5. Which benchmark improvements and regressions should constitute the promotion gate for an advanced rhythm engine?

Provisional recommendation: per-job event subscriptions; existing retention defaults initially; subprocesses for the baseline and isolated runtime for incompatible models; explicit deployment capability profiles; benchmark thresholds approved after collecting baseline results.

## 18. First reviewable slice and dependency roadmap

This RFC describes a target architecture, not capabilities already shipped. The
first implementation PR only extracts the existing array-based librosa rhythm
calculations into `app/rhythm.py` (SONIC-02a, #5), keeping audio selection,
sanitization and legacy orchestration unchanged. It does not satisfy the wider
release criteria in section 16. No public API version bump is needed for this
behavior-preserving extraction; the deployed API remains 1.6.0.

| Planning ID | Work | Dependency / review gate |
| --- | --- | --- |
| SONIC-01 (#4) | Architecture RFC and compatibility contract | Review this proposal |
| SONIC-02 (#6; first slice #5) | Extract analysis, then orchestration/worker/storage boundaries | Small behavior-preserving PRs, existing tests green |
| SONIC-05 (#9) | Reproducible rhythm benchmark | Baseline boundary; corpus rights, annotations and metric definitions |
| SONIC-06 (#10) | Artifact retention and security | Review source/callback transport, quotas and deletion semantics |
| SONIC-04 (#8) | Versioned rhythm results and provenance | Reviewed schema and compatibility projection |
| SONIC-03 (#7) | Persistent independent analysis runs | Results + artifact lifecycle; transactional claiming, recovery and hard timeout tests |
| Later | Experimental Beat This! adapter | Baseline benchmarks, checkpoint/license review and isolated optional runtime |

Do not treat the roadmap as permission to install neural models. The extracted
baseline is not yet a versioned analyzer
adapter: it accepts aligned mono sample arrays and returns the existing raw
rhythm fields, including a private section-alignment hint. A benchmark can call
it without importing FastAPI, SQLite or workers. The future adapter must wrap
validated immutable artifact descriptions and the normalized result contract.

### Verified compatibility dependencies

The repository review covered Sonic `AGENTS.md`, `CLAUDE.md`, `API.md`,
`README.md`, deployment scripts and CI; PRs
[#1](https://github.com/Vulcio/songverse-sonic/pull/1) and
[#2](https://github.com/Vulcio/songverse-sonic/pull/2), and deployment issue
[#3](https://github.com/Vulcio/songverse-sonic/issues/3).

Oneverse's existing contracts are documented in
[#175](https://github.com/Vulcio/Oneverse/issues/175),
[#299](https://github.com/Vulcio/Oneverse/issues/299) and
[#300](https://github.com/Vulcio/Oneverse/issues/300), and implemented in
`apps/api/src/stem-separation/demucs-client.ts`,
`apps/api/src/stem-separation/analysis.ts` and `e2e/lib/fake-demucs.mjs`.
Detected values fill missing metadata; existing manual values remain authoritative.
The current client accepts omitted confidence, so new experimental results must
not accidentally gain trust through a legacy projection that omits unknown
confidence. Decide that policy explicitly before integrating a new engine.

The extraction changes no client-visible field or setting and requires no
Oneverse code change. Independent result consumption is tracked in
[Oneverse #371](https://github.com/Vulcio/Oneverse/issues/371) and requires a
separate PR covering client types, ingestion, fake server,
manual-override preservation and relevant end-to-end tests.

### Review decisions before the next milestone

- Specify result validation tolerances, tempo coherence and missing-evidence
  behavior before implementing SONIC-04. The current sanitizer is not the future
  normalized validator; it does not independently enforce every proposed invariant.
- Document migration and rollback before creating persistent tables.
- Pin benchmark input hashes, generator/settings, engine/library versions,
  hardware, timing policy and metric parameters in each report. Synthetic tests
  establish regressions, not real-music accuracy or model promotion.
- Keep current retention and private-source defaults until explicitly reviewed.
- Treat checkpoint licensing, training-data caveats and commercial distribution
  suitability as unverified release gates; do not infer approval from this RFC.

