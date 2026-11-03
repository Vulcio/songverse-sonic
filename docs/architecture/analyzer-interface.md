# Internal analyzer interface (SONIC-02)

This is the second extraction slice of [#6](https://github.com/Vulcio/songverse-sonic/issues/6),
following the standalone rhythm baseline in #11. It implements the legacy
full-analysis boundary; the versioned capability-specific contract in the
[evolution RFC](sonic-evolution-rfc.md) remains future work.

- `AnalysisInput` is a frozen descriptor containing a tuple of selected audio
  paths, optional source path, work directory and requested device. It snapshots
  the selection, not file contents. Retention and alignment remain caller duties.
- `Analyzer.analyze(inputs, budget)` returns raw optional findings or raises.
  The current backend may include the private `_duration` validation hint.
  An engine does not write job state, persist results or send notifications.
- `AnalysisBudget.ok()` retains the existing advisory time budget. It cannot
  terminate inference; supervised hard deadlines belong to SONIC-03.
- `LibrosaAnalyzer` and `AllInOneAnalyzer` in `analysis.py` adapt the unchanged
  backend functions. All-In-One imports only when invoked and remains opt-in.
- `LegacyAnalysisAdapter` accepts a baseline engine, optional primary engine
  and sanitizer. It always runs the baseline, fills absent primary findings,
  preserves the existing beat-field grouping and validates with baseline duration.
  Primary failure falls back; baseline failure retains available primary findings.
- `analysis.analyze_job` preserves its signature, environment settings, disabled
  behavior and sanitized return shape. Unknown backend names still use librosa.

The interface module imports only the standard library. Tests can inject engines
without FastAPI, SQLite, Demucs or optional model installations. Algorithm helpers
and audio loading remain in `analysis.py`; the rhythm calculations stay in
`rhythm.py`. Moving them is a later extraction, not part of this slice.

The existing legacy merge and sanitizer policies are intentionally preserved.
This does not establish the RFC's stronger cross-field coherence guarantees or
versioned provenance. Those require the reviewed schema/projection in SONIC-04
(#8), including an explicit policy for meter and unknown confidence.

Both LXC download paths fetch `analyzers.py`; Docker already copies all of `app/`.
No public request, response, event or setting changes, so API 1.6.0 and the
Oneverse client/fake remain applicable. No SQLite schema changes are made.
