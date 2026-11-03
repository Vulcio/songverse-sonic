"""Music analysis (tempo, first downbeat, meter, key, sections) for finished fast passes.

Everything here is best effort: every field of the result is optional and a
failure of any step only leaves its fields out. `analyze_job` never raises.

Default backend: librosa (ISC licence, no trained model files) plus small
heuristics. Optional backend: All-In-One (`ANALYSIS_BACKEND=allin1`), see README.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from analyzers import AnalysisBudget, AnalysisInput, LegacyAnalysisAdapter
from rhythm import analyze_rhythm

log = logging.getLogger("demucs-api.analysis")

ANALYSIS_ENABLED = os.getenv("ANALYSIS_ENABLED", "true").lower() == "true"
ANALYSIS_BACKEND = os.getenv("ANALYSIS_BACKEND", "librosa").lower()
ANALYSIS_TIMEOUT = float(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "60"))

BPM_RANGE = (20.0, 400.0)
FIRST_BEAT_RANGE = (0.0, 600.0)
DENOMINATORS = {2, 4, 8, 16}
SECTION_LABELS = {
    "intro", "verse", "pre-chorus", "chorus", "bridge", "inst", "instrumental",
    "solo", "break", "interlude", "outro", "tag",
}
KEY_RE = re.compile(r"^[A-G][#b]?m?$")

SHARP_TO_FLAT_TONIC = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_TONIC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


# --------------------------------------------------------------------------- #
# Output normalisation: the contract Songverse relies on lives here.
# --------------------------------------------------------------------------- #

def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _confidence(value: Any) -> float | None:
    v = _num(value)
    return None if v is None else round(min(1.0, max(0.0, v)), 3)


def _with_conf(out: dict, raw: dict) -> dict:
    conf = _confidence(raw.get("confidence"))
    if conf is not None:
        out["confidence"] = conf
    return out


def _time_list(value: Any, duration: float | None) -> list[float] | None:
    """A strictly increasing list of times in [0, duration], rounded to the millisecond."""
    if not isinstance(value, list) or not value:
        return None
    out = []
    for v in value:
        t = _num(v)
        if t is None or t < 0 or t > (duration if duration is not None else FIRST_BEAT_RANGE[1] * 10):
            return None
        out.append(round(t, 3))
    return out if all(b > a for a, b in zip(out, out[1:])) else None


def sanitize_analysis(raw: Any, duration: float | None = None) -> dict | None:
    """Return a contract-conformant analysis dict, or None when nothing valid is left.

    `beats` / `downbeats` are all-or-nothing: an invalid list is dropped, never trimmed.
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}

    tempo = raw.get("tempo")
    if isinstance(tempo, dict):
        bpm = _num(tempo.get("bpm"))
        if bpm is not None and BPM_RANGE[0] <= bpm <= BPM_RANGE[1]:
            out["tempo"] = _with_conf({"bpm": round(bpm, 1)}, tempo)

    first_beat = _num(raw.get("first_beat"))
    if first_beat is not None and FIRST_BEAT_RANGE[0] <= first_beat <= FIRST_BEAT_RANGE[1]:
        out["first_beat"] = round(first_beat, 3)

    beats = _time_list(raw.get("beats"), duration)
    downbeats = _time_list(raw.get("downbeats"), duration)
    if beats and downbeats and set(downbeats) <= set(beats):
        out["beats"], out["downbeats"] = beats, downbeats
        conf = _confidence(raw.get("beats_confidence"))
        if conf is not None:
            out["beats_confidence"] = conf
        # Keep first_beat and the downbeats consistent: the first downbeat is the first beat of bar 1.
        if downbeats[0] <= FIRST_BEAT_RANGE[1]:
            out["first_beat"] = downbeats[0]
        else:
            out.pop("first_beat", None)
    if isinstance(raw.get("intro_free"), bool):
        out["intro_free"] = raw["intro_free"]

    ts = raw.get("time_signature")
    if isinstance(ts, dict):
        n, d = ts.get("numerator"), ts.get("denominator")
        if (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 16
                and isinstance(d, int) and not isinstance(d, bool) and d in DENOMINATORS):
            out["time_signature"] = _with_conf({"numerator": n, "denominator": d}, ts)

    key = raw.get("key")
    if isinstance(key, dict):
        name = key.get("name")
        if isinstance(name, str) and KEY_RE.match(name):
            out["key"] = _with_conf({"name": name}, key)

    sections = raw.get("sections")
    if isinstance(sections, list):
        clean = []
        for s in sections:
            if not isinstance(s, dict):
                continue
            start = _num(s.get("start"))
            label = s.get("label")
            if start is None or start < 0 or not isinstance(label, str):
                continue
            label = label.strip().lower()
            if label in SECTION_LABELS:
                clean.append({"start": round(start, 3), "label": label})
        clean.sort(key=lambda s: s["start"])
        if clean:
            out["sections"] = clean

    return out or None


def key_name(tonic_pc: int, minor: bool) -> str:
    return (MINOR_TONIC[tonic_pc] + "m") if minor else SHARP_TO_FLAT_TONIC[tonic_pc]


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #

class _Budget:
    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds

    def ok(self) -> bool:
        return time.monotonic() < self.deadline


def _find_stem(files: list[Path], name: str) -> Path | None:
    for p in files:
        if p.stem == name:
            return p
    return None


def _load(path: Path, sr: int):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def _mix(paths: list[Path], sr: int):
    import numpy as np
    ys = [_load(p, sr) for p in paths]
    n = min(len(y) for y in ys)
    return np.sum([y[:n] for y in ys], axis=0)


# --------------------------------------------------------------------------- #
# librosa backend
# --------------------------------------------------------------------------- #

_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _key(y_harm, sr: int) -> dict:
    """Krumhansl-Schmuckler key estimate on a chroma average."""
    import librosa
    import numpy as np

    tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr, n_fft=4096, hop_length=2048, tuning=tuning)
    profile = chroma.mean(axis=1)
    if not np.any(profile > 0):
        return {}
    candidates = []
    for minor, ref in ((False, _KS_MAJOR), (True, _KS_MINOR)):
        ref = np.asarray(ref)
        for tonic in range(12):
            r = float(np.corrcoef(profile, np.roll(ref, tonic))[0, 1])
            candidates.append((r, tonic, minor))
    candidates.sort(reverse=True)
    (r1, tonic, minor), (r2, _, _) = candidates[0], candidates[1]
    # Margin to the runner-up (often the relative major/minor) plus absolute fit.
    conf = 0.6 * min(1.0, (r1 - r2) / 0.08) + 0.4 * min(1.0, max(0.0, (r1 - 0.4) / 0.4))
    return {"name": key_name(tonic, minor), "confidence": min(1.0, max(0.0, conf))}


def _sections(y_mix, y_vocals, sr: int, first_downbeat: float | None, bar_len: float | None) -> list[dict]:
    """Heuristic structure: segment on beat-less chroma+MFCC, cluster, label by energy/repetition."""
    import librosa
    import numpy as np

    hop = 2048
    duration = len(y_mix) / sr
    if duration < 30:
        return []
    chroma = librosa.feature.chroma_stft(y=y_mix, sr=sr, n_fft=4096, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y_mix, sr=sr, n_mfcc=13, hop_length=hop)
    rms = librosa.feature.rms(y=y_mix, hop_length=hop)[0]
    # Smooth over ~1.5s so boundaries follow sections rather than notes.
    win = max(1, int(1.5 * sr / hop))
    kernel = np.ones(win) / win

    def smooth(m):
        return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, m)

    def z(m):
        return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)

    feats = np.vstack([z(smooth(chroma)), z(smooth(mfcc))])
    n = feats.shape[1]
    k = int(min(12, max(3, round(duration / 22))))
    k = min(k, n // 4)
    if k < 2:
        return []
    bounds = librosa.segment.agglomerative(feats, k)
    bounds = sorted({int(b) for b in bounds} | {0})
    starts = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)

    # Snap boundaries to the bar grid when we have one.
    if first_downbeat is not None and bar_len and bar_len > 0.5:
        starts = np.array([first_downbeat + round((t - first_downbeat) / bar_len) * bar_len for t in starts])
        starts = np.clip(starts, 0.0, duration - 1.0)
    starts[0] = 0.0
    keep = [0]
    for i in range(1, len(starts)):
        if starts[i] - starts[keep[-1]] >= 4.0:
            keep.append(i)
    starts = starts[keep]
    ends = np.append(starts[1:], duration)

    seg_vec, seg_rms, seg_voc = [], [], []
    voc_rms = librosa.feature.rms(y=y_vocals, hop_length=hop)[0] if y_vocals is not None else None
    for s, e in zip(starts, ends):
        a, b = int(s * sr / hop), max(int(s * sr / hop) + 1, int(e * sr / hop))
        seg_vec.append(feats[:, a:b].mean(axis=1))
        seg_rms.append(float(rms[a:b].mean()))
        if voc_rms is not None:
            seg_voc.append(float(voc_rms[a:b].mean()))
    seg_vec = np.asarray(seg_vec)
    m = len(seg_vec)

    # Group similar segments (cosine similarity) greedily.
    norm = seg_vec / (np.linalg.norm(seg_vec, axis=1, keepdims=True) + 1e-9)
    group = [-1] * m
    g = 0
    for i in range(m):
        if group[i] >= 0:
            continue
        group[i] = g
        for j in range(i + 1, m):
            if group[j] < 0 and float(norm[i] @ norm[j]) > 0.6:
                group[j] = g
        g += 1
    counts = {gi: group.count(gi) for gi in set(group)}
    mean_rms = {gi: float(np.mean([seg_rms[i] for i in range(m) if group[i] == gi])) for gi in counts}
    repeated = [gi for gi, c in counts.items() if c >= 2]
    chorus_g = max(repeated, key=lambda gi: (mean_rms[gi], counts[gi])) if repeated else None
    verse_candidates = [gi for gi in repeated if gi != chorus_g]
    verse_g = max(verse_candidates, key=lambda gi: (counts[gi], -mean_rms[gi])) if verse_candidates else None

    voc_floor = (0.25 * max(seg_voc)) if seg_voc else 0.0
    labels = []
    for i in range(m):
        gi = group[i]
        if i == 0 and (m > 2) and seg_rms[0] < 0.8 * np.mean(seg_rms) and gi not in (chorus_g,):
            label = "intro"
        elif i == m - 1 and m > 2 and (counts[gi] == 1 or seg_rms[i] < 0.8 * np.mean(seg_rms)):
            label = "outro"
        elif gi == chorus_g:
            label = "chorus"
        elif gi == verse_g:
            label = "verse"
        elif counts[gi] == 1:
            label = "bridge" if 0 < i < m - 1 else "verse"
        else:
            label = "verse"
        if seg_voc and seg_voc[i] < voc_floor and label in {"verse", "bridge"}:
            label = "inst"
        labels.append(label)

    out, prev = [], None
    for s, lab in zip(starts, labels):
        if lab == prev:
            continue
        out.append({"start": float(s), "label": lab})
        prev = lab
    return out


def _analyze_librosa(files: list[Path], source: Path | None, budget: AnalysisBudget) -> dict:
    sr = 22050
    drums, bass = _find_stem(files, "drums"), _find_stem(files, "bass")
    other, vocals = _find_stem(files, "other"), _find_stem(files, "vocals")
    no_vocals = _find_stem(files, "no_vocals")
    result: dict[str, Any] = {}
    mix_cache: list = []

    def full_mix():
        if not mix_cache:
            stems = [p for p in (drums, bass, other, vocals) if p] if sum(map(bool, (drums, bass, other, vocals))) >= 3 \
                else ([no_vocals, vocals] if no_vocals and vocals else None)
            if stems:
                mix_cache.append(_mix([p for p in stems if p], sr))
            elif source:
                mix_cache.append(_load(source, sr))
            else:
                raise RuntimeError("no audio available")
        return mix_cache[0]

    rhythm: dict = {}
    try:
        rp = [p for p in (drums, bass) if p]
        y = _mix(rp, sr) if rp else full_mix()
        harm_parts = [p for p in (bass, other) if p] or ([no_vocals] if no_vocals else [])
        rhythm = analyze_rhythm(full_mix(), y, sr, _load(drums, sr) if drums else None,
                                _mix(harm_parts, sr) if harm_parts else None)
        for k in ("tempo", "time_signature", "first_beat", "beats", "downbeats", "beats_confidence", "intro_free"):
            if k in rhythm:
                result[k] = rhythm[k]
        result["_duration"] = len(full_mix()) / sr
    except Exception:
        log.exception("analysis: rhythm step failed")

    if budget.ok():
        try:
            parts = [p for p in (bass, other) if p] or ([no_vocals] if no_vocals else [])
            y = _mix(parts, sr) if parts else full_mix()
            k = _key(y, sr)
            if k:
                result["key"] = k
        except Exception:
            log.exception("analysis: key step failed")
    else:
        log.warning("analysis: time budget exhausted before key")

    if budget.ok():
        try:
            y_voc = _load(vocals, sr) if vocals else None
            secs = _sections(full_mix(), y_voc, sr, rhythm.get("first_beat"), rhythm.get("_downbeat_period"))
            if secs:
                result["sections"] = secs
        except Exception:
            log.exception("analysis: sections step failed")
    else:
        log.warning("analysis: time budget exhausted before sections")
    return result


# --------------------------------------------------------------------------- #
# Optional All-In-One backend (opt-in; see README for licence caveats)
# --------------------------------------------------------------------------- #

_AIO_LABELS = {"start": None, "end": None, "intro": "intro", "verse": "verse", "chorus": "chorus",
               "bridge": "bridge", "inst": "inst", "solo": "solo", "break": "break", "outro": "outro"}


def _analyze_allin1(files: list[Path], source: Path | None, work_dir: Path, device: str, budget: AnalysisBudget) -> dict:
    import shutil

    import allin1  # type: ignore  # optional dependency

    stems = {p.stem: p for p in files if p.stem in {"bass", "drums", "other", "vocals"}}
    if source is None:
        raise RuntimeError("no source file")
    # All-In-One reuses Demucs stems when they sit in <demix_dir>/htdemucs/<track>/<stem>.wav.
    demix = work_dir / "allin1-demix"
    track_dir = demix / "htdemucs" / source.stem
    if len(stems) == 4 and all(p.suffix == ".wav" for p in stems.values()):
        track_dir.mkdir(parents=True, exist_ok=True)
        for name, p in stems.items():
            shutil.copy2(p, track_dir / f"{name}.wav")
    res = allin1.analyze(
        str(source), device=device if device in {"cpu", "cuda"} else "cpu",
        demix_dir=str(demix), spec_dir=str(work_dir / "allin1-spec"),
        out_dir=None, keep_byproducts=False,
    )
    res = res[0] if isinstance(res, list) else res
    out: dict[str, Any] = {}
    if res.bpm:
        out["tempo"] = {"bpm": float(res.bpm)}
    downbeats = [round(float(t), 3) for t in (res.downbeats or [])]
    beats = [round(float(t), 3) for t in (res.beats or [])]
    if beats and downbeats and set(downbeats) <= set(beats):
        out["beats"], out["downbeats"] = beats, downbeats
        out["first_beat"] = downbeats[0]
        if len(beats) > 2:
            import statistics
            out["tempo"] = {"bpm": 60.0 / statistics.median(b - a for a, b in zip(beats, beats[1:]))}
    elif downbeats:
        out["first_beat"] = downbeats[0]
    positions = list(res.beat_positions or [])
    if positions:
        out["time_signature"] = {"numerator": int(max(positions)), "denominator": 4}
    secs = []
    for seg in res.segments or []:
        label = _AIO_LABELS.get(str(seg.label).lower())
        if label:
            secs.append({"start": float(seg.start), "label": label})
    if secs:
        secs[0]["start"] = 0.0
        out["sections"] = secs
    shutil.rmtree(work_dir / "allin1-demix", ignore_errors=True)
    shutil.rmtree(work_dir / "allin1-spec", ignore_errors=True)
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

class LibrosaAnalyzer:
    """Adapt the existing stem-aware backend to the internal interface."""

    engine_id = "librosa"

    def analyze(self, inputs: AnalysisInput, budget: AnalysisBudget) -> dict[str, Any]:
        return _analyze_librosa(list(inputs.files), inputs.source, budget)


class AllInOneAnalyzer:
    """Optional backend; importing this adapter does not import allin1."""

    engine_id = "allin1"

    def analyze(self, inputs: AnalysisInput, budget: AnalysisBudget) -> dict[str, Any]:
        return _analyze_allin1(list(inputs.files), inputs.source, inputs.work_dir, inputs.device, budget)


def analyze_job(files: list[Path], source: Path | None, work_dir: Path, device: str = "cpu") -> dict | None:
    """Legacy entry point: same settings, best-effort behavior and sanitized result."""
    if not ANALYSIS_ENABLED:
        return None
    budget = _Budget(ANALYSIS_TIMEOUT)
    started = time.monotonic()
    inputs = AnalysisInput(tuple(files), source, work_dir, device)
    adapter = LegacyAnalysisAdapter(
        LibrosaAnalyzer(), sanitize_analysis,
        AllInOneAnalyzer() if ANALYSIS_BACKEND == "allin1" else None,
    )
    result = adapter.analyze(inputs, budget)
    log.info("analysis done in %.1fs: fields=%s", time.monotonic() - started, sorted(result or {}))
    return result
