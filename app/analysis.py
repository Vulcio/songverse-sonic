"""Music analysis (tempo, beats, meter, key, sections, chords) for finished fast passes.

Everything here is best effort: every field of the result is optional and a
failure of any step only leaves its fields out. `analyze_job` never raises.

Default backend: librosa (ISC licence, no trained model files) plus small
heuristics. Optional backend: All-In-One (`ANALYSIS_BACKEND=allin1`), see README.
Optional chords (`ANALYSIS_CHORDS=btc`): see chords.py.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from rhythm import analyze_rhythm

log = logging.getLogger("demucs-api.analysis")

SONIC_VERSION = (Path(__file__).with_name("VERSION").read_text().strip()
                 if Path(__file__).with_name("VERSION").exists() else "dev")
ANALYSIS_ENABLED = os.getenv("ANALYSIS_ENABLED", "true").lower() == "true"
ANALYSIS_BACKEND = os.getenv("ANALYSIS_BACKEND", "librosa").lower()
ANALYSIS_TIMEOUT = float(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "60"))
# Past this the analysis process is stopped (issue #7): ANALYSIS_TIMEOUT_SECONDS is a budget its
# steps check between them, this is the hard limit for a step that never returns.
ANALYSIS_HARD_LIMIT = float(os.getenv("ANALYSIS_HARD_LIMIT_SECONDS", str(max(120.0, 3 * ANALYSIS_TIMEOUT))))
# Chords (issue #15): off by default; "btc" or "btc-chordmini" (see README "Licences").
ANALYSIS_CHORDS = os.getenv("ANALYSIS_CHORDS", "off").lower()
# What the chord engine listens to: the whole mix, or the mix without drums and vocals.
MAX_LYRICS_BYTES = 64 * 1024   # a job's lyrics (issue #22)
ANALYSIS_CHORDS_INPUT = os.getenv("ANALYSIS_CHORDS_INPUT", "mix").lower()

BPM_RANGE = (20.0, 400.0)
FIRST_BEAT_RANGE = (0.0, 600.0)
DENOMINATORS = {2, 4, 8, 16}
SECTION_LABELS = {
    "intro", "verse", "pre-chorus", "chorus", "bridge", "inst", "instrumental",
    "solo", "break", "interlude", "outro", "tag",
}
KEY_RE = re.compile(r"^[A-G][#b]?m?$")
CHORD_QUALITIES = {"maj", "min", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7", "dim7",
                   "hdim7", "sus2", "sus4"}
ENGINE_PARTS = {"rhythm", "key", "sections", "chords"}
ENGINE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,40}@[A-Za-z0-9._+-]{1,40}$")

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


def _score(value: Any) -> float | None:
    x = _num(value)
    return round(x, 2) if x is not None and 0.0 <= x <= 1.0 else None


def _alternatives(raw: Any, out: dict) -> dict:
    """`alternatives` (issue #25): for each part the analysis has, up to 4 other values it nearly
    picked, each valid as the part itself is and with a 0-1 `score`, the best first."""
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, list] = {}

    def keep(part: str, items: Any, make) -> None:
        if part not in out or not isinstance(items, list):
            return
        seen, values = set(), []
        for item in items:
            if not isinstance(item, dict):
                continue
            score = _score(item.get("score"))
            value = make(item) if score is not None else None
            if value is None:
                continue
            ident = tuple(sorted(value.items()))
            if ident in seen:
                continue
            seen.add(ident)
            values.append({**value, "score": score})
        values.sort(key=lambda v: -v["score"])
        if values:
            clean[part] = values[:4]

    def tempo(item):
        bpm = _num(item.get("bpm"))
        return {"bpm": round(bpm, 1)} if bpm is not None and BPM_RANGE[0] <= bpm <= BPM_RANGE[1] \
            and round(bpm, 1) != out["tempo"]["bpm"] else None

    def meter(item):
        n, d = item.get("numerator"), item.get("denominator")
        ok = (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 16 and isinstance(d, int)
              and not isinstance(d, bool) and d in DENOMINATORS)
        now = out["time_signature"]
        return {"numerator": n, "denominator": d} if ok and (n, d) != (now["numerator"], now["denominator"]) else None

    def first(item):
        start = _num(item.get("start"))
        return {"start": round(start, 3)} if start is not None and FIRST_BEAT_RANGE[0] <= start <= FIRST_BEAT_RANGE[1] \
            and round(start, 3) != out["first_beat"] else None

    def key(item):
        name = item.get("name")
        return {"name": name} if isinstance(name, str) and KEY_RE.match(name) and name != out["key"]["name"] else None

    keep("tempo", raw.get("tempo"), tempo)
    keep("time_signature", raw.get("time_signature"), meter)
    keep("first_beat", raw.get("first_beat"), first)
    keep("key", raw.get("key"), key)
    return clean


HINT_KEYS = {"bpm", "time_signature", "first_beat", "key"}


def read_hints(raw: Any) -> dict:
    """What the user says about a recording (issue #25): `{"rhythm": {...}, "key": "Gm"}` for the
    analysis, from the API's `hints` object. Raises ValueError, with a message for the client, on
    anything it can't take."""
    if not isinstance(raw, dict):
        raise ValueError("hints must be an object")
    unknown = set(raw) - HINT_KEYS
    if unknown:
        raise ValueError(f"hints has unknown fields: {', '.join(sorted(unknown))}")
    rhythm: dict[str, Any] = {}
    out: dict[str, Any] = {}
    if raw.get("bpm") is not None:
        bpm = _num(raw["bpm"])
        if bpm is None or not BPM_RANGE[0] <= bpm <= BPM_RANGE[1]:
            raise ValueError(f"hints.bpm must be a number from {BPM_RANGE[0]} to {BPM_RANGE[1]}")
        rhythm["bpm"] = float(bpm)
    if raw.get("time_signature") is not None:
        ts = raw["time_signature"]
        if isinstance(ts, str) and re.fullmatch(r"\d{1,2}/\d{1,2}", ts.strip()):
            n, d = (int(x) for x in ts.strip().split("/"))
        elif isinstance(ts, dict):
            n, d = ts.get("numerator"), ts.get("denominator")
        else:
            n = d = None
        if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 16 and d in DENOMINATORS):
            raise ValueError('hints.time_signature must be like "6/8": 1 to 16 over 2, 4, 8 or 16')
        rhythm["time_signature"] = (n, d)
    if raw.get("first_beat") is not None:
        first = _num(raw["first_beat"])
        if first is None or not FIRST_BEAT_RANGE[0] <= first <= FIRST_BEAT_RANGE[1]:
            raise ValueError(f"hints.first_beat must be seconds from {FIRST_BEAT_RANGE[0]} to {FIRST_BEAT_RANGE[1]}")
        rhythm["first_beat"] = float(first)
    if raw.get("key") is not None:
        if not isinstance(raw["key"], str) or not KEY_RE.match(raw["key"]):
            raise ValueError('hints.key must be a key like "G" or "Bbm"')
        out["key"] = raw["key"]
    if rhythm:
        out["rhythm"] = rhythm
    return out


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

    chords = _chord_list(raw.get("chords"), duration)
    if chords:
        out["chords"] = chords
        per_beat = raw.get("beat_chords")
        if ("beats" in out and isinstance(per_beat, list) and len(per_beat) == len(out["beats"])
                and all(c is None or (isinstance(c, int) and not isinstance(c, bool) and 0 <= c < len(chords))
                        for c in per_beat)):
            out["beat_chords"] = per_beat

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
                section = {"start": round(start, 3), "label": label}
                # Since 1.9 (issue #22): which pass of the client's lyrics it is, and its name there.
                hint = s.get("hint")
                if isinstance(hint, int) and not isinstance(hint, bool) and 0 <= hint < 1000:
                    section["hint"] = hint
                    name = s.get("name")
                    if isinstance(name, str) and name.strip():
                        section["name"] = name.strip()[:80]
                clean.append(section)
        clean.sort(key=lambda s: s["start"])
        if clean:
            out["sections"] = clean

    alternatives = _alternatives(raw.get("alternatives"), out)
    if alternatives:
        out["alternatives"] = alternatives

    engines = raw.get("engines")
    if out and isinstance(engines, dict):
        present = {"rhythm": "tempo" in out or "beats" in out or "time_signature" in out,
                   "key": "key" in out, "sections": "sections" in out, "chords": "chords" in out}
        clean_engines = {part: name for part, name in engines.items()
                         if part in ENGINE_PARTS and present[part] and isinstance(name, str) and ENGINE_RE.match(name)}
        if clean_engines:
            out["engines"] = clean_engines

    return out or None


def _chord_list(value: Any, duration: float | None) -> list[dict] | None:
    """Chords in order, not overlapping, within the recording; all or nothing, like beats."""
    if not isinstance(value, list) or not value:
        return None
    limit = (duration + 0.5) if duration is not None else FIRST_BEAT_RANGE[1] * 10
    out: list[dict] = []
    for c in value:
        if not isinstance(c, dict):
            return None
        start, end = _num(c.get("start")), _num(c.get("end"))
        if start is None or end is None or start < 0 or end <= start or end > limit:
            return None
        start, end = round(start, 3), round(min(end, duration) if duration is not None else end, 3)
        if end <= start or (out and start < out[-1]["end"]):
            return None
        label = c.get("label")
        if label == "N":
            out.append({"start": start, "end": end, "label": "N"})
            continue
        root, quality, bass = c.get("root"), c.get("quality"), c.get("bass")
        if (not isinstance(root, int) or isinstance(root, bool) or not 0 <= root <= 11
                or quality not in CHORD_QUALITIES or not isinstance(label, str) or not label):
            return None
        chord = {"start": start, "end": end, "root": root, "quality": quality, "label": label}
        if bass is not None:
            if not isinstance(bass, int) or isinstance(bass, bool) or not 0 <= bass <= 11:
                return None
            if bass != root:
                chord["bass"] = bass
        out.append(chord)
    return out


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
    # What was nearly picked (issue #25): the next keys, scored like the confidence against the one
    # after them - the relative major or minor is usually first.
    alternatives = []
    for i, (r, t, m) in enumerate(candidates[1:4], start=1):
        nxt = candidates[i + 1][0]
        score = 0.6 * min(1.0, (r - nxt) / 0.08) + 0.4 * min(1.0, max(0.0, (r - 0.4) / 0.4))
        score = round(min(1.0, max(0.0, min(score, conf))), 2)
        if score >= 0.2:
            alternatives.append({"name": key_name(t, m), "score": score})
    return {"name": key_name(tonic, minor), "confidence": min(1.0, max(0.0, conf)), "_alternatives": alternatives}


def _bar_novelty(F, downbeats, sr: int, hop: int):
    """Each whole bar's mean features, how much the music changes at each bar line (the 4 bars
    before against the 4 after, plus half of 2 against 2), and the level of a clear change (robust:
    one huge change doesn't hide the rest). None with fewer than 16 bars or no change at all."""
    import numpy as np

    frames = np.round(np.asarray(downbeats) * sr / hop).astype(int)
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 0
    frames = frames[frames + step <= F.shape[1]]          # whole bars only: not the silence at the end
    if len(frames) < 16:
        return None
    ends = np.append(frames[1:], frames[-1] + step)
    bars = np.array([F[:, a:max(a + 1, b)].mean(axis=1) for a, b in zip(frames, ends)])
    nb = len(bars)

    def novelty(w):
        out = np.zeros(nb)
        for k in range(1, nb):
            lo, hi = bars[max(0, k - w):k], bars[k:min(nb, k + w)]
            out[k] = float(np.linalg.norm(lo.mean(axis=0) - hi.mean(axis=0)))
        return out

    nov = novelty(4) + 0.5 * novelty(2)
    nov[0] = 0.0
    if nov.max() <= 0:
        return None
    usual = float(np.median(nov[1:]))
    level = usual + 1.4826 * float(np.median(np.abs(nov[1:] - usual)))
    return bars, nov, max(level, 1e-9)


def _bar_boundaries(F, downbeats, sr: int, hop: int):
    """Section starts at bar lines where the music changes most (seconds, the first one 0.0), or
    None when there are too few bars to judge.

    Each bar is summed up by its mean features. A bar line's novelty compares the 4 bars before
    it with the 4 after (and 2 with 2, for short sections). Peaks at least 4 bars apart, clearly
    above the song's usual change, are section starts. Between two bar lines that change about as
    much, the one that keeps sections a multiple of 4 bars long wins: songs move in phrases.
    """
    import numpy as np

    found = _bar_novelty(F, downbeats, sr, hop)
    if found is None:
        return None
    bars, nov, level = found
    nb = len(bars)
    peaks = [k for k in range(1, nb)
             if nov[k] >= level and nov[k] == nov[max(1, k - 2):min(nb, k + 3)].max()]
    chosen = [0]
    for k in sorted(peaks, key=lambda k: -nov[k]):
        if all(abs(k - c) >= 4 for c in chosen):
            chosen.append(k)
    chosen.sort()
    # Phrases: move a start by one bar when that bar changes nearly as much and the section
    # before it then lasts a multiple of 4 bars.
    for i in range(1, len(chosen)):
        k, prev = chosen[i], chosen[i - 1]
        if (k - prev) % 4 == 0:
            continue
        for alt in (k - 1, k + 1):
            nxt = chosen[i + 1] if i + 1 < len(chosen) else nb + 4
            if 1 <= alt < nb and (alt - prev) % 4 == 0 and alt - prev >= 4 and nxt - alt >= 4 \
                    and nov[alt] >= 0.85 * nov[k]:
                chosen[i] = alt
                break
    starts = np.array([downbeats[k] for k in chosen], dtype=float)
    starts[0] = 0.0
    return starts


def _bar_chords(chords: list[dict] | None, downbeats, end: float) -> list[set] | None:
    """For each bar, the chords (root, minor) heard over at least a quarter of it."""
    if not chords:
        return None
    out = []
    for k, start in enumerate(downbeats):
        stop = downbeats[k + 1] if k + 1 < len(downbeats) else end
        heard = set()
        for c in chords:
            if c.get("label") == "N" or "root" not in c:
                continue
            overlap = min(stop, c["end"]) - max(start, c["start"])
            if overlap >= 0.25 * (stop - start):
                heard.add((int(c["root"]), str(c.get("quality", "")).startswith(("min", "dim", "hdim"))))
        out.append(heard)
    return out


def _sections_from_lyrics(passes, F, y_vocals, sr: int, hop: int, downbeats, chords) -> list[dict] | None:
    """The client's sections placed on the bars (issue #22), or None when they don't fit."""
    import librosa
    import numpy as np

    import lyrics

    found = _bar_novelty(F, downbeats, sr, hop)
    if found is None:
        return None
    bars, nov, level = found
    nb = len(bars)
    db = np.asarray(downbeats, dtype=float)
    bar_len = float(np.median(np.diff(db)))
    voice = None
    if y_vocals is not None:
        rms = librosa.feature.rms(y=y_vocals, hop_length=hop)[0]
        frames = np.round(db[:nb] * sr / hop).astype(int)
        ends = np.append(frames[1:], frames[-1] + int(round(bar_len * sr / hop)))
        per_bar = np.array([rms[a:max(a + 1, b)].mean() if a < len(rms) else 0.0 for a, b in zip(frames, ends)])
        if per_bar.max() > 1e-3:
            voice = np.clip(per_bar / (0.4 * per_bar.max()), 0.0, 1.0)
    return lyrics.align(passes, bars, nov / level, db[:nb], bar_len, voice,
                        _bar_chords(chords, list(db[:nb]), float(db[nb - 1] + bar_len)))


def _sections(y_mix, y_vocals, sr: int, first_downbeat: float | None, bar_len: float | None,
              downbeats: list[float] | None = None, passes=None, chords=None) -> list[dict]:
    """Heuristic structure: segment on beat-less chroma+MFCC, cluster, label by energy/repetition.
    With the client's `passes` (issue #22), its own sections placed on the bars when they fit."""
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
    if passes and downbeats and len(downbeats) >= 16:
        try:
            placed = _sections_from_lyrics(passes, np.vstack([z(chroma), z(mfcc), z(rms[None, :])]), y_vocals,
                                           sr, hop, downbeats, chords)
            if placed:
                return placed
            log.info("analysis: the lyrics' sections don't fit the recording; finding them without")
        except Exception:
            log.exception("analysis: placing the lyrics' sections failed")
    bar_starts = None
    if downbeats and len(downbeats) >= 16:
        # Sections start on bar lines, so look for them there: how much the music changes from the
        # bars before each bar line to the bars after it (issue #21).
        bar_starts = _bar_boundaries(np.vstack([z(chroma), z(mfcc), z(rms[None, :])]), np.asarray(downbeats), sr, hop)
    if bar_starts is not None:
        starts = bar_starts
    else:
        bounds = librosa.segment.agglomerative(feats, k)
        bounds = sorted({int(b) for b in bounds} | {0})
        starts = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)

    # Snap boundaries to bar lines: the detected downbeats when there are any (they follow the
    # music when its tempo moves), else a steady grid from the first downbeat.
    if bar_starts is not None:
        starts = np.clip(starts, 0.0, duration - 1.0)
    elif downbeats and len(downbeats) >= 2:
        bars = np.asarray(downbeats)
        starts = np.array([bars[int(np.argmin(np.abs(bars - t)))] for t in starts])
        starts = np.clip(starts, 0.0, duration - 1.0)
    elif first_downbeat is not None and bar_len and bar_len > 0.5:
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


def _analyze_librosa(files: list[Path], source: Path | None, budget: _Budget, passes=None, hints=None) -> dict:
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
                                _mix(harm_parts, sr) if harm_parts else None, (hints or {}).get("rhythm"))
        for k in ("tempo", "time_signature", "first_beat", "beats", "downbeats", "beats_confidence", "intro_free"):
            if k in rhythm:
                result[k] = rhythm[k]
        if rhythm.get("_alternatives"):
            result["alternatives"] = dict(rhythm["_alternatives"])
        result["_duration"] = len(full_mix()) / sr
    except Exception:
        log.exception("analysis: rhythm step failed")

    if budget.ok():
        try:
            parts = [p for p in (bass, other) if p] or ([no_vocals] if no_vocals else [])
            y = _mix(parts, sr) if parts else full_mix()
            k = {"name": hints["key"], "confidence": 1.0} if (hints or {}).get("key") else _key(y, sr)
            alternatives = k.pop("_alternatives", None)
            if k:
                result["key"] = k
            if alternatives:
                result.setdefault("alternatives", {})["key"] = alternatives
        except Exception:
            log.exception("analysis: key step failed")
    else:
        log.warning("analysis: time budget exhausted before key")

    if ANALYSIS_CHORDS != "off":
        if budget.ok():
            try:
                harmonic = [p for p in (bass, other) if p] or ([no_vocals] if no_vocals else [])
                y = _mix(harmonic, sr) if ANALYSIS_CHORDS_INPUT == "harmonic" and harmonic else full_mix()
                result.update(_chords_on_beats(y, sr, result))
            except Exception:
                log.exception("analysis: chords step failed")
        else:
            log.warning("analysis: time budget exhausted before chords")

    if budget.ok():
        try:
            y_voc = _load(vocals, sr) if vocals else None
            secs = _sections(full_mix(), y_voc, sr, result.get("first_beat"), rhythm.get("_downbeat_period"),
                             result.get("downbeats"), passes, result.get("chords"))
            if secs:
                result["sections"] = secs
        except Exception:
            log.exception("analysis: sections step failed")
    else:
        log.warning("analysis: time budget exhausted before sections")
    result["engines"] = {part: f"librosa@{SONIC_VERSION}" for part in ("rhythm", "key", "sections")}
    if any("hint" in s for s in result.get("sections", [])):
        result["engines"]["sections"] = f"librosa-lyrics@{SONIC_VERSION}"
    if "chords" in result:
        result["engines"]["chords"] = _chord_engine_id()
    return result


def _chord_engine_id() -> str:
    import chords
    spec = chords.CHECKPOINTS.get(ANALYSIS_CHORDS)
    return f"{ANALYSIS_CHORDS}@{spec.sha256[:12]}" if spec else f"{ANALYSIS_CHORDS}@unknown"


def _chords_on_beats(y, sr: int, rhythm: dict) -> dict:
    """Chords (issue #15), the chord on each beat, and the bar lines moved where the chord changes
    clearly put them. Returns the fields to add or replace; empty when no chord was found."""
    import chords

    found = chords.recognize(y, sr, ANALYSIS_CHORDS)
    if not any(c["label"] != "N" for c in found):
        return {}
    out: dict[str, Any] = {"chords": found}
    beats, downbeats = rhythm.get("beats"), rhythm.get("downbeats")
    if beats and downbeats:
        meter = (rhythm.get("time_signature") or {}).get("numerator") or 4
        phase = chords.bar_phase_from_chords(found, beats, downbeats, meter)
        if phase is not None:
            moved = chords.apply_bar_phase(rhythm, phase, meter)
            log.info("analysis: chord changes moved bar 1 from %.3f s to %.3f s",
                     rhythm["first_beat"], moved["first_beat"])
            out["downbeats"], out["first_beat"] = moved["downbeats"], moved["first_beat"]
        out["beat_chords"] = chords.chords_per_beat(found, beats, len(y) / sr)
    return out


# --------------------------------------------------------------------------- #
# Optional All-In-One backend (opt-in; see README for licence caveats)
# --------------------------------------------------------------------------- #

_AIO_LABELS = {"start": None, "end": None, "intro": "intro", "verse": "verse", "chorus": "chorus",
               "bridge": "bridge", "inst": "inst", "solo": "solo", "break": "break", "outro": "outro"}


def _analyze_allin1(files: list[Path], source: Path | None, work_dir: Path, device: str, budget: _Budget) -> dict:
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
    if secs and out.get("downbeats"):
        # On its own bar lines, like the built-in engine's: a section is looped and cued by bars.
        bars = out["downbeats"]
        for sec in secs:
            sec["start"] = min(bars, key=lambda t: abs(t - sec["start"]))
        secs = [sec for i, sec in enumerate(secs) if i == 0 or sec["start"] > secs[i - 1]["start"]]
    if secs:
        secs[0]["start"] = 0.0
        out["sections"] = secs
    shutil.rmtree(work_dir / "allin1-demix", ignore_errors=True)
    shutil.rmtree(work_dir / "allin1-spec", ignore_errors=True)
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _child(queue, target, args) -> None:
    try:
        queue.put(("ok", target(*args)))
    except BaseException as exc:  # reported to the parent, which logs it
        queue.put(("error", repr(exc)))


def run_with_hard_limit(target, args: tuple, limit: float):
    """`target(*args)` in a child process; None when it raises or isn't done within `limit` s,
    in which case the child is killed. The result must pickle."""
    import multiprocessing
    import queue as queues

    ctx = multiprocessing.get_context("spawn")
    channel = ctx.Queue()
    process = ctx.Process(target=_child, args=(channel, target, args), daemon=True)
    process.start()
    try:
        status, value = channel.get(timeout=limit)
    except queues.Empty:
        log.error("analysis: no result after %.0fs; stopping it", limit)
        process.kill()
        process.join(5)
        return None
    finally:
        channel.close()
    process.join(5)
    if status != "ok":
        log.error("analysis: failed in its process: %s", value)
        return None
    return value


def analyze_job_with_limit(files: list[Path], source: Path | None, work_dir: Path, device: str = "cpu",
                           lyrics_text: str | None = None, hints: dict | None = None) -> dict | None:
    """analyze_job in its own process, stopped at ANALYSIS_HARD_LIMIT_SECONDS. Never raises."""
    if not ANALYSIS_ENABLED:
        return None
    try:
        return run_with_hard_limit(analyze_job, (list(files), source, work_dir, device, lyrics_text, hints),
                                   ANALYSIS_HARD_LIMIT)
    except Exception:
        log.exception("analysis: could not start its process")
        return None


def analyze_job(files: list[Path], source: Path | None, work_dir: Path, device: str = "cpu",
                lyrics_text: str | None = None, hints: dict | None = None) -> dict | None:
    """Analyse a job after its fast pass. Never raises; returns a sanitised dict or None.
    `lyrics_text`: the client's lyrics (issue #22), to find its own sections. `hints`: what the
    user says (issue #25), as read by read_hints."""
    if not ANALYSIS_ENABLED:
        return None
    passes = None
    if lyrics_text:
        try:
            import lyrics
            passes = lyrics.parse(lyrics_text) or None
        except Exception:
            log.exception("analysis: the lyrics couldn't be read")
    budget = _Budget(ANALYSIS_TIMEOUT)
    started = time.monotonic()
    raw: dict[str, Any] = {}
    duration = None
    try:
        if ANALYSIS_BACKEND == "allin1":
            try:
                raw = _analyze_allin1(files, source, work_dir, device, budget)
            except Exception:
                log.exception("analysis: allin1 backend failed; falling back to librosa")
        hints = read_hints(hints) if hints else None
        if hints and ANALYSIS_BACKEND == "allin1":
            raw = {}                      # the user's values decide the beats: librosa's tracker takes them
        fallback = _analyze_librosa(files, source, budget, passes, hints)
        duration = fallback.pop("_duration", None)
        # Keep All-In-One's values where it had them; fill the rest from librosa. Beats and the
        # fields derived from them travel together, so never mix the two trackers' versions.
        beat_keys = {"tempo", "first_beat", "beats", "downbeats", "beats_confidence", "intro_free", "beat_chords",
                     "alternatives"}
        engines = fallback.pop("engines", {})
        if raw:
            primary = f"allin1@{SONIC_VERSION}"
            if "beats" in raw or "tempo" in raw:
                engines["rhythm"] = primary
            for part in ("key", "sections"):
                if part in raw:
                    engines[part] = primary
            # The client's own sections, placed on the bars, win over All-In-One's guess.
            if any("hint" in s for s in fallback.get("sections", [])):
                raw.pop("sections", None)
                engines["sections"] = f"librosa-lyrics@{SONIC_VERSION}"
        if "beats" in raw:
            fallback = {k: v for k, v in fallback.items() if k not in beat_keys}
        for k, v in fallback.items():
            if k not in raw:
                raw[k] = v
        raw["engines"] = engines
    except Exception:
        log.exception("analysis failed")
    result = sanitize_analysis(raw, duration)
    log.info("analysis done in %.1fs: fields=%s", time.monotonic() - started, sorted(result or {}))
    return result
