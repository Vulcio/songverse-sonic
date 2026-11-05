"""Benchmarks for beats, bars and chords on annotated recordings (issue #9).

Not part of the test suite: run it by hand, on a set of songs kept outside Git.

    pip install mir_eval            # MIT; torch too for the chord engines
    python benchmarks/run.py --synthetic                      # built-in cases, no data needed
    python benchmarks/run.py /path/to/set --chords btc btc-chordmini --chords-input mix harmonic

A set is a folder with one folder per song:

    song-name/
      mix.wav | mix.mp3 | mix.flac          the recording (or the stems below, or both)
      drums.wav bass.wav other.wav vocals.wav   optional Demucs stems (used like Sonic uses them)
      annotations.json                      what is true, every part optional:
        {"beats": [s, ...], "downbeats": [s, ...], "time_signature": 4,
         "chords": [{"start": s, "end": s, "label": "C:min7"}, ...],   (Harte labels, as mir_eval)
         "sections": [{"start": s, "label": "verse"}, ...],
         "notes": "free text: who annotated it, what is uncertain"}

The report (JSON, plus a Markdown table on stdout) names Sonic's version, each engine and
checkpoint, and every input's sha256, so two runs can be compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tests"))

import analysis  # noqa: E402
import chords as chord_engine  # noqa: E402

AUDIO = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
STEMS = ("drums", "bass", "other", "vocals")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def find_audio(folder: Path, name: str) -> Path | None:
    for ext in AUDIO:
        if (folder / f"{name}{ext}").exists():
            return folder / f"{name}{ext}"
    return None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def rhythm_metrics(found: dict, truth: dict) -> dict:
    import mir_eval

    out: dict = {}
    if truth.get("beats"):
        ref = np.asarray(truth["beats"])
        est = np.asarray(found.get("beats") or [])
        out["beat_f"] = float(mir_eval.beat.f_measure(ref, est)) if len(est) else 0.0
        if len(est) > 1 and len(ref) > 1:
            ratio = float(np.median(np.diff(ref)) / np.median(np.diff(est)))
            out["tempo_ratio"] = round(ratio, 3)  # 2 = Sonic counts half time, 0.5 = double time
        if len(est):
            # Continuity (CMLt / AMLt): how much of the song is tracked without a break.
            _, _, cmlt, amlt = mir_eval.beat.continuity(ref, est)
            out["beat_cmlt"], out["beat_amlt"] = float(cmlt), float(amlt)
    if truth.get("downbeats"):
        ref = np.asarray(truth["downbeats"])
        est = np.asarray(found.get("downbeats") or [])
        out["downbeat_f"] = float(mir_eval.beat.f_measure(ref, est)) if len(est) else 0.0
        if found.get("first_beat") is not None:
            out["first_downbeat_error"] = round(abs(found["first_beat"] - float(ref[0])), 3)
    if truth.get("time_signature") and found.get("time_signature"):
        out["meter_ok"] = found["time_signature"]["numerator"] == truth["time_signature"]
    if truth.get("sections"):
        starts = [s["start"] for s in found.get("sections") or []]
        if starts:
            errors = [min(abs(s["start"] - t) for t in starts) for s in truth["sections"]]
            out["section_start_error_median"] = round(float(np.median(errors)), 3)
            out["sections_within_1s"] = round(sum(e <= 1.0 for e in errors) / len(errors), 3)
    return out


def chord_metrics(found: list[dict], truth: dict, duration: float, beats: list[float] | None) -> dict:
    import mir_eval

    if not truth.get("chords"):
        return {}
    ref_int = np.array([[c["start"], c["end"]] for c in truth["chords"]])
    ref_lab = [c["label"] for c in truth["chords"]]
    est_int = np.array([[c["start"], c["end"]] for c in found]) if found else np.array([[0.0, duration]])
    est_lab = [c["label"] for c in found] if found else ["N"]
    est_int, est_lab = mir_eval.util.adjust_intervals(est_int, est_lab, ref_int.min(), ref_int.max(),
                                                      mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    scores = mir_eval.chord.evaluate(ref_int, ref_lab, est_int, est_lab)
    out = {k: round(float(scores[k]), 3) for k in ("root", "majmin", "sevenths", "mirex", "seg")}
    # What Songverse would show: the chord on each beat of the truth's beats (or Sonic's).
    grid = truth.get("beats") or beats
    if grid and found:
        ref_segs = [{"start": c["start"], "end": c["end"], "label": c["label"]} for c in truth["chords"]]
        r = chord_engine.chords_per_beat(ref_segs, grid, duration)
        e = chord_engine.chords_per_beat(found, grid, duration)
        same = [ref_segs[a]["label"] == found[b]["label"] if a is not None and b is not None else a is b
                for a, b in zip(r, e)]
        roots = []
        for a, b in zip(r, e):
            if a is None or b is None:
                continue
            ref_root = mir_eval.chord.encode(ref_segs[a]["label"])[0]
            est_root = found[b].get("root", -1)
            roots.append(ref_root == est_root)
        out["per_beat_exact"] = round(sum(same) / len(same), 3)
        out["per_beat_root"] = round(sum(roots) / len(roots), 3) if roots else 0.0
    return out


# --------------------------------------------------------------------------- #
# Running Sonic's analysis
# --------------------------------------------------------------------------- #

class Budget:
    def ok(self):
        return True


def run_song(folder: Path, chord_engines: list[str], chord_inputs: list[str]) -> dict:
    import librosa

    truth = json.loads((folder / "annotations.json").read_text())
    mix = find_audio(folder, "mix")
    stems = [p for p in (find_audio(folder, s) for s in STEMS) if p]
    files = stems if stems else []
    inputs = {p.name: sha256(p) for p in ([mix] if mix else []) + stems}
    report: dict = {"song": folder.name, "inputs": inputs, "notes": truth.get("notes")}

    t0, cpu0 = time.perf_counter(), time.process_time()
    raw = analysis._analyze_librosa(files, mix, Budget())
    report["rhythm_seconds"] = round(time.perf_counter() - t0, 2)
    report["rhythm_cpu_seconds"] = round(time.process_time() - cpu0, 2)
    duration = raw.pop("_duration", None)
    found = analysis.sanitize_analysis(raw, duration) or {}
    report["rhythm"] = rhythm_metrics(found, truth)
    report["found"] = {k: found.get(k) for k in ("tempo", "time_signature", "first_beat", "key")}

    report["chords"] = {}
    if chord_engines:
        sr = 22050
        y_mix = (sum(librosa.load(p, sr=sr, mono=True)[0] for p in stems) if len(stems) >= 3
                 else librosa.load(mix, sr=sr, mono=True)[0])
        harm = [p for p in stems if p.stem in ("bass", "other")]
        y_harm = sum(librosa.load(p, sr=sr, mono=True)[0] for p in harm) if harm else None
        for engine in chord_engines:
            for which in chord_inputs:
                y = y_harm if which == "harmonic" else y_mix
                if y is None:
                    continue
                t0 = time.perf_counter()
                try:
                    segs = chord_engine.recognize(y, sr, engine)
                except Exception as exc:  # report the failure, keep going
                    report["chords"][f"{engine}/{which}"] = {"error": str(exc)[:300]}
                    continue
                result = chord_metrics(segs, truth, len(y) / sr, found.get("beats"))
                result["seconds"] = round(time.perf_counter() - t0, 2)
                # Bar lines corrected by the chord changes, scored like the rhythm.
                if found.get("beats") and found.get("downbeats") and truth.get("downbeats"):
                    meter = (found.get("time_signature") or {}).get("numerator") or 4
                    phase = chord_engine.bar_phase_from_chords(segs, found["beats"], found["downbeats"], meter)
                    moved = chord_engine.apply_bar_phase(found, phase, meter) if phase is not None else found
                    after = rhythm_metrics(moved, {"downbeats": truth["downbeats"]})
                    result["bar_phase_moved"] = phase is not None
                    result["downbeat_f_after_chords"] = after.get("downbeat_f")
                    result["first_downbeat_error_after_chords"] = after.get("first_downbeat_error")
                report["chords"][f"{engine}/{which}"] = result
    return report


# --------------------------------------------------------------------------- #
# Built-in synthetic cases (known truth; a smoke test, not evidence about real music)
# --------------------------------------------------------------------------- #

def synthetic_set(dest: Path) -> Path:
    from synth import QUALITY_INTERVALS, make_progression, make_song  # noqa: F401
    import soundfile as sf

    cases = {
        "pop-4-4": dict(progression=[(0, "maj"), (9, "min"), (5, "maj"), (7, "maj")], bpm=96, bars=16),
        "waltz-3-4": dict(progression=[(7, "maj"), (2, "maj"), (4, "min"), (0, "maj")], bpm=120, bars=16, meter=3),
        "sevenths": dict(progression=[(2, "min7"), (7, "7"), (0, "maj7"), (9, "min7")], bpm=84, bars=16),
        "kick-on-3": dict(progression=[(0, "maj"), (5, "maj"), (9, "min"), (7, "maj")], bpm=100, bars=16, kick_on=2),
    }
    for name, kw in cases.items():
        folder = dest / name
        folder.mkdir(parents=True, exist_ok=True)
        meter = kw.get("meter", 4)
        first, bar, beat = make_progression(folder, **kw)
        n_beats = kw["bars"] * meter
        beats = [round(first + i * beat, 3) for i in range(n_beats)]
        prog = kw["progression"]
        truth_chords = [{"start": round(first + b * bar, 3), "end": round(first + (b + 1) * bar, 3),
                         "label": chord_engine.harte_label(*prog[b % len(prog)])} for b in range(kw["bars"])]
        (folder / "annotations.json").write_text(json.dumps({
            "beats": beats, "downbeats": beats[::meter], "time_signature": meter, "chords": truth_chords,
            "notes": "synthetic (tests/synth.py make_progression)"}))
        # A mix as well, for the mix input.
        y = sum(sf.read(folder / f"{s}.wav")[0] for s in STEMS)
        sf.write(folder / "mix.wav", y.astype("float32"), 22050)
    return dest


def table(reports: list[dict]) -> str:
    keys = sorted({k for r in reports for k in r["chords"]})
    lines = ["| song | beat F | downbeat F | 1st downbeat err (s) | tempo ratio | " +
             " | ".join(f"{k} majmin / per-beat / downbeat F after" for k in keys) + " |",
             "|" + "---|" * (5 + len(keys))]
    for r in reports:
        rh = r["rhythm"]
        cells = [r["song"], f'{rh.get("beat_f", float("nan")):.2f}', f'{rh.get("downbeat_f", float("nan")):.2f}',
                 str(rh.get("first_downbeat_error", "-")), str(rh.get("tempo_ratio", "-"))]
        for k in keys:
            c = r["chords"].get(k, {})
            if "error" in c:
                cells.append("error")
            else:
                after = c.get("downbeat_f_after_chords")
                cells.append(f'{c.get("majmin", float("nan")):.2f} / {c.get("per_beat_exact", float("nan")):.2f} / '
                             + (f"{after:.2f}" if after is not None else "-"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", nargs="?", type=Path, help="folder of annotated songs")
    parser.add_argument("--synthetic", action="store_true", help="run the built-in synthetic cases")
    parser.add_argument("--chords", nargs="*", default=["btc"], help="chord engines (empty for none)")
    parser.add_argument("--chords-input", nargs="*", default=["mix", "harmonic"], choices=["mix", "harmonic"])
    parser.add_argument("--out", type=Path, help="where to write the JSON report")
    args = parser.parse_args(argv)
    if not args.set and not args.synthetic:
        parser.error("give a set folder or --synthetic")

    with tempfile.TemporaryDirectory() as tmp:
        folder = synthetic_set(Path(tmp)) if args.synthetic else args.set
        songs = sorted(p for p in folder.iterdir() if (p / "annotations.json").exists())
        reports = []
        for song in songs:
            print(f"… {song.name}", file=sys.stderr)
            reports.append(run_song(song, args.chords, args.chords_input))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report = {
        "sonic_version": analysis.SONIC_VERSION,
        "engines": {"rhythm": f"librosa@{analysis.SONIC_VERSION}",
                    **{e: f"{e}@{chord_engine.CHECKPOINTS[e].sha256[:12]}" for e in args.chords
                       if e in chord_engine.CHECKPOINTS}},
        "set": "synthetic" if args.synthetic else str(args.set),
        "peak_memory_mb": round(peak_mb),
        "songs": reports,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    print(table(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
