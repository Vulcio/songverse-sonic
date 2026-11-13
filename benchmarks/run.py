"""Benchmarks for beats, bars and chords on annotated recordings (issue #9).

Not part of the test suite: run it by hand, on a set of songs kept outside Git.

    pip install mir_eval            # MIT; torch too for the chord engines
    python benchmarks/run.py --synthetic                      # built-in cases, no data needed
    python benchmarks/run.py /path/to/set --chords btc btc-chordmini --chords-input mix harmonic
    pip install beat_this           # MIT code and weights; the checkpoint downloads from JKU on first use
    python benchmarks/run.py /path/to/set --chords --beats sonic beat_this --beats-input mix drums

A set is a folder with one folder per song:

    song-name/
      mix.wav | mix.mp3 | mix.flac          the recording (or the stems below, or both)
      drums.wav bass.wav other.wav vocals.wav   optional Demucs stems (used like Sonic uses them)
      click.wav                             optional: a multitrack's click stem, read as the true beats
                                            and bars when annotations.json has none (truth_from_click)
      annotations.json                      what is true, every part optional (optional with a click):
        {"beats": [s, ...], "downbeats": [s, ...], "time_signature": 4,
         "chords": [{"start": s, "end": s, "label": "C:min7"}, ...],   (Harte labels, as mir_eval)
         "sections": [{"start": s, "label": "verse"}, ...],
         "notes": "free text: who annotated it, what is uncertain"}

The mix must not hold the click: export it from the multitrack without the click stem, or give the
other stems. A click that also clicks subdivisions shows up as tempo ratio 0.5.

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
# Truth from a click track
# --------------------------------------------------------------------------- #

def truth_from_click(path: Path, music_starts: float | None = None) -> dict:
    """The beats and bars a multitrack's click stem plays (issue #9): real songs as their band plays
    them, without annotating by hand. Each click's time is its attack (the first sample reaching
    half its peak); the downbeats are the clicks set apart by a higher pitch or a louder level, when
    one is (a click that accents nothing gives beats only). Clicks before `music_starts` (a count-in)
    are dropped, keeping half a beat of slack."""
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    a = np.abs(y)
    if not len(a) or a.max() <= 0:
        return {}
    hop = 32
    frames = librosa.util.frame(np.pad(a, (0, hop)), frame_length=hop, hop_length=hop).max(axis=0)
    on = frames > 0.1 * frames.max()
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    clicks: list[tuple[float, float, float]] = []   # (time, peak, pitch)
    gap = int(0.06 * sr)
    for f in starts:
        s = f * hop
        if clicks and s / sr - clicks[-1][0] < gap / sr:
            continue                                  # the same click ringing on
        seg = y[max(0, s - hop): s + int(0.03 * sr)]
        peak = float(np.abs(seg).max())
        attack = max(0, s - hop) + int(np.flatnonzero(np.abs(seg) >= 0.5 * peak)[0])
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=8192))
        pitch = float(np.argmax(spec[8:]) + 8) * sr / 8192
        clicks.append((attack / sr, peak, pitch))
    if len(clicks) < 8:
        return {}
    times = np.array([c[0] for c in clicks])
    if music_starts is not None:
        beat = float(np.median(np.diff(times)))
        keep = times >= music_starts - beat / 2
        clicks = [c for c, k in zip(clicks, keep) if k]
        times = times[keep]
    out: dict = {"beats": [round(float(t), 4) for t in times]}

    def two_groups(values: np.ndarray, ratio: float) -> np.ndarray | None:
        """A mask of the rarer of two clearly separate groups, or None."""
        order = np.sort(np.log(values))
        jumps = np.diff(order)
        i = int(np.argmax(jumps)) if len(jumps) else 0
        if not len(jumps) or jumps[i] < np.log(ratio):
            return None
        high = np.log(values) > order[i]
        return high if high.sum() <= (~high).sum() else ~high

    for values, ratio in ((np.array([c[2] for c in clicks]), 1.08), (np.array([c[1] for c in clicks]), 1.25)):
        accents = two_groups(values, ratio)
        if accents is None:
            continue
        idx = np.flatnonzero(accents)
        spacing = np.diff(idx)
        if len(spacing) == 0:
            break
        meter = int(np.bincount(spacing).argmax())
        if meter >= 2 and np.mean(spacing == meter) >= 0.8:
            out["downbeats"] = [round(float(times[i]), 4) for i in idx]
            out["time_signature"] = meter
        break
    return out


def music_start(paths: list[Path], sr: int = 22050) -> float | None:
    """When the music (anything but the click) first sounds: within 40 dB of its loudest moment."""
    import librosa

    if not paths:
        return None
    y = sum(librosa.load(p, sr=sr, mono=True)[0] for p in paths)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    if not len(rms) or rms.max() <= 0:
        return None
    first = int(np.flatnonzero(rms >= rms.max() * 0.01)[0])
    return first * 256 / sr


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


class BeatThis:
    """Beat This! (Foscarin et al., ISMIR 2024; code and weights MIT), a candidate beat and downbeat
    engine for #9 and #18, run here only, never by the server. Its own minimal post-processing:
    madmom's DBN is left out, its models being CC BY-NC-SA."""

    def __init__(self, checkpoint: str):
        from beat_this.inference import Audio2Beats  # pip install beat_this

        self.checkpoint = checkpoint
        self.run = Audio2Beats(checkpoint_path=checkpoint, device="cpu", dbn=False)

    @property
    def name(self) -> str:
        path = Path(self.checkpoint)
        return f"beat_this@{sha256(path)[:12] if path.exists() else self.checkpoint}"

    def __call__(self, y: np.ndarray, sr: int) -> dict:
        beats, downbeats = self.run(y, sr)
        beats, downbeats = [float(b) for b in beats], [float(d) for d in downbeats]
        found: dict = {"beats": beats, "downbeats": downbeats}
        if downbeats:
            found["first_beat"] = downbeats[0]
            per_bar = [sum(1 for b in beats if a - 0.02 <= b < z - 0.02) for a, z in zip(downbeats, downbeats[1:])]
            if per_bar:
                found["time_signature"] = {"numerator": int(np.bincount(per_bar).argmax())}
        return found


def candidate_input(which: str, mix: Path | None, stems: list[Path], sr: int) -> np.ndarray | None:
    import librosa

    if which == "drums":
        drums = [p for p in stems if p.stem == "drums"]
        return librosa.load(drums[0], sr=sr, mono=True)[0] if drums else None
    if mix:
        return librosa.load(mix, sr=sr, mono=True)[0]
    return sum(librosa.load(p, sr=sr, mono=True)[0] for p in stems) if stems else None


def run_song(folder: Path, chord_engines: list[str], chord_inputs: list[str],
             beat_engines: list | None = None, beat_inputs: list[str] = ("mix",)) -> dict:
    import librosa

    notes = folder / "annotations.json"
    truth = json.loads(notes.read_text()) if notes.exists() else {}
    mix = find_audio(folder, "mix")
    stems = [p for p in (find_audio(folder, s) for s in STEMS) if p]
    files = stems if stems else []
    click = find_audio(folder, "click")
    if click and not truth.get("beats"):
        heard = truth_from_click(click, music_start([mix] if mix else stems))
        truth = {**heard, **truth, "truth": "click"}
    inputs = {p.name: sha256(p) for p in ([mix] if mix else []) + stems + ([click] if click else [])}
    report: dict = {"song": folder.name, "inputs": inputs, "notes": truth.get("notes"), "truth": truth.get("truth", "annotations")}

    t0, cpu0 = time.perf_counter(), time.process_time()
    raw = analysis._analyze_librosa(files, mix, Budget())
    report["rhythm_seconds"] = round(time.perf_counter() - t0, 2)
    report["rhythm_cpu_seconds"] = round(time.process_time() - cpu0, 2)
    duration = raw.pop("_duration", None)
    found = analysis.sanitize_analysis(raw, duration) or {}
    report["rhythm"] = rhythm_metrics(found, truth)
    report["found"] = {k: found.get(k) for k in ("tempo", "time_signature", "first_beat", "key")}

    # Other beat engines on the same song, on the mix and on the drums stem (#18: does isolating the
    # drums help?), scored the same way. Sonic itself is the "rhythm" above.
    report["beats"] = {}
    for engine in beat_engines or []:
        for which in beat_inputs:
            y = candidate_input(which, mix, stems, 22050)
            if y is None:
                continue
            t0, cpu0 = time.perf_counter(), time.process_time()
            try:
                other = engine(y, 22050)
            except Exception as exc:  # report the failure, keep going
                report["beats"][f"{engine.name}/{which}"] = {"error": str(exc)[:300]}
                continue
            result = rhythm_metrics(other, truth)
            result["seconds"] = round(time.perf_counter() - t0, 2)
            result["cpu_seconds"] = round(time.process_time() - cpu0, 2)
            report["beats"][f"{engine.name.split('@')[0]}/{which}"] = result

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
        if name == "pop-4-4":
            # The same song as a multitrack with a click stem: its beats and bars are read from the click.
            from synth import make_click
            make_click(folder / "click.wav", beats, beats[::meter], duration=first + (kw["bars"] + 1) * bar)
            (folder / "annotations.json").write_text(json.dumps({
                "chords": truth_chords, "notes": "synthetic, beats and bars from its click (tests/synth.py)"}))
        else:
            (folder / "annotations.json").write_text(json.dumps({
                "beats": beats, "downbeats": beats[::meter], "time_signature": meter, "chords": truth_chords,
                "notes": "synthetic (tests/synth.py make_progression)"}))
        # A mix as well, for the mix input.
        y = sum(sf.read(folder / f"{s}.wav")[0] for s in STEMS)
        sf.write(folder / "mix.wav", y.astype("float32"), 22050)
    return dest


def table(reports: list[dict]) -> str:
    keys = sorted({k for r in reports for k in r["chords"]})
    beat_keys = sorted({k for r in reports for k in r.get("beats", {})})
    head = ["song", "beat F", "downbeat F", "1st downbeat err (s)", "tempo ratio"]
    head += [f"{k} beat F / downbeat F / 1st err" for k in beat_keys]
    head += [f"{k} majmin / per-beat / downbeat F after" for k in keys]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * (5 + len(beat_keys) + len(keys))]
    for r in reports:
        rh = r["rhythm"]
        cells = [r["song"], f'{rh.get("beat_f", float("nan")):.2f}', f'{rh.get("downbeat_f", float("nan")):.2f}',
                 str(rh.get("first_downbeat_error", "-")), str(rh.get("tempo_ratio", "-"))]
        for k in beat_keys:
            b = r.get("beats", {}).get(k, {})
            cells.append("error" if "error" in b else
                         f'{b.get("beat_f", float("nan")):.2f} / {b.get("downbeat_f", float("nan")):.2f} / '
                         f'{b.get("first_downbeat_error", "-")}')
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
    parser.add_argument("--beats", nargs="*", default=["sonic"], choices=["sonic", "beat_this"],
                        help="beat engines to compare (Sonic always runs)")
    parser.add_argument("--beats-input", nargs="*", default=["mix", "drums"], choices=["mix", "drums"])
    parser.add_argument("--beat-this-checkpoint", default="final0",
                        help="Beat This! checkpoint: a name (downloaded from JKU) or a local .ckpt")
    parser.add_argument("--out", type=Path, help="where to write the JSON report")
    args = parser.parse_args(argv)
    if not args.set and not args.synthetic:
        parser.error("give a set folder or --synthetic")
    beat_engines = []
    if "beat_this" in args.beats:
        try:
            beat_engines.append(BeatThis(args.beat_this_checkpoint))
        except Exception as exc:
            parser.error(f"Beat This! couldn't load its checkpoint {args.beat_this_checkpoint!r} ({exc}). "
                         "Install it with pip install beat_this; without access to cloud.cp.jku.at, "
                         "download the .ckpt elsewhere and give its path.")

    with tempfile.TemporaryDirectory() as tmp:
        folder = synthetic_set(Path(tmp)) if args.synthetic else args.set
        songs = sorted(p for p in folder.iterdir()
                       if (p / "annotations.json").exists() or find_audio(p, "click"))
        reports = []
        for song in songs:
            print(f"… {song.name}", file=sys.stderr)
            reports.append(run_song(song, args.chords, args.chords_input, beat_engines, args.beats_input))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report = {
        "sonic_version": analysis.SONIC_VERSION,
        "engines": {"rhythm": f"librosa@{analysis.SONIC_VERSION}",
                    **{e: f"{e}@{chord_engine.CHECKPOINTS[e].sha256[:12]}" for e in args.chords
                       if e in chord_engine.CHECKPOINTS},
                    **{e.name.split("@")[0]: e.name for e in beat_engines}},
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
