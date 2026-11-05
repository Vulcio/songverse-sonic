"""Chords on beats and bar lines from chord changes (issue #15)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import analysis
import chords
from synth import SR, make_progression


def seg(start, end, root, quality="maj"):
    return {"start": start, "end": end, "root": root, "quality": quality,
            "label": chords.harte_label(root, quality)}


def test_vocabulary():
    assert chords.chord_of(0) == (0, "min")
    assert chords.chord_of(1) == (0, "maj")
    assert chords.chord_of(14 * 9 + 9) == (9, "7")
    assert chords.chord_of(chords.NO_CHORD) is None and chords.chord_of(chords.UNKNOWN) is None
    assert chords.harte_label(1, "maj") == "C#" and chords.harte_label(9, "min7") == "A:min7"


def test_short_runs_join_their_longer_neighbour():
    C, G = 1, 14 * 7 + 1
    frames = [C] * 20 + [G] * 2 + [C] * 10 + [G] * 30 + [chords.NO_CHORD] * 10
    out = chords.segments_from_frames(frames, frame=0.1)
    assert [c["label"] for c in out] == ["C", "G", "N"]
    assert out[0]["start"] == 0 and out[0]["end"] == pytest.approx(3.2)
    assert out[1]["end"] == pytest.approx(6.2)


def test_chord_on_each_beat_is_the_one_sounding_longest():
    found = [seg(0, 1.9, 0), seg(1.9, 4.0, 7)]
    assert chords.chords_per_beat(found, [0.0, 1.0, 2.0, 3.0], 4.0) == [0, 0, 1, 1]
    assert chords.chords_per_beat(found, [0.0, 1.0, 2.0, 3.0, 5.0], 6.0)[-1] is None


def changes_every_bar(meter, phase, bars=12, beat=0.5, offset=0):
    beats = [round(i * beat, 3) for i in range(bars * meter)]
    found = []
    roots = [0, 9, 5, 7]
    starts = [beats[i] for i in range(phase, len(beats), meter)]
    edges = [0.0] + starts[1:] + [len(beats) * beat] if phase == 0 else [0.0] + starts + [len(beats) * beat]
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        found.append(seg(a, b, roots[(k + offset) % 4]))
    return beats, found


def test_bars_move_to_where_the_chords_change():
    beats, found = changes_every_bar(4, phase=2)
    downbeats = beats[0::4]  # the kick put "1" two beats early
    phase = chords.bar_phase_from_chords(found, beats, downbeats, 4)
    assert phase == 2
    moved = chords.apply_bar_phase({"beats": beats, "downbeats": downbeats, "first_beat": beats[0]}, phase, 4)
    assert moved["first_beat"] == beats[2] and moved["downbeats"] == beats[2::4]
    assert moved["beats"] == beats


def test_bars_kept_when_chords_agree_or_say_little():
    beats, found = changes_every_bar(4, phase=0)
    assert chords.bar_phase_from_chords(found, beats, beats[0::4], 4) is None
    # Too few changes to say anything.
    assert chords.bar_phase_from_chords(found[:3], beats, beats[0::4], 4) is None
    # Changes everywhere (busy harmony): no phase clearly wins.
    busy = [seg(b, b + 0.5, i % 12) for i, b in enumerate(beats)]
    assert chords.bar_phase_from_chords(busy, beats, beats[0::4], 4) is None


def test_waltz_bars():
    beats, found = changes_every_bar(3, phase=1)
    assert chords.bar_phase_from_chords(found, beats, beats[0::3], 3) == 1


def test_sanitize_keeps_valid_chords_and_drops_invalid_lists():
    good = [seg(0.0, 2.0, 0), {"start": 2.0, "end": 3.0, "label": "N"}, seg(3.0, 5.0, 9, "min7") | {"bass": 4}]
    beats = {"beats": [0.0, 1.0, 2.0, 3.0, 4.0], "downbeats": [0.0, 4.0]}
    out = analysis.sanitize_analysis({**beats, "chords": good, "beat_chords": [0, 0, 1, 2, 2]}, 5.0)
    assert out["chords"] == good and out["beat_chords"] == [0, 0, 1, 2, 2]
    for bad in ([seg(0, 2, 0), seg(1.5, 3, 7)],          # overlapping
                [{"start": 0, "end": 2, "root": 12, "quality": "maj", "label": "?"}],  # root out of range
                [seg(0, 2, 0, "maj9")],                   # unknown quality
                [seg(0, 9, 0)]):                          # past the end
        assert "chords" not in (analysis.sanitize_analysis({**beats, "chords": bad}, 5.0) or {})
    # beat_chords only with beats, of the same length, pointing at chords
    assert "beat_chords" not in analysis.sanitize_analysis({"chords": good, "beat_chords": [0]}, 5.0)
    assert "beat_chords" not in analysis.sanitize_analysis({**beats, "chords": good, "beat_chords": [0, 0, 1, 2, 9]}, 5.0)


def test_engines_only_for_parts_present():
    out = analysis.sanitize_analysis({"key": {"name": "G"}, "engines": {
        "key": "librosa@1.7.0", "chords": "btc@1673d23f8f9a", "rhythm": "librosa@1.7.0", "other": "x@1"}})
    assert out["engines"] == {"key": "librosa@1.7.0"}


# --------------------------------------------------------------------------- #
# With the real model (needs torch and the checkpoint; skipped where they aren't)
# --------------------------------------------------------------------------- #

def _model_available():
    try:
        import torch  # noqa: F401
        return chords.checkpoint_path("btc") is not None
    except Exception:
        return False


needs_model = pytest.mark.skipif(not _model_available(), reason="torch or the BTC checkpoint unavailable")

PROGRESSION = [(0, "maj"), (9, "min"), (5, "maj"), (7, "maj")]  # C Am F G


def beat_accuracy(found, beats, expected_per_beat):
    per_beat = chords.chords_per_beat(found, beats)
    hits = 0
    for c, (root, quality) in zip(per_beat, expected_per_beat):
        if c is not None and found[c].get("root") == root and found[c].get("quality") == quality:
            hits += 1
    return hits / len(expected_per_beat)


@needs_model
def test_btc_finds_a_simple_progression(tmp_path):
    first, bar, beat = make_progression(tmp_path, PROGRESSION, bpm=90, bars=16)
    y = sum(np.asarray(__import__("soundfile").read(tmp_path / f"{n}.wav")[0]) for n in ("bass", "other"))
    found = chords.recognize(y, SR, "btc")
    beats = [first + i * beat for i in range(16 * 4)]
    expected = [PROGRESSION[(i // 4) % 4] for i in range(len(beats))]
    assert beat_accuracy(found, beats, expected) >= 0.8


@needs_model
def test_job_analysis_with_chords_moves_bar_one_to_the_chord_changes(tmp_path, monkeypatch):
    # Kick accents on beat 3 would put "1" there; the chords change on the real 1.
    first, bar, beat = make_progression(tmp_path, PROGRESSION, bpm=96, bars=16, kick_on=2)
    monkeypatch.setattr(analysis, "ANALYSIS_CHORDS", "btc")
    files = [tmp_path / f"{n}.wav" for n in ("drums", "bass", "other", "vocals")]
    a = analysis.analyze_job(files, None, tmp_path)
    assert a["chords"] and a["engines"]["chords"].startswith("btc@")
    assert len(a["beat_chords"]) == len(a["beats"])
    # bar 1 on a bar line of the progression (within a third of a beat)
    offset = (a["first_beat"] - first) / bar
    assert abs(offset - round(offset)) * bar < beat / 3
    for d in a["downbeats"][1:6]:
        k = (d - first) / bar
        assert abs(k - round(k)) * bar < beat / 3
