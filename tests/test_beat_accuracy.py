"""Beats and bar 1 to the millisecond (issue #19): a click played with the recording must land on
the hits, and its 1 on the song's 1, so the synthetic songs' beats are checked within 10 ms."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

CASES = [
    dict(bpm=72, lead=0.0),
    dict(bpm=72, lead=0.37),
    dict(bpm=96, lead=1.13),
    dict(bpm=120, lead=0.61),
    dict(bpm=134, lead=2.05),                 # was counted at 89 (3:2) with the 1 two seconds late
    dict(bpm=84, lead=0.0),
    dict(bpm=140, lead=0.2, subdivide=4),     # was half time, its beats on the snare
    dict(bpm=66, lead=0.8, meter=3),
    dict(bpm=110, lead=0.3, intro_bars=0),    # no intro: the very first hit is bar 1
    dict(bpm=150, lead=1.0),                  # was half time, its 1 on a 2
    dict(bpm=78, lead=0.5, seed=3),
    dict(bpm=126, lead=0.0, subdivide=1),
    # Found by a random sweep (180 songs, all within 10 ms now):
    dict(bpm=125, lead=2.37, intro_bars=0, subdivide=4, seed=43),     # was 83 (3:2)
    dict(bpm=62, lead=0.73, intro_bars=3, subdivide=4, seed=24),      # was 82 (4:3)
    dict(bpm=60, lead=0.75, subdivide=2, meter=3, seed=4),            # was double time, in 4/4
    dict(bpm=175, lead=1.1, intro_bars=0, seed=2),                    # tracker 50 ms early throughout
    dict(bpm=163, lead=0.62, intro_bars=1, seed=4),                   # bar 1 a beat late
    dict(bpm=140, lead=1.73, intro_bars=3, subdivide=4, seed=46),     # pad intro taken for a free one
    dict(bpm=91, lead=2.45, intro_bars=1, subdivide=4, seed=17),      # last beats on the hi-hats
]


def _analyse(tmp_path, case):
    librosa = pytest.importorskip("librosa")
    import synth
    from rhythm import analyze_rhythm

    kw = dict(intro_bars=2, bars=16) | case
    synth.make_song(tmp_path, **kw)
    ys = {n: librosa.load(str(tmp_path / f"{n}.wav"), sr=synth.SR)[0] for n in ("drums", "bass", "other", "vocals")}
    out = analyze_rhythm(sum(ys.values()), ys["drums"] + ys["bass"], synth.SR, ys["drums"], ys["other"] + ys["bass"])
    return kw, out


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()))
def test_beats_and_bar_one_within_10_ms(tmp_path, case):
    kw, out = _analyse(tmp_path, case)
    beat = 60.0 / kw["bpm"]
    lead = kw["lead"]
    meter = kw.get("meter", 4)

    assert abs(out["tempo"]["bpm"] - kw["bpm"]) < 0.5
    assert out["time_signature"]["numerator"] == meter
    # Bar 1 is where the music starts (the pad intro is in time), to the millisecond it's rounded to.
    assert abs(out["first_beat"] - lead) <= 0.010
    beats = np.array(out["beats"])
    n = np.round((beats - lead) / beat)
    assert np.all(np.diff(n) == 1), "a beat missing or doubled"
    assert np.max(np.abs(beats - (lead + n * beat))) <= 0.010
    downs = np.array(out["downbeats"])
    assert np.all(np.round((downs - lead) / beat) % meter == 0), "a downbeat off the 1"


@pytest.mark.parametrize("seed,bpm,free", [(1, 100, 14), (3, 120, 10), (5, 140, 9), (7, 110, 11)])
def test_free_intro_then_bar_one_within_10_ms(tmp_path, seed, bpm, free):
    librosa = pytest.importorskip("librosa")
    import synth
    from rhythm import analyze_rhythm

    first, _ = synth.make_song(tmp_path, bpm=bpm, free_intro=float(free), bars=14, seed=seed)
    ys = {n: librosa.load(str(tmp_path / f"{n}.wav"), sr=synth.SR)[0] for n in ("drums", "bass", "other", "vocals")}
    out = analyze_rhythm(sum(ys.values()), ys["drums"] + ys["bass"], synth.SR, ys["drums"], ys["other"] + ys["bass"])
    assert out["intro_free"] is True
    assert abs(out["first_beat"] - first) <= 0.010


@pytest.mark.parametrize("eighths,lead,seed,hats", [(138, 0.6, 5, True), (150, 1.2, 6, True), (120, 0.0, 7, True),
                                                    (165, 0.4, 8, True), (142, 0.4, 9, False), (126, 1.0, 10, False)])
def test_six_eight_counted_in_eighths(tmp_path, eighths, lead, seed, hats):
    """A 6/8 song (kick on 1, snare on 4) heard as 3/4 has the right bar lines but the wrong
    beats: it's counted in its six eighths, its tempo in eighths as API.md says (the beats'). Without
    hi-hats, the kick and snare alone make a 4/4 half-time bar too: the keys' eighths tell."""
    librosa = pytest.importorskip("librosa")
    import synth
    from rhythm import analyze_rhythm

    first, e = synth.make_compound(tmp_path, eighths_bpm=eighths, lead=lead, seed=seed, hats=hats)
    ys = {n: librosa.load(str(tmp_path / f"{n}.wav"), sr=synth.SR)[0] for n in ("drums", "bass", "other", "vocals")}
    out = analyze_rhythm(sum(ys.values()), ys["drums"] + ys["bass"], synth.SR, ys["drums"], ys["other"] + ys["bass"])
    assert (out["time_signature"]["numerator"], out["time_signature"]["denominator"]) == (6, 8)
    assert abs(out["tempo"]["bpm"] - eighths) < 0.5
    assert abs(out["first_beat"] - first) <= 0.010
    beats = np.array(out["beats"])
    n = np.round((beats - first) / e)
    assert np.all(np.diff(n) == 1)
    assert np.max(np.abs(beats - (first + n * e))) <= 0.010
    assert np.all(np.round((np.array(out["downbeats"]) - first) / e) % 6 == 0)
