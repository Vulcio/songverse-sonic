"""The benchmark's own reading of a song (issue #9, #18): truth from a click stem, other beat engines
scored alongside Sonic's."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run  # noqa: E402
import synth  # noqa: E402


def grid(first, bpm, meter, bars):
    beat = 60 / bpm
    beats = [round(first + i * beat, 4) for i in range(bars * meter)]
    return beats, beats[::meter]


@pytest.mark.parametrize("accent,meter,bpm", [("pitch", 4, 96), ("level", 3, 132), ("pitch", 6, 70)])
def test_beats_and_bars_from_a_click(tmp_path, accent, meter, bpm):
    beats, downbeats = grid(1.37, bpm, meter, 12)
    synth.make_click(tmp_path / "click.wav", beats, downbeats, accent=accent)
    truth = run.truth_from_click(tmp_path / "click.wav")
    assert len(truth["beats"]) == len(beats)
    assert np.max(np.abs(np.array(truth["beats"]) - beats)) <= 0.001     # the attack, to the ms
    assert truth["downbeats"] == pytest.approx(downbeats, abs=0.001)
    assert truth["time_signature"] == meter


def test_a_click_without_accents_gives_beats_only(tmp_path):
    beats, downbeats = grid(0.5, 100, 4, 8)
    synth.make_click(tmp_path / "click.wav", beats, downbeats, accent=None)
    truth = run.truth_from_click(tmp_path / "click.wav")
    assert len(truth["beats"]) == len(beats) and "downbeats" not in truth


def test_a_count_in_before_the_music_is_left_out(tmp_path):
    beats, downbeats = grid(0.5, 120, 4, 10)            # bar 1 is a count-in, the music starts at bar 2
    synth.make_click(tmp_path / "click.wav", beats, downbeats)
    truth = run.truth_from_click(tmp_path / "click.wav", music_starts=downbeats[1] + 0.01)
    assert truth["beats"][0] == pytest.approx(downbeats[1], abs=0.001)
    assert truth["downbeats"][0] == pytest.approx(downbeats[1], abs=0.001)


class Steady:
    """A stand-in beat engine that hears a steady grid, 20 ms late."""
    name = "steady@test"

    def __init__(self, beats, downbeats):
        self.beats, self.downbeats = beats, downbeats

    def __call__(self, y, sr):
        return {"beats": [b + 0.02 for b in self.beats], "downbeats": [d + 0.02 for d in self.downbeats],
                "first_beat": self.downbeats[0] + 0.02}


def test_a_song_with_a_click_is_scored_for_every_engine_and_input(tmp_path):
    song = tmp_path / "song"
    song.mkdir()
    first, bar = synth.make_song(song, bpm=100, intro_bars=0, bars=12, lead=3.0)
    beats, downbeats = grid(first, 100, 4, 12)
    synth.make_click(song / "click.wav", [first - 4 * 0.6 + i * 0.6 for i in range(4)] + beats,
                     [first - 4 * 0.6] + downbeats, duration=first + 13 * bar)       # with a count-in bar
    report = run.run_song(song, [], [], [Steady(beats, downbeats)], ["mix", "drums"])
    assert report["truth"] == "click"
    assert set(report["beats"]) == {"steady/mix", "steady/drums"}
    for result in report["beats"].values():
        assert result["beat_f"] == 1.0 and result["first_downbeat_error"] == pytest.approx(0.02, abs=0.002)
    assert report["rhythm"]["beat_f"] > 0.9                               # Sonic's own, against the click
    assert "beat F / downbeat F" in run.table([report])


def test_annotations_win_over_the_click(tmp_path):
    song = tmp_path / "song"
    song.mkdir()
    first, bar = synth.make_song(song, bpm=90, intro_bars=0, bars=8, lead=0.5)
    beats, downbeats = grid(first, 90, 4, 8)
    synth.make_click(song / "click.wav", [b + 0.3 for b in beats], [d + 0.3 for d in downbeats])
    (song / "annotations.json").write_text(json.dumps({"beats": beats, "downbeats": downbeats}))
    report = run.run_song(song, [], [])
    assert report["truth"] == "annotations" and report["rhythm"]["beat_f"] > 0.9
