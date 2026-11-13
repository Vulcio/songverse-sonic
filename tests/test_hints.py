"""What was nearly picked, and what the user says (issue #25): `alternatives` with the analysis,
`hints` on a job, kept as given while the rest is found around them."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import analysis  # noqa: E402


def rhythm_of(tmp_path, make, hints=None, **kw):
    librosa = pytest.importorskip("librosa")
    import synth
    from rhythm import analyze_rhythm

    first, unit = make(tmp_path, **kw)
    ys = {n: librosa.load(str(tmp_path / f"{n}.wav"), sr=synth.SR)[0] for n in ("drums", "bass", "other", "vocals")}
    out = analyze_rhythm(sum(ys.values()), ys["drums"] + ys["bass"], synth.SR, ys["drums"], ys["other"] + ys["bass"],
                         hints)
    return first, unit, out


def compound(path, **kw):
    import synth
    return synth.make_compound(path, **kw)


def simple(path, **kw):
    import synth
    first, bar = synth.make_song(path, **kw)
    return first, bar


def test_six_eight_offers_the_three_four_it_was_heard_as(tmp_path):
    _, _, out = rhythm_of(tmp_path, compound, eighths_bpm=142, lead=0.4, seed=9, hats=False)
    assert (out["time_signature"]["numerator"], out["time_signature"]["denominator"]) == (6, 8)
    alts = out["_alternatives"]
    assert {(m["numerator"], m["denominator"]) for m in alts.get("time_signature", [])} & {(3, 4), (4, 4)}
    assert any(abs(t["bpm"] - 71) < 1 for t in alts["tempo"])                  # its quarters
    for part in alts.values():
        assert all(a["score"] < 1.0 for a in part)


@pytest.mark.parametrize("meter,beats_per_bar", [((6, 8), 6), ((3, 4), 3), ((12, 8), 12)])
def test_a_time_signature_hint_keeps_the_bar_lines(tmp_path, meter, beats_per_bar):
    """Told 3/4 or 12/8, a 6/8 song keeps its bar lines (two bars for 12/8), counted in the given beats."""
    first, e, out = rhythm_of(tmp_path, compound, hints={"time_signature": meter}, eighths_bpm=138, lead=0.6, seed=5)
    bar = 6 * e * (2 if meter == (12, 8) else 1)
    assert (out["time_signature"]["numerator"], out["time_signature"]["denominator"]) == meter
    assert out["time_signature"]["confidence"] == 1.0
    downs = np.array(out["downbeats"])
    assert abs(downs[0] - first) <= 0.010
    assert np.max(np.abs(np.diff(downs) - bar)) <= 0.020
    assert len(out["beats"]) == pytest.approx(beats_per_bar * (len(downs) - 1) + 1, abs=1)
    assert abs(out["tempo"]["bpm"] - beats_per_bar * 60 / bar) < 1.0
    assert "time_signature" not in out["_alternatives"]


def test_a_six_eight_hint_on_a_song_heard_as_four_four(tmp_path):
    """6/8 at the dotted quarter (a 162 eighths a minute song tracked at 54) is the case Sonic
    gets wrong on its own: the user's 6/8 gives its eighths."""
    kw = dict(eighths_bpm=162, lead=0.3, seed=1, hats=False)
    first, e, plain = rhythm_of(tmp_path, compound, **kw)
    assert (plain["time_signature"]["numerator"], plain["time_signature"]["denominator"]) == (4, 4)
    _, _, out = rhythm_of(tmp_path, compound, hints={"time_signature": (6, 8)}, **kw)
    assert (out["time_signature"]["numerator"], out["time_signature"]["denominator"]) == (6, 8)
    assert abs(out["tempo"]["bpm"] - 162) < 1.0
    assert abs(out["first_beat"] - first) <= 0.010
    beats = np.array(out["beats"])
    assert np.max(np.abs(beats - (first + np.round((beats - first) / e) * e))) <= 0.010


def test_a_first_beat_hint_moves_bar_one_only(tmp_path):
    first, bar, plain = rhythm_of(tmp_path, simple, bpm=100, lead=0.8, intro_bars=0, bars=16)
    beat = bar / 4
    _, _, out = rhythm_of(tmp_path, simple, hints={"first_beat": first + beat + 0.02}, bpm=100, lead=0.8, intro_bars=0, bars=16)
    assert abs(out["first_beat"] - (first + beat)) <= 0.010              # the beat nearest the hint
    assert abs(out["tempo"]["bpm"] - plain["tempo"]["bpm"]) < 0.5
    assert "first_beat" not in out["_alternatives"]


def test_a_bpm_hint_is_kept(tmp_path):
    _, _, out = rhythm_of(tmp_path, simple, hints={"bpm": 50.0}, bpm=100, lead=0.5, bars=16)
    assert abs(out["tempo"]["bpm"] - 50.0) < 0.5 and out["tempo"]["confidence"] == 1.0
    assert "tempo" not in out["_alternatives"]


# --------------------------------------------------------------------------- reading and checking


def test_read_hints():
    assert analysis.read_hints({"bpm": 142, "time_signature": "6/8", "first_beat": 0.36, "key": "Gm"}) == {
        "rhythm": {"bpm": 142.0, "time_signature": (6, 8), "first_beat": 0.36}, "key": "Gm"}
    assert analysis.read_hints({"time_signature": {"numerator": 3, "denominator": 4}}) == {"rhythm": {"time_signature": (3, 4)}}
    for bad in ({"tempo": 90}, {"bpm": 5}, {"time_signature": "6/7"}, {"first_beat": -1}, {"key": "H"}, []):
        with pytest.raises(ValueError):
            analysis.read_hints(bad)


def test_alternatives_are_checked_like_their_parts():
    out = analysis.sanitize_analysis({
        "tempo": {"bpm": 142.2}, "time_signature": {"numerator": 6, "denominator": 8}, "key": {"name": "Gm"},
        "alternatives": {
            "tempo": [{"bpm": 71.1, "score": 0.8}, {"bpm": 142.2, "score": 0.5}, {"bpm": 900, "score": 0.5},
                      {"bpm": 47.4, "score": 1.5}],
            "time_signature": [{"numerator": 3, "denominator": 4, "score": 0.3}, {"numerator": 4, "denominator": 3, "score": 0.3}],
            "key": [{"name": "Bb", "score": 0.5}, {"name": "Gm", "score": 0.4}],
            "first_beat": [{"start": 1.0, "score": 0.5}],      # no first_beat in the analysis: left out
            "chords": [{"label": "C"}],
        }})
    assert out["alternatives"] == {"tempo": [{"bpm": 71.1, "score": 0.8}],
                                   "time_signature": [{"numerator": 3, "denominator": 4, "score": 0.3}],
                                   "key": [{"name": "Bb", "score": 0.5}]}


@pytest.fixture()
def main(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("boto3")
    monkeypatch.setenv("DEMUX_API_KEY", "k")
    monkeypatch.setenv("DEMUX_DATA_DIR", str(tmp_path))
    m = importlib.reload(importlib.import_module("main"))
    m.init_db()
    return m


def test_a_job_s_hints_reach_its_analysis(main, monkeypatch):
    from fastapi.testclient import TestClient

    async def nothing(*args):
        return None

    monkeypatch.setattr(main, "enqueue_work", nothing)
    client = TestClient(main.app)
    hints = {"time_signature": "6/8", "first_beat": 0.365}
    r = client.post("/api/v1/jobs", headers={"X-API-Key": "k"}, files={"file": ("mix.wav", b"RIFF")},
                    data={"separate": "false", "hints": json.dumps(hints)})
    assert r.status_code == 202, r.text
    seen = []
    monkeypatch.setattr(main.analysis, "analyze_job_with_limit",
                        lambda files, source, work_dir, device, lyrics=None, hints=None: seen.append(hints) or {"tempo": {"bpm": 90.0}})
    monkeypatch.setattr(main, "enqueue_notification", lambda *a: None)
    main.process_stage(r.json()["id"], "fast")
    assert seen == [hints]


def test_hints_that_can_t_be_read_are_refused(main):
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    r = client.post("/api/v1/jobs", headers={"X-API-Key": "k"}, files={"file": ("mix.wav", b"RIFF")},
                    data={"hints": '{"time_signature": "6/7"}'})
    assert r.status_code == 400 and "time_signature" in r.text
    r = client.post("/api/v1/jobs", headers={"X-API-Key": "k"}, files={"file": ("mix.wav", b"RIFF")}, data={"hints": "{"})
    assert r.status_code == 400
    r = client.post("/api/v1/jobs/url", headers={"X-API-Key": "k"},
                    json={"source_url": "https://example.com/a.wav", "hints": {"bpm": "fast"}})
    assert r.status_code == 400


def test_health_says_alternatives_and_hints(main):
    info = main.health()["analysis"]
    assert info["alternatives"] is True and info["hints"] is True
    assert main.health()["version"] == "1.11.0"
