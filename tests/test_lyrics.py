"""Sections from the lyrics a client already has (issue #22, API 1.9): reading LRC, ChordPro and plain
text into passes, placing them on the recording's bars, and the `lyrics` field of a job."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import lyrics  # noqa: E402


# --------------------------------------------------------------------------- reading


def test_chordpro_headings_environments_and_recall():
    passes = lyrics.parse("""{title: Song}
{c: Verse 1}
[G]Amazing [C]grace how [G]sweet
That saved a [D]wretch

{start_of_chorus: label="Chorus"}
[Em]My chains are [C]gone

[G]I've been set [D]free
{end_of_chorus}
{c: Verse 2}
[G]Twas grace
{sob}
[Am]Like a flood
{eob}
{chorus}
""")
    assert [(p.name, p.kind, len(p.lines)) for p in passes] == [
        ("Verse 1", "verse", 2), ("Chorus", "chorus", 2), ("Verse 2", "verse", 1),
        (None, "bridge", 1), ("Chorus", "chorus", 2)]
    assert passes[0].chords == [(7, False), (0, False), (7, False), (2, False)]
    assert passes[1].chords[0] == (4, True)                       # Em
    assert passes[0].music == passes[2].music == "verse"          # verses share their music
    assert passes[4].chords == passes[1].chords                   # the recall sings the chorus again


def test_plain_text_headings_chord_lines_and_unnamed_repeats():
    passes = lyrics.parse("""Intro
G  C  G  D

Couplet 1 :
Line one
Line two

Refrain
Sing it now
Sing it again

A line of the second verse
Another

Sing it now
Sing it again
""")
    assert [(p.name, p.kind) for p in passes[:3]] == [("Intro", "intro"), ("Couplet 1", "verse"), ("Refrain", "chorus")]
    assert not passes[0].sung and passes[0].chords == [(7, False), (0, False), (7, False), (2, False)]
    unnamed = passes[3:]
    assert [p.kind for p in unnamed] == [None, None]
    assert unnamed[1].music != unnamed[0].music
    # A block with no heading that repeats another's words is the same section as the other unnamed ones.
    blocks = lyrics.parse("one\ntwo\n\nthree\nfour\n\none\ntwo\n")
    assert blocks[0].music == blocks[2].music != blocks[1].music


@pytest.mark.parametrize("symbol,chord", [
    ("C", (0, False)), ("Am7", (9, True)), ("F#m7/C#", (6, True)), ("Bb", (10, False)),
    ("Dmaj7", (2, False)), ("Ebmin", (3, True)), ("Gsus4", (7, False)), ("Chorus", None),
])
def test_chords(symbol, chord):
    assert (lyrics.chord_of(symbol) if lyrics._CHORD_RE.match(symbol) else None) == chord


def test_lrc_repeated_stamps_blank_lines_and_offset():
    passes = lyrics.parse("""[ar:Someone]
[offset:+500]
[00:10.00]Hello there
[00:14.00]Second line

[00:30.00][01:10.00]Chorus line
[00:34.00][01:14.00]Chorus again

[00:50.00]Verse two
[00:54.00]Verse two more
""")
    assert [p.times for p in passes] == [[9.5, 13.5], [29.5, 33.5], [49.5, 53.5], [69.5, 73.5]]
    assert passes[1].music == passes[3].music != passes[0].music


def test_nothing_to_read():
    assert lyrics.parse("") == [] and lyrics.parse("   \n\n") == []


# --------------------------------------------------------------------------- placing

NAMES = {"A": "Verse", "B": "Chorus", "C": "Bridge"}


def chart(plan, names=NAMES):
    """ChordPro for `plan` as a client writes it: a heading per pass, two bars a line."""
    out, verse = [], 0
    for part, bars in plan:
        name = names[part]
        if name == "Verse":
            verse += 1
            name = f"Verse {verse}"
        out.append("{c: %s}" % name)
        out += [f"words of {name} {i}" for i in range(max(1, bars // 2))]
        out.append("")
    return "\n".join(out)


def lrc(plan, bpm=96.0, lead=0.4, late=2.0, speed=1.03):
    """An LRC of `plan` timed to another release: 2 s late and running 3 % fast, each line a
    quarter beat before its bar (a pickup)."""
    bar, t, out, verse = 4 * 60 / bpm, lead, [], 0
    for part, bars in plan:
        verse += part == "A"
        for i in range(max(1, bars // 2)):
            at = (t + i * 2 * bar - 0.25 * 60 / bpm + late) * speed
            words = {"A": f"verse {verse} line {i}", "B": f"chorus line {i}", "C": f"over the bridge {i}"}[part]
            out.append("[%02d:%05.2f]%s" % (at // 60, at % 60, words))
        out.append("")
        t += bars * bar
    return "\n".join(out)


SONG = (("A", 8), ("B", 8), ("A", 8), ("B", 8), ("C", 4), ("B", 8))


def place(tmp_path, plan, text, **kw):
    pytest.importorskip("librosa")
    import analysis
    import synth

    starts, _ = synth.make_sections(tmp_path, plan=plan, **kw)
    out = analysis.analyze_job(sorted(tmp_path.glob("*.wav")), None, tmp_path, lyrics_text=text)
    return starts, out


@pytest.mark.parametrize("plan,text,names,kw", [
    (SONG, chart(SONG), ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Chorus"], {}),
    # The recording sings the chorus once more than the chart.
    (SONG + (("B", 8),), chart(SONG), ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Chorus", "Chorus"], {}),
    # ... leaves the second verse out.
    ((("A", 8), ("B", 8), ("B", 8), ("C", 4), ("B", 8)), chart(SONG), ["Verse 1", "Chorus", "Chorus", "Bridge", "Chorus"], {}),
    # ... plays a 2-bar interlude the chart doesn't have (a section without a hint).
    ((("A", 8), ("B", 8), ("C", 2), ("A", 8), ("B", 8), ("B", 8)),
     chart((("A", 8), ("B", 8), ("A", 8), ("B", 8), ("B", 8))), ["Verse 1", "Chorus", None, "Verse 2", "Chorus", "Chorus"], {}),
    # 3/4, and a chorus sung twice in a row, where nothing changes between the two.
    ((("A", 4), ("B", 8), ("A", 4), ("B", 8), ("B", 8), ("B", 8)),
     chart((("A", 4), ("B", 8), ("A", 4), ("B", 8), ("B", 8), ("B", 8))),
     ["Verse 1", "Chorus", "Verse 2", "Chorus", "Chorus", "Chorus"], dict(bpm=117, meter=3, seed=3)),
], ids=["as-charted", "extra-chorus", "verse-left-out", "interlude-not-charted", "3/4-choruses-in-a-row"])
def test_the_chart_s_sections_on_their_bars(tmp_path, plan, text, names, kw):
    starts, out = place(tmp_path, plan, text, **kw)
    sections = out["sections"]
    assert out["engines"]["sections"].startswith("librosa-lyrics@")
    assert [s.get("name") for s in sections] == names
    assert sections[0]["start"] == 0.0
    for want, have in zip(starts[1:], sections[1:]):
        assert abs(have["start"] - want) <= 0.010
        assert have["start"] in out["downbeats"]
    hinted = [s["hint"] for s in sections if "hint" in s]
    assert hinted == sorted(hinted)                       # in the chart's order


def test_an_lrc_timed_to_another_release(tmp_path):
    starts, out = place(tmp_path, SONG, lrc(SONG))
    sections = out["sections"]
    assert [s.get("hint") for s in sections] == [0, 1, 2, 3, 4, 5]
    assert [s["label"] for s in sections] == ["verse", "chorus", "verse", "chorus", "verse", "chorus"]
    for want, have in zip(starts[1:], sections[1:]):
        assert abs(have["start"] - want) <= 0.010


def test_a_chart_that_doesn_t_fit_is_left_unused(tmp_path):
    wrong = chart((("A", 8), ("A", 8), ("A", 8), ("B", 8), ("A", 8), ("A", 8), ("B", 8), ("B", 8)))
    _, out = place(tmp_path, SONG, wrong)
    assert out["sections"] and not any("hint" in s or "name" in s for s in out["sections"])
    assert out["engines"]["sections"].startswith("librosa@")


# --------------------------------------------------------------------------- the API


@pytest.fixture()
def main(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("boto3")
    monkeypatch.setenv("DEMUX_API_KEY", "k")
    monkeypatch.setenv("DEMUX_DATA_DIR", str(tmp_path))
    m = importlib.reload(importlib.import_module("main"))
    m.init_db()
    return m


def test_a_job_s_lyrics_reach_its_analysis_and_go(main, monkeypatch):
    from fastapi.testclient import TestClient

    async def nothing(*args):
        return None

    monkeypatch.setattr(main, "enqueue_work", nothing)
    client = TestClient(main.app)
    r = client.post("/api/v1/jobs", headers={"X-API-Key": "k"}, files={"file": ("mix.wav", b"RIFF")},
                    data={"separate": "false", "lyrics": "{c: Chorus}\nLa la"})
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    seen = []
    monkeypatch.setattr(main.analysis, "analyze_job_with_limit",
                        lambda files, source, work_dir, device, lyrics=None: seen.append(lyrics) or {"tempo": {"bpm": 90.0}})
    monkeypatch.setattr(main, "enqueue_notification", lambda *a: None)
    main.process_stage(jid, "fast")
    assert seen == ["{c: Chorus}\nLa la"]
    assert not (main.JOBS_DIR / jid / "lyrics.txt").exists()     # read, then gone


def test_lyrics_too_long_are_refused(main):
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    r = client.post("/api/v1/jobs", headers={"X-API-Key": "k"}, files={"file": ("mix.wav", b"RIFF")},
                    data={"lyrics": "x" * (64 * 1024 + 1)})
    assert r.status_code == 400
    r = client.post("/api/v1/jobs/url", headers={"X-API-Key": "k"},
                    json={"source_url": "https://example.com/a.wav", "lyrics": "x" * (64 * 1024 + 1)})
    assert r.status_code == 400


def test_health_says_lyrics_are_read(main):
    assert main.health()["analysis"]["lyrics"] is True
    assert main.health()["version"] == "1.9.0"


def test_hint_and_name_kept_only_when_valid():
    import analysis

    out = analysis.sanitize_analysis({"sections": [
        {"start": 0.0, "label": "verse", "hint": 0, "name": " Verse 1 "},
        {"start": 10.0, "label": "chorus", "hint": True, "name": "Chorus"},
        {"start": 20.0, "label": "inst"},
    ]})
    assert out["sections"] == [{"start": 0.0, "label": "verse", "hint": 0, "name": "Verse 1"},
                               {"start": 10.0, "label": "chorus"}, {"start": 20.0, "label": "inst"}]
