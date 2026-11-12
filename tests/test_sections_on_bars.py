"""Sections start on bar lines, where the music changes, so a section loops on whole bars (issue #21)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))


@pytest.mark.parametrize("kw", [
    dict(),                                                                           # 8-bar sections, a 4-bar one
    dict(bpm=128, lead=1.2, plan=(("C", 4), ("A", 8), ("B", 8), ("A", 6), ("B", 8), ("C", 4))),   # a 6-bar one
    dict(bpm=84, meter=3, plan=(("A", 8), ("B", 8), ("A", 8), ("B", 8), ("C", 4))),   # 3/4
], ids=["96-4/4", "128-six-bar", "84-3/4"])
def test_each_section_starts_on_its_bar(tmp_path, kw):
    pytest.importorskip("librosa")
    import analysis
    import synth

    starts, _ = synth.make_sections(tmp_path, **kw)
    out = analysis.analyze_job(sorted(tmp_path.glob("*.wav")), None, tmp_path)
    got = [s["start"] for s in out["sections"]]
    assert got[0] == 0.0
    assert len(got) == len(starts)
    for want, have in zip(starts[1:], got[1:]):
        assert abs(have - want) <= 0.010
        assert have in out["downbeats"]       # exactly on a bar line, as reported
