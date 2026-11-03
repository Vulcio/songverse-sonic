"""Exercise legacy policy with lightweight engines, without model installs."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import analysis
from analyzers import AnalysisInput, LegacyAnalysisAdapter


class Budget:
    def ok(self):
        return True


class StubAnalyzer:
    engine_id = "test"

    def __init__(self, result=None, error=False):
        self.result = result or {}
        self.error = error
        self.calls = []

    def analyze(self, inputs, budget):
        self.calls.append((inputs, budget))
        if self.error:
            raise RuntimeError("backend unavailable")
        return dict(self.result)


def test_inputs_are_a_frozen_snapshot(tmp_path):
    files = [tmp_path / "drums.wav"]
    inputs = AnalysisInput(tuple(files), None, tmp_path)
    files.clear()
    assert inputs.files == (tmp_path / "drums.wav",)
    with pytest.raises(FrozenInstanceError):
        inputs.device = "cuda"


def test_baseline_result_is_sanitized_with_its_duration(tmp_path):
    # An out-of-file prediction must disappear as a whole, not be trimmed.
    baseline = StubAnalyzer({"beats": [0.0, 1.0, 3.0], "downbeats": [0.0],
                             "_duration": 2.0, "key": {"name": "C"}})
    inputs, budget = AnalysisInput((), None, tmp_path), Budget()
    adapter = LegacyAnalysisAdapter(baseline, analysis.sanitize_analysis)
    assert adapter.analyze(inputs, budget) == {"key": {"name": "C"}}
    assert baseline.calls == [(inputs, budget)]


def test_primary_beats_do_not_gain_baseline_tempo_or_confidence(tmp_path):
    primary = StubAnalyzer({"beats": [0.2, 0.8, 1.4], "downbeats": [0.2],
                            "tempo": {"bpm": 100}, "time_signature": {"numerator": 4, "denominator": 4}})
    baseline = StubAnalyzer({"beats": [0.0, 1.0, 2.0], "downbeats": [0.0],
                             "tempo": {"bpm": 60, "confidence": 0.9},
                             "beats_confidence": 0.8, "intro_free": True,
                             "key": {"name": "G"}, "_duration": 3.0})
    inputs, budget = AnalysisInput((), None, tmp_path), Budget()
    result = LegacyAnalysisAdapter(baseline, analysis.sanitize_analysis, primary).analyze(inputs, budget)
    assert result == {"beats": [0.2, 0.8, 1.4], "downbeats": [0.2], "first_beat": 0.2,
                      "tempo": {"bpm": 100.0}, "time_signature": {"numerator": 4, "denominator": 4},
                      "key": {"name": "G"}}
    assert primary.calls == baseline.calls == [(inputs, budget)]


@pytest.mark.parametrize("primary_fails,baseline_fails,expected", [
    (True, False, {"key": {"name": "G"}}),
    (False, True, {"key": {"name": "C"}}),
    (True, True, None),
])
def test_backend_failures_preserve_available_findings(tmp_path, primary_fails, baseline_fails, expected):
    primary = StubAnalyzer({"key": {"name": "C"}}, error=primary_fails)
    baseline = StubAnalyzer({"key": {"name": "G"}}, error=baseline_fails)
    adapter = LegacyAnalysisAdapter(baseline, analysis.sanitize_analysis, primary)
    assert adapter.analyze(AnalysisInput((), None, tmp_path), Budget()) == expected
    assert len(primary.calls) == len(baseline.calls) == 1


@pytest.mark.parametrize("backend", ["librosa", "allin1", "unknown"])
def test_legacy_entry_point_selects_adapters_and_forwards_inputs(tmp_path, monkeypatch, backend):
    seen = []
    files, source = [tmp_path / "drums.wav"], tmp_path / "mix.wav"

    def baseline(selected, selected_source, budget):
        seen.append(("librosa", selected, selected_source, budget))
        return {"key": {"name": "G"}, "_duration": 3.0}

    def primary(selected, selected_source, work_dir, device, budget):
        seen.append(("allin1", selected, selected_source, budget))
        assert (work_dir, device) == (tmp_path, "cuda")
        return {"key": {"name": "C"}}

    monkeypatch.setattr(analysis, "ANALYSIS_ENABLED", True)
    monkeypatch.setattr(analysis, "ANALYSIS_BACKEND", backend)
    monkeypatch.setattr(analysis, "_analyze_librosa", baseline)
    monkeypatch.setattr(analysis, "_analyze_allin1", primary)
    assert analysis.analyze_job(files, source, tmp_path, "cuda") == {
        "key": {"name": "C" if backend == "allin1" else "G"}}
    assert [call[0] for call in seen] == (["allin1", "librosa"] if backend == "allin1" else ["librosa"])
    assert all(call[1] == files and call[2] == source for call in seen)
    assert all(call[3] is seen[0][3] for call in seen)


def test_disabled_analysis_never_calls_an_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ANALYSIS_ENABLED", False)
    monkeypatch.setattr(analysis.LibrosaAnalyzer, "analyze", lambda *args: pytest.fail("disabled"))
    monkeypatch.setattr(analysis.AllInOneAnalyzer, "analyze", lambda *args: pytest.fail("disabled"))
    assert analysis.analyze_job([], None, tmp_path) is None
