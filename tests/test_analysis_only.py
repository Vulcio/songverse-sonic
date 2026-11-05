"""Analysis-only jobs and the analysis's hard time limit (issue #7, API 1.8)."""
import importlib
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("boto3")
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import limit_targets  # noqa: E402


@pytest.fixture()
def main(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMUX_API_KEY", "k")
    monkeypatch.setenv("DEMUX_DATA_DIR", str(tmp_path))
    m = importlib.reload(importlib.import_module("main"))
    m.init_db()
    return m


def test_hard_limit_returns_a_result_in_time():
    import analysis
    assert analysis.run_with_hard_limit(limit_targets.answer, (3,), 60) == {"value": 3}


def test_hard_limit_stops_what_never_returns():
    import analysis
    started = time.monotonic()
    assert analysis.run_with_hard_limit(limit_targets.hang, (60,), 3) is None
    assert time.monotonic() - started < 20


def test_hard_limit_reports_a_failure_as_nothing():
    import analysis
    assert analysis.run_with_hard_limit(limit_targets.fail, (), 60) is None


def test_analysis_only_job_skips_demucs(main, monkeypatch, tmp_path):
    jid = main.create_job("song.wav", "htdemucs", None, None, False, None, None, None, None, None, separate=False)
    (main.JOBS_DIR / jid / "input" / "song.wav").write_bytes(b"audio")
    found = {"tempo": {"bpm": 90.0}}
    events = []
    monkeypatch.setattr(main, "run_demucs", lambda *a: pytest.fail("analysis only: no Demucs"))
    monkeypatch.setattr(main, "upload_r2", lambda *a: pytest.fail("analysis only: nothing to upload"))
    monkeypatch.setattr(main.analysis, "analyze_job_with_limit",
                        lambda files, source, work_dir, device: found if source.name == "song.wav" and files == [] else None)
    monkeypatch.setattr(main, "enqueue_notification", lambda job_id, event: events.append(event))
    main.process_stage(jid, "fast")
    job = main.public_job(main.get_job(jid))
    assert job["status"] == "completed" and job["fast"]["status"] == "completed"
    assert job["separate"] is False and job["fast"]["files"] == [] and job["analysis"] == found
    assert events == ["fast.completed"]


def test_analysis_only_refuses_hq_and_needs_analysis(main, monkeypatch):
    with pytest.raises(HTTPException):
        main.create_job("s.wav", "htdemucs", None, None, True, None, None, None, None, None, separate=False)
    monkeypatch.setattr(main.analysis, "ANALYSIS_ENABLED", False)
    with pytest.raises(HTTPException) as refused:
        main.create_job("s.wav", "htdemucs", None, None, False, None, None, None, None, None, separate=False)
    assert "turned off" in refused.value.detail


def test_upload_with_separate_false(main, monkeypatch):
    async def nothing(*args):
        return None

    monkeypatch.setattr(main, "enqueue_work", nothing)
    client = TestClient(main.app)
    r = client.post("/api/v1/jobs", headers={"X-API-Key": "k"}, files={"file": ("mix.wav", b"RIFF")},
                    data={"separate": "false"})
    assert r.status_code == 202, r.text
    assert r.json()["separate"] is False
    r = client.post("/api/v1/jobs", headers={"X-API-Key": "k"}, files={"file": ("mix.wav", b"RIFF")})
    assert r.json()["separate"] is True


def test_health_lists_parts(main, monkeypatch):
    info = main.health()["analysis"]
    assert info["parts"] == ["rhythm", "key", "sections"] and info["analysis_only"] is True
    monkeypatch.setattr(main.analysis, "ANALYSIS_CHORDS", "btc")
    assert main.health()["analysis"]["parts"][-1] == "chords"


def test_old_jobs_are_separations(main):
    jid = main.create_job("s.wav", "htdemucs", None, None, False, None, None, None, None, None)
    assert main.public_job(main.get_job(jid))["separate"] is True
