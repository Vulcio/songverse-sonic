import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("boto3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))


@pytest.fixture()
def main(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMUX_API_KEY", "k")
    monkeypatch.setenv("DEMUX_DATA_DIR", str(tmp_path))
    m = importlib.import_module("main")
    m = importlib.reload(m)
    m.init_db()
    return m


def _job(main):
    jid = main.create_job("s.wav", "htdemucs", None, None, False, None, None, None, None, None)
    return jid


def test_analysis_null_until_stored(main):
    jid = _job(main)
    assert main.public_job(main.get_job(jid))["analysis"] is None


def test_analysis_exposed_in_job_and_webhook(main):
    jid = _job(main)
    a = {"tempo": {"bpm": 72.1, "confidence": 0.9}, "key": {"name": "G"}}
    with main.db() as conn:
        conn.execute("UPDATE jobs SET analysis_json=? WHERE id=?", (json.dumps(a), jid))
    assert main.public_job(main.get_job(jid))["analysis"] == a
    assert main.webhook_payload(jid, "fast.completed")["job"]["analysis"] == a


def test_analysis_failure_does_not_fail_job(main, monkeypatch):
    jid = _job(main)
    monkeypatch.setattr(main.analysis, "analyze_job", lambda *a, **k: 1 / 0)
    assert main.run_analysis(jid) is None


def test_health_reports_analysis(main):
    assert main.health()["analysis"]["enabled"] is True


@pytest.mark.parametrize("outcome", ["found", "empty", "failed"])
def test_fast_completion_waits_for_analysis(main, monkeypatch, outcome):
    jid = _job(main)
    expected = {"tempo": {"bpm": 100.0}} if outcome == "found" else None
    calls = []
    during_analysis = []

    def separate(job_id, quality):
        (main.result_dir(job_id, quality) / "drums.wav").write_bytes(b"separated audio")

    def analyze(*args, **kwargs):
        calls.append("analysis")
        during_analysis.append(main.public_job(main.get_job(jid)))
        if outcome == "failed":
            raise RuntimeError("unavailable analyzer")
        return expected

    def notify(job_id, event):
        calls.append(event)
        payload = main.webhook_payload(job_id, event)
        assert payload["job"]["fast"]["status"] == "completed"
        assert payload["job"]["analysis"] == expected

    monkeypatch.setattr(main, "run_demucs", separate)
    monkeypatch.setattr(main, "upload_r2", lambda *args: None)
    monkeypatch.setattr(main.analysis, "analyze_job", analyze)
    monkeypatch.setattr(main, "enqueue_notification", notify)
    main.process_stage(jid, "fast")

    assert calls == ["analysis", "fast.completed"]
    assert during_analysis[0]["fast"]["status"] == "processing"
    assert during_analysis[0]["analysis"] is None
    job = main.public_job(main.get_job(jid))
    assert job["status"] == "completed"
    assert job["error"] is None
    assert job["analysis"] == expected
    assert [f["name"] for f in job["fast"]["files"]] == ["drums.wav"]
    assert (main.result_dir(jid, "fast") / "drums.wav").read_bytes() == b"separated audio"


def test_hq_preserves_fast_analysis(main, monkeypatch):
    jid = _job(main)
    expected = {"tempo": {"bpm": 100.0}}
    with main.db() as conn:
        conn.execute(
            "UPDATE jobs SET analysis_json=?, fast_status='completed', hq_enabled=1, "
            "hq_status='scheduled' WHERE id=?", (json.dumps(expected), jid),
        )
    events = []
    monkeypatch.setattr(main, "run_demucs", lambda *args: None)
    monkeypatch.setattr(main, "upload_r2", lambda *args: None)
    monkeypatch.setattr(main, "run_analysis", lambda *args: pytest.fail("HQ must not rerun analysis"))
    monkeypatch.setattr(main, "enqueue_notification", lambda jid, event: events.append(main.webhook_payload(jid, event)))
    main.process_stage(jid, "hq")

    assert [event["event"] for event in events] == ["hq.started", "hq.completed"]
    assert all(event["job"]["analysis"] == expected for event in events)
    assert main.public_job(main.get_job(jid))["analysis"] == expected
