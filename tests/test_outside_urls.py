"""Source downloads and webhooks check every address they reach (issue #14)."""
import asyncio
import importlib
import socket
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("boto3")
import httpx
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

ADDRESSES = {
    "public.example": "93.184.216.34",
    "other.example": "93.184.216.35",
    "inside.example": "10.0.0.5",
    "metadata.example": "169.254.169.254",
    "mapped.example": "::ffff:127.0.0.1",
}


def fake_getaddrinfo(host, port, *args, **kwargs):
    if host not in ADDRESSES:
        raise socket.gaierror("unknown")
    ip = ADDRESSES[host]
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 0))]


@pytest.fixture()
def main(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMUX_API_KEY", "k")
    monkeypatch.setenv("DEMUX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEMUX_ALLOW_PRIVATE_SOURCE_URLS", "false")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    m = importlib.reload(importlib.import_module("main"))
    monkeypatch.setattr(m.socket, "getaddrinfo", fake_getaddrinfo)
    return m


def serve(main, routes):
    """Answer pinned requests by the Host they were for; record what was sent."""
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return routes[(request.headers["host"], request.url.path)]()

    main._OUTSIDE_TRANSPORT = httpx.MockTransport(handler)
    return seen


def download(main, url, tmp_path):
    dest = tmp_path / "audio.bin"
    asyncio.run(main.download_url(url, dest))
    return dest.read_bytes()


def test_download_connects_to_the_checked_address(main, tmp_path):
    seen = serve(main, {("public.example", "/song.mp3"): lambda: httpx.Response(200, content=b"audio")})
    assert download(main, "https://public.example/song.mp3", tmp_path) == b"audio"
    request = seen[0]
    assert request.url.host == "93.184.216.34"
    assert request.extensions["sni_hostname"] == "public.example"


def test_public_redirects_are_followed_and_checked(main, tmp_path):
    seen = serve(main, {
        ("public.example", "/a"): lambda: httpx.Response(302, headers={"location": "https://other.example/b"}),
        ("other.example", "/b"): lambda: httpx.Response(200, content=b"ok"),
    })
    assert download(main, "https://public.example/a", tmp_path) == b"ok"
    assert [r.url.host for r in seen] == ["93.184.216.34", "93.184.216.35"]


@pytest.mark.parametrize("target", [
    "http://inside.example/x", "http://metadata.example/latest", "http://127.0.0.1:8000/x",
    "http://[::1]/x", "http://mapped.example/x",
])
def test_redirect_to_a_private_address_is_refused(main, tmp_path, target):
    seen = serve(main, {("public.example", "/a"): lambda: httpx.Response(302, headers={"location": target})})
    with pytest.raises(HTTPException) as refused:
        download(main, "https://public.example/a", tmp_path)
    assert refused.value.status_code == 400
    assert len(seen) == 1


def test_a_name_that_now_resolves_privately_is_refused(main, tmp_path, monkeypatch):
    serve(main, {("public.example", "/a"): lambda: httpx.Response(200, content=b"x")})
    main.validate_remote_url("https://public.example/a", False, "source")  # fine when submitted
    monkeypatch.setitem(ADDRESSES, "public.example", "192.168.1.10")  # then rebound
    with pytest.raises(HTTPException):
        download(main, "https://public.example/a", tmp_path)


def test_behind_a_proxy_names_are_still_checked_on_every_redirect(main, tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    monkeypatch.setenv("NO_PROXY", "")
    seen = serve(main, {
        ("public.example", "/a"): lambda: httpx.Response(302, headers={"location": "https://inside.example/b"}),
    })
    with pytest.raises(HTTPException):
        download(main, "https://public.example/a", tmp_path)
    assert seen[0].url.host == "public.example"  # the proxy resolves it, not Sonic
    assert len(seen) == 1


def test_too_many_redirects(main, tmp_path):
    serve(main, {("public.example", "/loop"): lambda: httpx.Response(302, headers={"location": "/loop"})})
    with pytest.raises(HTTPException) as refused:
        download(main, "https://public.example/loop", tmp_path)
    assert "redirects" in refused.value.detail


def test_private_addresses_still_allowed_when_configured(main, tmp_path):
    main.ALLOW_PRIVATE_SOURCE_URLS = True
    seen = serve(main, {("inside.example", "/a"): lambda: httpx.Response(200, content=b"lan")})
    assert download(main, "http://inside.example/a", tmp_path) == b"lan"
    assert seen[0].url.host == "inside.example"


def test_webhooks_are_pinned_and_refused_when_private(main, monkeypatch):
    main.init_db()
    jid = main.create_job("s.wav", "htdemucs", None, None, False, None, None, None, None, None)
    seen = serve(main, {("public.example", "/hook"): lambda: httpx.Response(204)})

    def notify(url):
        with main.db() as conn:
            conn.execute(
                "INSERT INTO notifications(job_id,event,url,secret,payload_json,status,attempts,next_attempt,created_at) "
                "VALUES (?,?,?,?,?,'pending',0,0,0)", (jid, "fast.completed", url, None, "{}"))

    async def one_round():
        task = asyncio.create_task(main.webhook_loop())
        await asyncio.sleep(0.2)
        task.cancel()

    notify("https://public.example/hook")
    monkeypatch.setitem(ADDRESSES, "public.example", "93.184.216.34")
    asyncio.run(one_round())
    assert seen and seen[0].url.host == "93.184.216.34"
    with main.db() as conn:
        assert conn.execute("SELECT status FROM notifications").fetchone()["status"] == "delivered"
