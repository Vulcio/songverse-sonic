"""Functions run in a child process by test_analysis_only.py (they must be importable there)."""
import time


def answer(value):
    return {"value": value}


def hang(seconds):
    time.sleep(seconds)
    return {"too": "late"}


def fail():
    raise RuntimeError("broken engine")
