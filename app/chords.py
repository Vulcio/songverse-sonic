"""Chords from a recording, placed on its beats (issue #15).

The engine is BTC, the Bi-directional Transformer for Chord recognition (Park et al., ISMIR 2019,
https://github.com/jayg996/BTC-ISMIR19, MIT licence, Copyright (c) 2019 Jonggwon Park). Its
inference code is reproduced below, trimmed to what inference needs and kept loadable from the
published checkpoints. ChordMini (https://github.com/ptnghia-j/ChordMini, MIT) retrained the same
model; its checkpoint loads here too, as a separate engine, because its weights were trained partly
on labelled data its authors keep private (see README "Licences").

Torch is imported only when chords are asked for: Demucs already needs it, the rest of the
analysis does not. Nothing here touches jobs, webhooks or the database.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("demucs-api.analysis")

# --------------------------------------------------------------------------- #
# Vocabulary: BTC's 170 classes, 12 roots x 14 qualities, then X (unknown) and N (no chord)
# --------------------------------------------------------------------------- #

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7", "dim7", "hdim7",
             "sus2", "sus4"]
NO_CHORD, UNKNOWN = 169, 168


def chord_of(index: int) -> tuple[int, str] | None:
    """(root pitch class, quality) for a class index; None for no chord or unknown."""
    if index >= 168 or index < 0:
        return None
    return index // 14, QUALITIES[index % 14]


def harte_label(root: int, quality: str) -> str:
    """The chord in Harte's syntax, as mir_eval reads it ("C#:min7", "C" for C major)."""
    return ROOTS[root] if quality == "maj" else f"{ROOTS[root]}:{quality}"


# --------------------------------------------------------------------------- #
# Engines (checkpoints), downloaded on first use and checked against their sha256
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Checkpoint:
    engine_id: str
    url: str
    sha256: str
    file_name: str


CHECKPOINTS = {
    "btc": Checkpoint(
        "btc",
        "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model_large_voca.pt",
        "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51",
        "btc_model_large_voca.pt",
    ),
    "btc-chordmini": Checkpoint(
        "btc-chordmini",
        "https://raw.githubusercontent.com/ptnghia-j/ChordMini/aa6e3a8d7b017f082fd2aaff9329d5c26af49c03/checkpoints/btc_model_best.pth",
        "e0a12ca6d881f81e01dfb459b4836ec7dac8d8f87b0170a412b5807256b3a1fc",
        "btc_chordmini_cl.pth",
    ),
}

SR = 22050
HOP = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
TIMESTEP = 108
FRAME = HOP / SR  # 0.0929 s per frame
MIN_CHORD_SECONDS = 0.35  # shorter segments are merged into their neighbours


def models_dir() -> Path:
    data = Path(os.getenv("DEMUX_DATA_DIR", "/var/lib/demucs-api"))
    return Path(os.getenv("ANALYSIS_MODELS_DIR", str(data / "models")))


def checkpoint_path(engine: str, download: bool = True) -> Path | None:
    """The engine's checkpoint on disk, downloading and checking it the first time."""
    spec = CHECKPOINTS.get(engine)
    if spec is None:
        return None
    path = models_dir() / spec.file_name
    if path.exists() and _sha256(path) == spec.sha256:
        return path
    if not download:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    log.info("analysis: downloading the %s chord model", engine)
    with urllib.request.urlopen(spec.url, timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            if out.tell() > 200 << 20:
                raise RuntimeError("chord model larger than expected")
    if _sha256(partial) != spec.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"the {engine} chord model does not match its checksum")
    partial.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# BTC model (from BTC-ISMIR19's btc_model.py and utils/transformer_modules.py, MIT)
# --------------------------------------------------------------------------- #

def _build_model(num_chords: int = 170):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def bias_mask(length):
        mask = np.triu(np.full([length, length], -np.inf), 1)
        return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(1)

    def timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = np.arange(length)
        num = channels // 2
        increment = math.log(float(max_timescale) / float(min_timescale)) / (float(num) - 1)
        inv = min_timescale * np.exp(np.arange(num).astype(np.float64) * -increment)
        scaled = np.expand_dims(position, 1) * np.expand_dims(inv, 0)
        signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
        signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant")
        return torch.from_numpy(signal.reshape([1, length, channels])).float()

    class LayerNorm(nn.Module):
        def __init__(self, features, eps=1e-6):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta

    class MultiHeadAttention(nn.Module):
        def __init__(self, depth, key_depth, value_depth, out_depth, heads, mask):
            super().__init__()
            self.heads = heads
            self.query_scale = (key_depth // heads) ** -0.5
            self.bias_mask = mask
            self.query_linear = nn.Linear(depth, key_depth, bias=False)
            self.key_linear = nn.Linear(depth, key_depth, bias=False)
            self.value_linear = nn.Linear(depth, value_depth, bias=False)
            self.output_linear = nn.Linear(value_depth, out_depth, bias=False)

        def split(self, x):
            b, t, d = x.shape
            return x.view(b, t, self.heads, d // self.heads).permute(0, 2, 1, 3)

        def forward(self, x):
            q = self.split(self.query_linear(x)) * self.query_scale
            k = self.split(self.key_linear(x))
            v = self.split(self.value_linear(x))
            logits = torch.matmul(q, k.permute(0, 1, 3, 2))
            logits = logits + self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits)
            context = torch.matmul(F.softmax(logits, dim=-1), v)
            b, h, t, d = context.shape
            return self.output_linear(context.permute(0, 2, 1, 3).contiguous().view(b, t, h * d))

    class Conv(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.pad = nn.ConstantPad1d((1, 1), 0)
            self.conv = nn.Conv1d(n_in, n_out, kernel_size=3, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x.permute(0, 2, 1))).permute(0, 2, 1)

    class FeedForward(nn.Module):
        def __init__(self, size, filter_size):
            super().__init__()
            self.layers = nn.ModuleList([Conv(size, filter_size), Conv(filter_size, size)])

        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))  # BTC applies ReLU after the last layer too
            return x

    class Block(nn.Module):
        def __init__(self, size, mask):
            super().__init__()
            self.multi_head_attention = MultiHeadAttention(size, size, size, size, 4, mask)
            self.positionwise_convolution = FeedForward(size, size)
            self.layer_norm_mha = LayerNorm(size)
            self.layer_norm_ffn = LayerNorm(size)

        def forward(self, x):
            x = x + self.multi_head_attention(self.layer_norm_mha(x))
            return x + self.positionwise_convolution(self.layer_norm_ffn(x))

    class BiBlock(nn.Module):
        def __init__(self, size, length):
            super().__init__()
            self.attn_block = Block(size, bias_mask(length))
            self.backward_attn_block = Block(size, torch.transpose(bias_mask(length), 2, 3))
            self.linear = nn.Linear(size * 2, size)

        def forward(self, x):
            return self.linear(torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2))

    class Layers(nn.Module):
        def __init__(self, features, size, layers, length):
            super().__init__()
            self.timing_signal = timing_signal(length, size)
            self.embedding_proj = nn.Linear(features, size, bias=False)
            self.self_attn_layers = nn.Sequential(*[BiBlock(size, length) for _ in range(layers)])
            self.layer_norm = LayerNorm(size)

        def forward(self, x):
            y = self.embedding_proj(x) + self.timing_signal[:, :x.shape[1], :].type_as(x)
            return self.layer_norm(self.self_attn_layers(y))

    class Output(nn.Module):
        def __init__(self, size, n_out):
            super().__init__()
            self.output_projection = nn.Linear(size, n_out)
            self.lstm = nn.LSTM(input_size=size, hidden_size=size // 2, batch_first=True, bidirectional=True)  # unused

        def forward(self, x):
            return self.output_projection(x)

    class BTC(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn_layers = Layers(N_BINS, 128, 8, TIMESTEP)
            self.output_layer = Output(128, num_chords)

        def forward(self, x):
            return self.output_layer(self.self_attn_layers(x))

    return BTC()


_loaded: dict[str, Any] = {}


def _load(engine: str, device: str = "cpu"):
    if engine in _loaded:
        return _loaded[engine]
    import torch

    path = checkpoint_path(engine)
    if path is None:
        raise RuntimeError(f"unknown chord engine {engine!r}")
    checkpoint = _safe_load(path)
    state = checkpoint.get("model") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise RuntimeError(f"the {engine} checkpoint has no model weights")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    mean, std = checkpoint.get("mean"), checkpoint.get("std")
    if mean is None or std is None:
        norm = checkpoint.get("normalization") or {}
        mean, std = norm.get("mean"), norm.get("std")
    if mean is None or std is None:
        raise RuntimeError(f"the {engine} checkpoint has no feature normalisation")
    model = _build_model()
    model.load_state_dict(state, strict=True)
    model.eval()
    if device != "cpu":
        try:
            model.to(device)
        except Exception:
            device = "cpu"
    _loaded[engine] = (model, _to_float(mean), _to_float(std), device)
    return _loaded[engine]


def _safe_load(path: Path):
    """torch.load without running code from the file (weights_only), allowing the numpy scalars
    BTC's checkpoint stores its normalisation as. The file's sha256 is checked before this too."""
    import numpy as np
    import torch

    scalar = np._core.multiarray.scalar if hasattr(np, "_core") else np.core.multiarray.scalar
    allowed = [(scalar, "numpy.core.multiarray.scalar"), (scalar, "numpy._core.multiarray.scalar"), np.dtype,
               *(type(np.dtype(t)) for t in (np.float32, np.float64))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def _to_float(value):
    import numpy as np
    try:
        import torch
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #

def features(y, sr: int = SR):
    """Log-magnitude CQT, 144 bins (6 octaves from C1), one frame per 2048 samples, as BTC was trained."""
    import librosa
    import numpy as np

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    # BTC computes the CQT in 10 s pieces; whole-signal CQT differs only at the piece edges.
    piece = int(SR * 10.0)
    parts = []
    for start in range(0, max(len(y), 1), piece):
        seg = y[start:start + piece]
        if len(seg) < HOP:
            break
        parts.append(librosa.cqt(seg, sr=SR, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP))
    if not parts:
        return np.zeros((0, N_BINS), dtype=np.float32)
    cqt = np.concatenate(parts, axis=1)
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)


def frame_probabilities(y, sr: int, engine: str = "btc", device: str = "cpu"):
    """Per-frame class probabilities (frames x 170), one frame every FRAME seconds."""
    import numpy as np
    import torch

    model, mean, std, device = _load(engine, device)
    feats = (features(y, sr) - mean) / std
    n = feats.shape[0]
    if n == 0:
        return np.zeros((0, 170), dtype=np.float32)
    pad = (-n) % TIMESTEP
    feats = np.pad(feats, ((0, pad), (0, 0)))
    x = torch.from_numpy(feats).view(-1, TIMESTEP, N_BINS).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1).reshape(-1, 170)[:n]
    return probs.cpu().numpy()


def segments_from_frames(classes, frame: float = FRAME, duration: float | None = None) -> list[dict]:
    """Runs of the same class become chords; runs shorter than MIN_CHORD_SECONDS join a neighbour."""
    runs: list[list] = []  # [class, start_frame, end_frame]
    for i, c in enumerate(classes):
        c = int(c)
        if c == UNKNOWN:
            c = NO_CHORD
        if runs and runs[-1][0] == c:
            runs[-1][2] = i + 1
        else:
            runs.append([c, i, i + 1])
    min_frames = max(1, round(MIN_CHORD_SECONDS / frame))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if run[2] - run[1] >= min_frames:
                continue
            # Join the longer neighbour (it is more likely the chord really playing).
            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None
            target = left if right is None or (left is not None and left[2] - left[1] >= right[2] - right[1]) else right
            if target is left:
                left[2] = run[2]
            else:
                right[1] = run[1]
            del runs[i]
            # Neighbours of the same class now touch: merge them.
            merged: list[list] = []
            for r in runs:
                if merged and merged[-1][0] == r[0]:
                    merged[-1][2] = r[2]
                else:
                    merged.append(r)
            runs = merged
            changed = True
            break
    out = []
    for c, a, b in runs:
        start, end = a * frame, b * frame
        if duration is not None:
            end = min(end, duration)
        if end - start <= 0:
            continue
        chord = chord_of(c)
        if chord is None:
            out.append({"start": round(start, 3), "end": round(end, 3), "label": "N"})
        else:
            root, quality = chord
            out.append({"start": round(start, 3), "end": round(end, 3), "root": root, "quality": quality,
                        "label": harte_label(root, quality)})
    return out


def recognize(y, sr: int, engine: str = "btc", device: str = "cpu") -> list[dict]:
    """The recording's chords as segments, in its own time."""
    import numpy as np

    probs = frame_probabilities(y, sr, engine, device)
    return segments_from_frames(np.argmax(probs, axis=1), FRAME, len(y) / sr)


# --------------------------------------------------------------------------- #
# Chords on beats, and bars from chord changes
# --------------------------------------------------------------------------- #

def chords_per_beat(chords: list[dict], beats: list[float], end: float | None = None) -> list[int | None]:
    """For each beat, the index in `chords` of the chord sounding longest until the next beat."""
    out: list[int | None] = []
    j = 0
    for i, start in enumerate(beats):
        stop = beats[i + 1] if i + 1 < len(beats) else (end if end is not None else start + (
            start - beats[i - 1] if i > 0 else 0.5))
        while j < len(chords) and chords[j]["end"] <= start:
            j += 1
        best, best_overlap, k = None, 0.0, j
        while k < len(chords) and chords[k]["start"] < stop:
            overlap = min(stop, chords[k]["end"]) - max(start, chords[k]["start"])
            if overlap > best_overlap:
                best, best_overlap = k, overlap
            k += 1
        out.append(best)
    return out


def _changes(chords: list[dict], per_beat: list[int | None]) -> list[int]:
    """Beat indexes where a new chord starts (no chord does not count as one)."""
    changes = []
    previous = None
    for i, c in enumerate(per_beat):
        label = chords[c]["label"] if c is not None else "N"
        if label != "N" and label != previous:
            changes.append(i)
        if label != "N":
            previous = label
    return changes


def bar_phase_from_chords(chords: list[dict], beats: list[float], downbeats: list[float],
                          meter: int) -> int | None:
    """The beat index where bar 1 should start, if the chord changes clearly say so.

    Chord changes land on downbeats far more often than elsewhere. Each phase of the bar is
    scored by the share of changes falling on it; the current phase is replaced only when another
    clearly wins (enough changes, at least 55 % of them, and twice the current phase's share).
    Returns None to keep the downbeats as they are.
    """
    if meter < 2 or len(beats) < meter * 4 or not downbeats:
        return None
    per_beat = chords_per_beat(chords, beats)
    changes = _changes(chords, per_beat)[1:]  # the first chord's start says nothing about bars
    if len(changes) < 6:
        return None
    try:
        current = beats.index(downbeats[0]) % meter
    except ValueError:
        return None
    share = [sum(1 for c in changes if c % meter == p) / len(changes) for p in range(meter)]
    best = max(range(meter), key=lambda p: share[p])
    if best == current or share[best] < 0.55 or share[best] < 2 * share[current]:
        return None
    return best


def apply_bar_phase(rhythm: dict, phase: int, meter: int) -> dict:
    """Downbeats and first beat moved to `phase`, beats unchanged; the first bar starts on the
    first beat with that phase at or after the music's start (the old first downbeat, less a bar)."""
    beats = rhythm["beats"]
    old_first = rhythm["downbeats"][0]
    old_index = beats.index(old_first)
    candidates = [i for i in range(len(beats)) if i % meter == phase and i >= old_index - meter + 1 and i >= 0]
    if not candidates:
        return rhythm
    first = candidates[0]
    out = dict(rhythm)
    out["downbeats"] = beats[first::meter]
    out["first_beat"] = beats[first]
    return out
