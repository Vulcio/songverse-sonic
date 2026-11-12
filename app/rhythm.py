"""Stem-aware librosa rhythm baseline, independent of jobs and storage.

Inputs are aligned mono sample arrays at the same sample rate, in the original
recording's time coordinates. This function does not load audio, enforce a timeout
or sanitize its raw result. The caller owns those policies. Missing findings stay
absent; _downbeat_period is a private hint for legacy section alignment.

No trained models are required. Keep algorithm changes separate from extraction
so synthetic regressions and later benchmarks can compare the same baseline.
"""
from __future__ import annotations

from typing import Any


TEMPO_PRIOR_CENTER = 92.0    # bpm; log-normal prior over the "counted" tempo
TEMPO_PRIOR_SIGMA = 0.5      # octaves
TEMPO_RANGE = (40.0, 220.0)


def _tempo_prior(bpm: float) -> float:
    import math
    return math.exp(-0.5 * (math.log2(bpm / TEMPO_PRIOR_CENTER) / TEMPO_PRIOR_SIGMA) ** 2)


def _norm_env(env):
    import numpy as np
    scale = np.percentile(env, 95) if env.size else 0.0
    return env / scale if scale > 0 else env


def _pick_tempo(env, sr: int, hop: int) -> tuple[float, float] | None:
    """Choose the tempo with a harmonic comb over the onset-envelope autocorrelation.

    A tempo is scored by the autocorrelation at 1, 2, 3 and 4 beat lags, so a true beat
    period (whose multiples are all accents) beats 3:2 look-alikes. Double/half-time is
    genuinely ambiguous in audio, so the choice rests on a tempo prior and the confidence reflects how contested it is.
    """
    import librosa
    import numpy as np

    fps = sr / hop
    max_lag = int(fps * 60.0 / TEMPO_RANGE[0] * 4) + 2
    if env.size < max_lag * 2:
        return None
    x = env - env.mean()
    ac = librosa.autocorrelate(x, max_size=max_lag + 1)
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lags = np.arange(len(ac))

    def comb(b: float) -> float:
        lag = 60.0 / b * fps
        vals = [float(np.interp(k * lag, lags, ac)) for k in (1, 2, 3, 4) if k * lag <= len(ac) - 1]
        return float(np.mean(vals)) if len(vals) == 4 else 0.0

    grid = np.arange(TEMPO_RANGE[0], TEMPO_RANGE[1], 0.5)
    scores = np.array([max(comb(b), 0.0) for b in grid])
    weighted = scores * np.array([_tempo_prior(b) for b in grid])
    if weighted.max() <= 0:
        return None
    best = float(grid[int(np.argmax(weighted))])

    def local(b: float) -> tuple[float, float]:
        """Best (score, bpm) within +-3% of b."""
        sel = (grid >= b * 0.97) & (grid <= b * 1.03)
        if not sel.any():
            return 0.0, b
        i = int(np.argmax(scores[sel]))
        return float(scores[sel][i]), float(grid[sel][i])

    s_best, best = local(best)
    # A 3:2 look-alike (89 for a 134 BPM song: the prior likes it and every other lag lands on a
    # beat) scores about as well as the true tempo at 3/2 or 3/4 of it, which score near zero otherwise.
    alts = [local(best * r) for r in (1.5, 0.75) if TEMPO_RANGE[0] <= best * r <= TEMPO_RANGE[1]]
    alts = [a for a in alts if a[0] >= 0.9 * s_best]
    if alts:
        s_best, best = max(alts, key=lambda a: a[0] * _tempo_prior(a[1]))
    lower = None
    if best / 2 >= 30.0:
        s_half, b_half = local(best / 2)
        lower = (s_half, b_half)
    higher = local(best * 2) if best * 2 <= TEMPO_RANGE[1] else None

    # Double/half time cannot be told apart from periodicity alone (all octaves score within a
    # few percent), so the prior decides. Near an octave boundary the call is a coin flip:
    # report that as a low confidence rather than a confident wrong answer.
    ratio = 0.0
    for alt in (lower, higher):
        if alt and alt[0] >= 0.75 * s_best:
            ratio = max(ratio, _tempo_prior(alt[1]) / _tempo_prior(best))
    conf = min(1.0, max(0.0, s_best / 0.35))
    if ratio >= 0.7:
        conf = min(conf, 0.45)   # coin flip between octaves: below 0.5 clients ignore it
    elif ratio >= 0.4:
        conf = min(conf, 0.6)    # prior-driven choice, plausible alternative exists
    return best, conf



ATTACK_HOP = 32          # samples: ~1.5 ms at 22050 Hz
ATTACK_WINDOW = 256      # samples: ~12 ms
ATTACK_SEARCH = 0.06     # s each side of a tracked beat, at most a fifth of a beat
ATTACK_LEAD = 0.0045     # s: the curve's peak comes this much before a hit starts (measured on synthetic hits)


def _attack_curve(y, sr: int):
    """How fast the level rises, every ~1.5 ms: its peaks are where notes and hits start.

    The beat tracker's frames are 23 ms wide and its onset envelope peaks after the attack, so
    its beats land a little late and wobble. This finer curve puts each beat on its attack.
    """
    import librosa
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    # Below 2 kHz (kick, snare body, bass, chords): sixteenth hi-hats in a quiet stretch would
    # otherwise rise as steeply as the kick and snare, a sixteenth either side of the beat.
    # Zero-phase, so it moves no attack.
    y = sosfiltfilt(butter(4, 2000, btype="low", fs=sr, output="sos"), y)
    rms = librosa.feature.rms(y=y, frame_length=ATTACK_WINDOW, hop_length=ATTACK_HOP, center=True)[0]
    level = np.log(rms + 1e-4)
    rise = np.maximum(0.0, np.diff(level, prepend=level[0]))
    # A touch of smoothing (3 points) so one noisy step doesn't win over the hit itself.
    return np.convolve(rise, np.ones(3) / 3, mode="same")


def _snap_to_attacks(times, curve, sr: int, beat: float):
    """Each beat moved onto the attack nearest it, where there's a clear one; the others moved by
    the beats' usual offset. Returns (times, which beats had their own attack)."""
    import numpy as np

    fps = sr / ATTACK_HOP
    # The tracker can sit a steady 50-100 ms off the hits at fast tempos: first find its usual
    # offset from the attacks summed over every beat (within a third of a beat), then look near that.
    times = np.asarray(times, dtype=float)
    span = int(0.3 * beat * fps)
    profile = np.zeros(2 * span + 1)
    for t in times:
        c = int(round((t - ATTACK_LEAD) * fps))
        if c - span >= 0 and c + span + 1 <= len(curve):
            profile += curve[c - span: c + span + 1]
    if profile.max() > 0:
        times = times + (int(np.argmax(profile)) - span) / fps
    reach = min(ATTACK_SEARCH, 0.2 * beat)
    found = np.full(len(times), np.nan)
    peaks = np.zeros(len(times))
    for i, t in enumerate(times):
        a, b = int((t - reach) * fps), int((t + reach) * fps) + 1
        a, b = max(0, a), min(len(curve), b)
        if b - a < 3:
            continue
        j = a + int(np.argmax(curve[a:b]))
        if 0 < j - a < b - a - 1:   # a peak inside the window, not a rise running past its edge
            found[i] = j / fps + ATTACK_LEAD
            peaks[i] = curve[j]
    strong = peaks > 0
    floor = 0.0
    if strong.sum() >= 4:
        floor = 0.4 * float(np.median(peaks[strong]))
        strong &= peaks >= floor
    if strong.sum() < 4:
        return np.asarray(times, dtype=float), np.zeros(len(times), dtype=bool)
    offsets = found[strong] - np.asarray(times)[strong]
    usual = float(np.median(offsets))
    out = np.asarray(times, dtype=float) + usual
    # A beat's own attack, unless it's far from where the others say (a fill, a hi-hat before it).
    own = strong & (np.abs((found - np.asarray(times)) - usual) <= 0.03)
    out[own] = found[own]
    # An attack well off the line its neighbours draw (the last hi-hats as the tracker runs out of
    # beats, a fill) isn't this beat's.
    near = min(0.025, 0.1 * beat)
    mine = np.nonzero(own)[0]
    wrong = []
    for i in mine:
        others = mine[(mine != i) & (np.abs(mine - i) <= 16)]
        others = others[np.argsort(np.abs(others - i))[:8]]
        if len(others) >= 4:
            slope, icpt = np.polyfit(others, out[others], 1)
            if abs(out[i] - (icpt + slope * i)) > near:
                wrong.append(i)
    own[wrong] = False
    # The tracker wanders by more than that in places (a few beats running 40 ms early). Where it
    # did, look again where the beats around say this one should be, and take the attack there.
    for _ in range(4):
        before = int(own.sum())
        for i in np.nonzero(~own)[0]:
            near_own = np.nonzero(own)[0]
            near_own = near_own[np.abs(near_own - i) <= 16]
            idx = near_own[np.argsort(np.abs(near_own - i))[:8]]
            if len(idx) < 3:
                continue
            slope, icpt = np.polyfit(idx, out[idx], 1)
            pred = icpt + slope * i
            a, b = max(0, int((pred - near - ATTACK_LEAD) * fps)), min(len(curve), int((pred + near - ATTACK_LEAD) * fps) + 1)
            if b - a >= 3:
                j = a + int(np.argmax(curve[a:b]))
                if 0 < j - a < b - a - 1 and curve[j] >= floor:
                    out[i], own[i] = j / fps + ATTACK_LEAD, True
                    continue
            out[i] = pred
        if int(own.sum()) == before:
            break
    return out, own



def _grid_fit(T, start: int, own=None, count: int = 16):
    """A steady grid through beats T[start:start+count] (least squares; those on an attack of their
    own when there are enough), as a function of beat index."""
    import numpy as np

    idx = np.arange(start, min(len(T), start + count))
    if own is not None and int(own[idx].sum()) >= 4:
        idx = idx[own[idx]]
    if len(idx) < 2:
        step = float(np.median(np.diff(T))) if len(T) > 1 else 0.5
        base = float(T[start])
        return lambda i: base + (i - start) * step
    slope, intercept = np.polyfit(idx, T[idx], 1)
    return lambda i: float(intercept + slope * i)


def _drum_flux(y_drums, sr: int, hop: int):
    """Kick (30-150 Hz) and snare-body (150-2500 Hz) onset flux; hi-hats barely register."""
    import librosa
    import numpy as np

    S = np.abs(librosa.stft(y_drums, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def flux(lo: float, hi: float):
        band = np.log1p(S[(freqs >= lo) & (freqs < hi)].sum(axis=0) * 1e3)
        return np.maximum(0.0, np.diff(band, prepend=band[0]))

    return flux(30, 150), flux(150, 2500)


def _kick_flux(y, sr: int, hop: int, bottom: float = 30.0, top: float = 150.0):
    """Rise of the energy between `bottom` and `top` Hz (30-150: the kick), linear rather than
    logarithmic so hi-hats (and below 150 Hz the snare's noise), tiny down there, don't count as
    much as a kick."""
    import librosa
    import numpy as np

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = S[(freqs >= bottom) & (freqs < top)].sum(axis=0)
    return np.maximum(0.0, np.diff(band, prepend=band[0]))


def _harmonic_novelty(y, sr: int, hop: int, beat_frames):
    """z-scored chroma change across each beat (current beat vs the previous one)."""
    import librosa
    import numpy as np

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=hop)
    out = np.zeros(len(beat_frames))
    for i in range(1, len(beat_frames) - 1):
        a, b, c = beat_frames[i - 1], beat_frames[i], beat_frames[i + 1]
        if b <= a or c <= b or c > chroma.shape[1]:
            continue
        prev, cur = chroma[:, a:b].mean(axis=1), chroma[:, b:c].mean(axis=1)
        out[i] = float(np.linalg.norm(cur - prev))
    if out.std() < 1e-9:
        return None
    return (out - out.mean()) / out.std()


def _music_span(y, sr: int) -> tuple[float, float]:
    """(start, end) in seconds of the clearly audible part (skips leading/trailing silence)."""
    import librosa
    import numpy as np
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    if rms.size == 0 or rms.max() <= 0:
        return 0.0, len(y) / sr
    idx = np.nonzero(rms > 0.03 * rms.max())[0]
    if not idx.size:
        return 0.0, len(y) / sr
    return (float(librosa.frames_to_time(idx[0], sr=sr, hop_length=512)),
            float(librosa.frames_to_time(idx[-1] + 1, sr=sr, hop_length=512)))


def analyze_rhythm(y_full, y_rhythm, sr: int, y_drums=None, y_harm=None) -> dict:
    """Tempo, first downbeat and meter.

    Beats are tracked on the full mix plus the drums/bass, so a drumless intro still gets a
    beat grid; the downbeat phase comes from low-frequency accents (kick/bass), and the grid
    is extended back over the intro so `first_beat` is where bar 1 starts, not where the drums enter.
    """
    import librosa
    import numpy as np

    hop = 512
    env_full = _norm_env(librosa.onset.onset_strength(y=y_full, sr=sr, hop_length=hop))
    env_rhy = _norm_env(librosa.onset.onset_strength(y=y_rhythm, sr=sr, hop_length=hop))
    n = min(len(env_full), len(env_rhy))
    env = env_full[:n] + env_rhy[:n]
    if n < 8 or not np.any(env > 0):
        return {}
    # Hi-hats and other busy high-frequency detail make the broadband envelope look like
    # double time. The tempo is chosen on a band below 1.5 kHz (kick, snare body, bass, chords).
    env_lo = (_norm_env(librosa.onset.onset_strength(y=y_full, sr=sr, hop_length=hop, fmax=1500))[:n]
              + _norm_env(librosa.onset.onset_strength(y=y_rhythm, sr=sr, hop_length=hop, fmax=1500))[:n])

    tempo_env = env_lo
    if y_drums is not None:
        low, mid = _drum_flux(y_drums, sr, hop)
        d_env = _norm_env(low)[:n] + _norm_env(mid)[:n]
        if len(d_env) == n and np.percentile(d_env, 95) > 0 and (d_env > 0.5).mean() > 0.02:
            tempo_env = d_env  # drum accents are the cleanest counting cue when there are drums

    picked = _pick_tempo(tempo_env, sr, hop)
    if picked is None:
        return {}
    bpm, tempo_conf = picked
    result: dict[str, Any] = {"tempo": {"bpm": bpm, "confidence": tempo_conf}}

    # Track beats on drum accents plus the low band of the mix, so a drumless intro is covered too.
    beat_env = env_lo + (tempo_env if tempo_env is not env_lo else 0.0)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=beat_env, sr=sr, hop_length=hop, bpm=bpm, tightness=120, trim=False)
    beat_frames = np.asarray(beat_frames, dtype=int)
    if len(beat_frames) < 4:
        return result
    # Half time: the prior favours ~92 BPM, so a 150 BPM song with kick and snare alternating on
    # every beat can be tracked at 75 with its "beats" all on the snare, the kicks halfway between
    # (the 1 then lands on a 2). When the kick hits as hard halfway between the tracked beats as on
    # them, those midpoints are beats too: count twice as fast.
    kick = _kick_flux(y_drums if y_drums is not None else y_rhythm, sr, hop)
    if bpm * 2 <= 200 and np.percentile(kick, 99) > 0:
        mids = ((beat_frames[:-1] + beat_frames[1:]) // 2)
        def peak(frames):
            return np.array([kick[max(0, i - 1): i + 2].max() for i in frames if i < len(kick)])
        on, mid = peak(beat_frames), peak(mids)
        if len(mid) >= 8 and mid.mean() > 0 and mid.mean() >= 0.7 * on.mean():
            bpm *= 2
            result["tempo"]["bpm"] = bpm
            _, beat_frames = librosa.beat.beat_track(onset_envelope=beat_env, sr=sr, hop_length=hop, bpm=bpm, tightness=120, trim=False)
            beat_frames = np.asarray(beat_frames, dtype=int)
            if len(beat_frames) < 4:
                return result
    # Double time, the other way round: a 60 BPM ballad with eighth-note hi-hats counted at 120,
    # every other "beat" a hi-hat alone. When the kick and snare play only on one beat in two,
    # count half as fast, on theirs.
    elif y_drums is not None and bpm / 2 >= TEMPO_RANGE[0]:
        def unit(x):
            return x / x.max() if x.max() > 0 else x
        body = unit(_kick_flux(y_drums, sr, hop)) + unit(_kick_flux(y_drums, sr, hop, 150.0, 2000.0))   # kick + snare
        hits = np.array([body[max(0, i - 1): i + 2].max() if i < len(body) else 0.0 for i in beat_frames])
        if hits.max() > 0:
            pairs = len(hits) // 2
            even, odd = hits[0: 2 * pairs: 2], hits[1: 2 * pairs: 2]
            busy = np.maximum(even, odd) > 0.1 * hits.max()   # pairs where the drums play
            if busy.sum() >= 8:
                e, o = float(even[busy].mean()), float(odd[busy].mean())
                if min(e, o) < 0.15 * max(e, o):
                    beat_frames = beat_frames[(0 if e > o else 1)::2]
                    bpm /= 2
                    result["tempo"]["bpm"] = bpm

    # Sub-frame refinement (parabolic peak interpolation): frames are 23 ms wide, which would
    # otherwise make the median beat interval flip between two values.
    idx = beat_frames[beat_frames < len(beat_env)]
    refined = []
    for i in idx:
        i = int(i)
        if 0 < i < len(beat_env) - 1:
            y0, y1, y2 = beat_env[i - 1], beat_env[i], beat_env[i + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
            i = i + float(np.clip(delta, -0.5, 0.5))
        refined.append(i)
    T = np.asarray(refined) * hop / sr
    # Each beat on its attack (the tracker's are ~10-25 ms late and wobble): from the drums when
    # there are some, else the rhythm part.
    source = y_drums if y_drums is not None and np.abs(y_drums).max() > 1e-3 else y_rhythm
    T, own = _snap_to_attacks(T, _attack_curve(source, sr), sr, float(np.median(np.diff(T))))
    if np.any(np.diff(T) <= 0):
        return result
    if own.any():
        # Past the last hit the tracker runs on into the ring-out (or onto the last hi-hats, out of
        # step): those are not beats.
        last = int(np.nonzero(own)[0][-1]) + 1
        step = float(np.median(np.diff(T)))
        while last > 8 and abs((T[last - 1] - T[last - 2]) - step) > 0.08 * step:
            last -= 1
        T, own = T[:last], own[:last]
        if len(T) < 8:
            return result
    # Meter / downbeat phase from low-frequency (kick/bass) accents per beat.
    # Sampled where each beat really starts (its attack, found above), a frame either side: the
    # tracker's own frames can sit 60 ms off, between a kick and the next beat.
    low = librosa.onset.onset_strength(y=y_rhythm, sr=sr, hop_length=hop, fmax=300)
    idx = np.clip(np.round(T * sr / hop).astype(int), 0, min(len(low), len(env_rhy), len(beat_env)) - 1)

    def around(env):
        return np.array([env[max(0, i - 1): i + 2].max() for i in idx])

    strength = around(low) + 0.5 * around(env_rhy)
    if len(strength) < 8:
        return result
    strength = (strength - strength.mean()) / (strength.std() + 1e-9)
    # Chord changes usually fall on bar lines: add the harmonic novelty at each beat. It also
    # works in a drumless intro, where the kick cue is silent.
    novelty = _harmonic_novelty(y_harm if y_harm is not None else y_full, sr, hop, idx)
    if novelty is not None:
        strength = strength + 1.2 * novelty

    scores: dict[int, tuple[float, int]] = {}
    for m in (3, 4):
        phase_means = [float(strength[p::m].mean()) for p in range(m)]
        best = int(np.argmax(phase_means))
        others = [v for i, v in enumerate(phase_means) if i != best]
        scores[m] = (phase_means[best] - float(np.mean(others)), best)
    # Prior toward 4/4: only call 3/4 when its accent pattern is clearly stronger.
    meter = 3 if scores[3][0] > scores[4][0] * 1.35 + 0.1 else 4
    contrast, phase = scores[meter]
    result["time_signature"] = {
        "numerator": meter, "denominator": 4, "confidence": min(1.0, max(0.0, contrast / 1.5)),
    }

    duration = len(y_full) / sr
    beat_est = float(np.median(np.diff(T)))
    bar_est = beat_est * meter
    s_beat = np.array([beat_env[max(0, i - 1): i + 2].max() for i in idx])
    level = float(np.percentile(s_beat, 60))

    onset, music_end = _music_span(y_full, sr)
    target = onset - 0.35 * beat_est
    hit = s_beat >= 0.5 * level if level > 0 else np.zeros(len(s_beat), dtype=bool)
    # Drop trailing beats after the last one that lands on an onset: the tracker's edge
    # effects and the ring-out of the last chord are not beats.
    if hit.any():
        last = int(np.nonzero(hit)[0][-1])
        T, s_beat, hit = T[: last + 1], s_beat[: last + 1], hit[: last + 1]
    dn = np.arange(phase, len(T), meter)
    if len(dn) < 2:
        return result
    ahead = [i for i in dn if T[i] >= target - 1e-6]
    i0 = int(ahead[0]) if ahead else int(dn[0])

    # First downbeat with a sustained beat behind it (where drums / the band really come in):
    # at least 7 of the next 8 beats land on an onset. Isolated notes of a free intro do not.
    supported = None
    for i in dn[dn >= i0]:
        if hit[i: i + 8].sum() >= 7 and len(hit[i: i + 8]) >= 8:
            supported = int(i)
            break
    if supported is None:
        intro_free = None
        supported = i0
    else:
        # Periodicity is measured at the beat lag, so it is only meaningful when the tempo is trusted.
        intro_free = (_intro_free(beat_env, sr, hop, onset, T[supported], T[supported] - T[i0], beat_est, level, s_beat)
                      if tempo_conf >= 0.5 else None)
        if intro_free and _intro_on_grid(beat_env, sr, hop, onset, T, supported, own):
            # A few soft notes are too few to judge periodicity by, but where each one lands on the
            # grid of the steady part carried back, the intro is in time (a pad's chord changes).
            intro_free = False
    first_idx = supported if intro_free else i0

    if not intro_free:
        # Before the band comes in (a pad or a quiet intro, or the very first hit) the tracker's
        # beats have no attack to stand on and stray by tens of ms: carry the grid of the first
        # steady bars back over them, keeping only beats with an attack of their own.
        T = T.copy()
        fit = _grid_fit(T, supported, own)
        for i in range(first_idx, supported + 1):
            if not own[i]:
                T[i] = fit(i)
        if np.any(np.diff(T) <= 0):
            return result
    beats = T[first_idx:]
    first = float(T[first_idx])
    if not intro_free:
        # The grid may start later than the music (no tracked beats over a quiet intro): extend it
        # back, bar by bar, to where the music starts, on the same fitted grid. Only for a steady intro.
        fit = _grid_fit(T, supported, own)
        k = first_idx
        while fit(k - meter) >= target and fit(k - meter) >= -0.05:
            k -= meter
        if k < first_idx:
            beats = np.concatenate([np.array([fit(i) for i in range(k, first_idx)]), beats])
            first = float(beats[0])
    # A first beat a few ms before 0:00 (the grid's own error) is the start.
    first = max(0.0, first)
    beats = beats[(beats >= -0.05) & (beats <= min(duration, music_end + 0.5 * beat_est))]
    beats = np.maximum(beats, 0.0)
    if len(beats) < 4:
        return result
    k0 = int(np.argmin(np.abs(beats - first)))
    downbeats = beats[k0::meter]

    beats_r = np.round(beats, 3)
    downbeats_r = np.round(downbeats, 3)
    if np.any(np.diff(beats_r) <= 0):
        return result
    result["tempo"]["bpm"] = 60.0 / float(np.median(np.diff(beats_r)))
    result["first_beat"] = float(downbeats_r[0])
    result["beats"] = [float(t) for t in beats_r]
    result["downbeats"] = [float(t) for t in downbeats_r]
    if intro_free is not None:
        result["intro_free"] = intro_free
    on_onset = float(hit[first_idx:].mean()) if len(hit[first_idx:]) else 0.0
    result["beats_confidence"] = min(on_onset, tempo_conf)
    result["_downbeat_period"] = bar_est
    return result


def _periodicity(seg, lag: float) -> float:
    """Strongest normalised autocorrelation of `seg` at 1-4 beat lags (0 when too short)."""
    import librosa
    import numpy as np

    lags = [k for k in (1, 2, 3, 4) if k * lag * 2 < len(seg)]
    if len(lags) < 4:   # needs the bar-length lag too, or a bar-periodic intro reads as aperiodic
        return -1.0
    x = seg - seg.mean()
    ac = librosa.autocorrelate(x, max_size=int(max(lags) * lag) + 3)
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    xs = np.arange(len(ac))
    best = 0.0
    for k in lags:
        centre = k * lag
        sel = (xs >= centre * 0.97) & (xs <= centre * 1.03)
        if sel.any():
            best = max(best, float(ac[sel].max()))
    return best


def _periodicity_any(seg, fps: float) -> float:
    """Highest autocorrelation peak of `seg` at any lag between 0.3 s and 4 s (-1 if too short).

    Independent of the tempo estimate: a steady intro repeats at some lag (beat, half bar, bar);
    a free one does not.
    """
    import librosa
    import numpy as np

    lo, hi = int(0.3 * fps), int(min(4.0 * fps, len(seg) / 2))
    if hi - lo < 4:
        return -1.0
    x = seg - seg.mean()
    ac = librosa.autocorrelate(x, max_size=hi + 2)
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    peaks = [float(ac[i]) for i in range(max(lo, 1), min(hi, len(ac) - 1)) if ac[i] >= ac[i - 1] and ac[i] > ac[i + 1]]
    return max(peaks) if peaks else 0.0


def _intro_on_grid(beat_env, sr: int, hop: int, onset: float, T, supported: int, own) -> bool:
    """Whether the intro's onsets (at least three) nearly all fall on the beat grid of the steady
    part carried back over it: a swell may peak a little after its beat, hardly before."""
    import librosa
    import numpy as np

    fit = _grid_fit(T, supported, own)
    step = fit(supported + 1) - fit(supported)
    frames = librosa.onset.onset_detect(onset_envelope=beat_env, sr=sr, hop_length=hop, units="frames")
    times = frames * hop / sr
    # The music's own start counts: the detector misses an onset at the very edge of the signal.
    times = np.concatenate([[onset], times[(times >= onset + 0.25 * step) & (times < fit(supported) - 0.5 * step)]])
    if len(times) < 3 or step <= 0:
        return False
    pos = (times - fit(supported)) / step
    off = pos - np.round(pos)
    return float(np.mean((off >= -0.08) & (off <= 0.15))) >= 0.75


def _intro_free(beat_env, sr: int, hop: int, onset: float, t_sup: float, pre_len: float,
                beat_est: float, level: float, s_beat) -> bool | None:
    """True: no steady beat before the first supported downbeat (free/rubato/spoken intro).
    False: the beat is steady before it (or there is nothing before it). None: can't tell.

    Conservative: compares how periodic the onset envelope is before vs after that downbeat,
    and only answers when the intro is long enough (>= 8 beats) to judge.
    """
    import numpy as np

    if pre_len < 1.5 * beat_est:
        return False
    if pre_len < 8 * beat_est or level <= 0:
        return None
    fps = sr / hop
    lag = beat_est * fps
    a, b = int(onset * fps), int(t_sup * fps)
    post_end = min(len(beat_env), b + int(40 * fps))
    pre, post = beat_env[a:b], beat_env[b:post_end]
    pre_p, post_p = _periodicity(pre, lag), _periodicity(post, lag)
    pre_any = _periodicity_any(pre, fps)
    if pre_p < 0 or pre_any < 0 or post_p < 0.25:
        return None
    if pre_p >= 0.6 * post_p:
        return False
    # Free only when clearly aperiodic, in absolute terms as well as relative to the groove.
    if pre_p < 0.25 and pre_p < 0.25 * post_p and pre_any < 0.3:
        return True
    return None
