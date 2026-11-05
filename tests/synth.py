"""Synthetic songs for rhythm tests."""
import numpy as np
import soundfile as sf

SR = 22050


def make_song(dirpath, bpm=72.0, intro_bars=4, bars=16, lead=0.0, subdivide=2, meter=4, seed=0, free_intro=0.0):
    """A pad-only intro (chord change each bar, soft swells), then drums with kick on 1,
    snare on 3, and `subdivide` hi-hats per beat. Returns (first_downbeat_sec, bar_sec).
    With `free_intro` seconds, the song starts with unmetered plucked notes at irregular times (no
    steady beat); the first downbeat is then at `lead + free_intro` and the pad intro is skipped.
    Writes drums.wav, bass.wav, other.wav, vocals.wav (silent)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * meter
    if free_intro:
        intro_bars = 0
    lead_total = lead + free_intro
    total = lead_total + (intro_bars + bars) * bar + 1.0
    n = int(total * SR)
    t = np.arange(n) / SR
    drums = np.zeros(n)
    bass = np.zeros(n)
    other = np.zeros(n)

    def hit(buf, at, sig):
        s = int(at * SR)
        if s < n:
            e = min(n, s + len(sig))
            buf[s:e] += sig[: e - s]

    k = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * 55 * k) * np.exp(-k * 28)
    snare = rng.standard_normal(len(k)) * np.exp(-k * 35) * 0.6
    noise = rng.standard_normal(len(k) + 2)
    hat = np.diff(noise, 2) * np.exp(-k * 90) * 0.15  # second difference: high-passed like a real hi-hat
    chords = [(261.63, 329.63, 392.0), (220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66)]
    for b in range(intro_bars + bars):
        t0 = lead_total + b * bar
        i0, i1 = int(t0 * SR), min(n, int((t0 + bar) * SR))
        f = chords[b % 4]
        seg = t[i0:i1] - t0
        env = np.minimum(1, seg / 0.3) * 0.8
        other[i0:i1] += sum(np.sin(2 * np.pi * x * seg) for x in f) * 0.08 * env
        bass[i0:i1] += np.sin(2 * np.pi * (f[0] / 2) * seg) * 0.2 * env
        if b >= intro_bars:
            for j in range(meter):
                hit(drums, t0 + j * beat, kick if j % 2 == 0 else snare)
                for h in range(subdivide):
                    hit(drums, t0 + j * beat + h * beat / subdivide, hat)
    if free_intro:
        tt = lead
        while tt < lead + free_intro - 0.8:
            f = float(rng.choice([196.0, 246.94, 293.66, 329.63, 392.0]))
            seg = np.arange(int(1.2 * SR)) / SR
            hit(other, tt, np.sin(2 * np.pi * f * seg) * np.exp(-seg * 3) * 0.3)
            tt += float(rng.uniform(0.35, 2.4))  # irregular: rubato / free
    for name, y in (("drums", drums), ("bass", bass), ("other", other), ("vocals", np.zeros(n))):
        sf.write(f"{dirpath}/{name}.wav", (y * 0.9).astype("float32"), SR)
    return lead_total + intro_bars * bar, bar


QUALITY_INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
                     "min7": (0, 3, 7, 10), "sus4": (0, 5, 7), "sus2": (0, 2, 7), "dim": (0, 3, 6)}


def make_progression(dirpath, progression, bpm=90.0, bars=16, meter=4, kick_on=0, seed=1):
    """One chord per bar from `progression` [(root pitch class, quality)], repeated: a sawtooth-ish
    pad (triad with its 2nd and 3rd harmonics, voiced around middle C) in other.wav, the root an
    octave below in bass.wav, and drums with a kick only on beat `kick_on` (0 = the downbeat) and a
    hi-hat on every beat. The first bar starts at 0.5 s. Returns (first_downbeat, bar, beat)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * meter
    first = 0.5
    n = int((first + bars * bar + 1.0) * SR)
    drums, bass, other = np.zeros(n), np.zeros(n), np.zeros(n)
    k = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * 55 * k) * np.exp(-k * 28)
    noise = rng.standard_normal(len(k) + 2)
    hat = np.diff(noise, 2) * np.exp(-k * 90) * 0.15

    def tone(freq, seg):
        return sum(np.sin(2 * np.pi * freq * h * seg) / h for h in (1, 2, 3))

    for b in range(bars):
        t0 = first + b * bar
        i0, i1 = int(t0 * SR), min(n, int((t0 + bar) * SR))
        seg = np.arange(i1 - i0) / SR
        env = np.minimum(1, seg / 0.05) * np.exp(-seg * 0.4)
        root, quality = progression[b % len(progression)]
        base = 261.63 * 2 ** (root / 12)
        if root >= 7:
            base /= 2  # keep voicings around middle C
        other[i0:i1] += sum(tone(base * 2 ** (iv / 12), seg) for iv in QUALITY_INTERVALS[quality]) * 0.06 * env
        bass[i0:i1] += tone(base / 2, seg) * 0.15 * env
        for j in range(meter):
            s = int((t0 + j * beat) * SR)
            if j == kick_on:
                drums[s:s + len(kick)] += kick[: max(0, min(len(kick), n - s))]
            drums[s:s + len(hat)] += hat[: max(0, min(len(hat), n - s))]
    for name, y in (("drums", drums), ("bass", bass), ("other", other), ("vocals", np.zeros(n))):
        sf.write(f"{dirpath}/{name}.wav", (y * 0.9).astype("float32"), SR)
    return first, bar, beat
