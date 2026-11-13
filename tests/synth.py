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


def make_sections(dirpath, plan=(("A", 8), ("B", 8), ("A", 8), ("B", 8), ("C", 4), ("B", 8)), bpm=96.0, meter=4, lead=0.4, seed=2):
    """Sections that differ in chords, bass and drums: A soft (pad, kick on 1), B full (chords
    changing every two beats, hi-hats on eighths, snare on the backbeat), C sparse (one held chord,
    no drums). `plan` is (name, bars) in order. Returns (section starts in seconds, bar_sec)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * meter
    total_bars = sum(b for _, b in plan)
    n = int((lead + total_bars * bar + 1.0) * SR)
    drums, bass, other = np.zeros(n), np.zeros(n), np.zeros(n)
    k = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * 55 * k) * np.exp(-k * 28)
    snare = rng.standard_normal(len(k)) * np.exp(-k * 35) * 0.6
    hat = np.diff(rng.standard_normal(len(k) + 2), 2) * np.exp(-k * 90) * 0.15

    def hit(buf, at, sig):
        s = int(at * SR)
        if s < n:
            e = min(n, s + len(sig))
            buf[s:e] += sig[: e - s]

    def chord(at, length, freqs, level):
        i0, i1 = int(at * SR), min(n, int((at + length) * SR))
        seg = np.arange(i1 - i0) / SR
        env = np.minimum(1, seg / 0.02) * np.exp(-seg * 0.5)
        other[i0:i1] += sum(np.sin(2 * np.pi * f * seg) + 0.4 * np.sin(4 * np.pi * f * seg) for f in freqs) * level * env
        bass[i0:i1] += np.sin(2 * np.pi * freqs[0] / 2 * seg) * 0.25 * env

    progressions = {
        "A": [(261.63, 329.63, 392.0), (196.0, 246.94, 293.66)],                       # C G
        "B": [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66), (261.63, 329.63, 392.0)],  # Am F G C
        "C": [(293.66, 349.23, 440.0)],                                                  # Dm
    }
    starts, t = [], lead
    for name, bars in plan:
        starts.append(t)
        for b in range(bars):
            t0 = t + b * bar
            prog = progressions[name]
            if name == "B":
                for half in range(2):
                    chord(t0 + half * bar / 2, bar / 2, prog[(2 * b + half) % len(prog)], 0.07)
            else:
                chord(t0, bar, prog[b % len(prog)], 0.05 if name == "A" else 0.04)
            for j in range(meter):
                at = t0 + j * beat
                if name == "A":
                    if j == 0:
                        hit(drums, at, kick * 0.7)
                elif name == "B":
                    hit(drums, at, kick if j % 2 == 0 else snare)
                    hit(drums, at, hat)
                    hit(drums, at + beat / 2, hat)
        t += bars * bar
    for name, y in (("drums", drums), ("bass", bass), ("other", other), ("vocals", np.zeros(n))):
        sf.write(f"{dirpath}/{name}.wav", (y * 0.9).astype("float32"), SR)
    return starts, bar


def make_click(path, beats, downbeats=(), duration=None, accent="pitch", jitter=0.0, seed=4):
    """A click track like a multitrack's click stem: a short tone on every beat, the downbeats
    accented by a higher pitch (`accent="pitch"`), by level ("level") or not at all (None).
    `jitter` seconds of random delay per click stand for a click rendered off the grid."""
    rng = np.random.default_rng(seed)
    n = int(((duration or (max(beats) + 1.0))) * SR)
    y = np.zeros(n)
    k = np.arange(int(0.03 * SR)) / SR
    down = {round(d, 6) for d in downbeats}
    for b in beats:
        strong = round(b, 6) in down
        f = 1600.0 if strong and accent == "pitch" else 1000.0
        level = 0.9 if strong and accent == "level" else 0.5
        tone = np.sin(2 * np.pi * f * k) * np.exp(-k * 160) * level
        s = int((b + (rng.uniform(0, jitter) if jitter else 0.0)) * SR)
        if s < n:
            e = min(n, s + len(tone))
            y[s:e] += tone[: e - s]
    sf.write(path, y.astype("float32"), SR)


def make_compound(dirpath, eighths_bpm=138.0, bars=16, lead=0.6, seed=5, hats=True):
    """A 6/8 song: kick on the first eighth, snare on the fourth (the second dotted quarter),
    hi-hats on every eighth, a chord change each bar. Writes the four stems like make_song;
    returns (first_downbeat_sec, eighth_sec). Without `hats`, the drums play only the kick and the
    snare (a 4/4 half-time bar as much as a 6/8 one) and the keys play the eighths, an arpeggio."""
    rng = np.random.default_rng(seed)
    e = 60.0 / eighths_bpm
    bar = 6 * e
    n = int((lead + bars * bar + 1.0) * SR)
    t = np.arange(n) / SR
    drums, bass, other = np.zeros(n), np.zeros(n), np.zeros(n)
    k = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * 55 * k) * np.exp(-k * 28)
    snare = rng.standard_normal(len(k)) * np.exp(-k * 35) * 0.6
    hat = np.diff(rng.standard_normal(len(k) + 2), 2) * np.exp(-k * 90) * 0.15

    def hit(buf, at, sig):
        s = int(at * SR)
        if s < n:
            buf[s:min(n, s + len(sig))] += sig[: min(n, s + len(sig)) - s]

    chords = [(196.0, 246.94, 293.66), (164.81, 196.0, 246.94), (130.81, 164.81, 196.0), (146.83, 185.0, 220.0)]
    for b in range(bars):
        t0 = lead + b * bar
        i0, i1 = int(t0 * SR), min(n, int((t0 + bar) * SR))
        seg = t[i0:i1] - t0
        f = chords[b % 4]
        env = np.minimum(1, seg / 0.2) * 0.8
        if hats:
            other[i0:i1] += sum(np.sin(2 * np.pi * x * seg) for x in f) * 0.08 * env
        else:
            for j in range(6):
                note = np.arange(int(e * SR)) / SR
                hit(other, t0 + j * e, np.sin(2 * np.pi * 2 * f[j % 3] * note) * np.exp(-note * 9) * 0.25)
        bass[i0:i1] += np.sin(2 * np.pi * (f[0] / 2) * seg) * 0.2 * env
        hit(drums, t0, kick)
        hit(drums, t0 + 3 * e, snare)
        for j in range(6 if hats else 0):
            hit(drums, t0 + j * e, hat)
    for name, y in (("drums", drums), ("bass", bass), ("other", other), ("vocals", np.zeros(n))):
        sf.write(f"{dirpath}/{name}.wav", (y * 0.9).astype("float32"), SR)
    return lead, e
