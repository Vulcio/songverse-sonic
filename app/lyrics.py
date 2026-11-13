"""Sections from lyrics a client already has (issue #22).

A client that has the song's words - often with chords, sometimes timed (LRC) - sends them with a
job, written out in the order they're sung. That tells the analysis which sections to expect and in
what order, so finding the sections becomes placing a known sequence on the recording's bars
(`align`), instead of guessing how many there are and what each is.

`parse` reads LRC, ChordPro or plain text into passes: one per sung (or played) section, in order,
with its name, its kind (verse, chorus...), how many lines it has, its chords and, from an LRC, when
each line starts. `align` places the passes on the bars by dynamic programming, and says None when
the result doesn't hold up, so the caller keeps the sections found without it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


# Words that name a section's kind, in the languages Songverse's users write in, and the label
# each becomes (the API's section labels).
_KINDS = [
    (r"pre[- ]?chorus|pr[ée][- ]?refrain|pre[- ]?coro|pre|pc", "pre-chorus"),
    (r"chorus|refrain|ref|coro|estribillo|c", "chorus"),
    (r"verse|couplet|strophe|estrofa|verso|v", "verse"),
    (r"bridge|pont|puente|br|b", "bridge"),
    (r"intro|introduction|introducci[oó]n", "intro"),
    (r"outro|ending|fin|final|coda|end", "outro"),
    (r"instrumental|inst|interlude|interlude musical|interludio", "inst"),
    (r"solo", "solo"),
    (r"break|pause", "break"),
    (r"tag|hook", "tag"),
]
_KIND_RE = [(re.compile(rf"^(?:{words})\.?\s*(\d+)?$", re.I), label) for words, label in _KINDS]
_ACC = r"(?:#|b|♯|♭)?"
_CHORD = (rf"[A-G]{_ACC}(?:maj|min|mi|m|M|dim|aug|sus|add|\+|°|ø|-)?\d{{0,2}}"
          rf"(?:(?:sus|add|maj|b|#|\+|-)\d{{1,2}})*(?:\([^)\s]*\))?(?:/[A-G]{_ACC})?")
_CHORD_RE = re.compile(rf"^(?:{_CHORD}|N\.?C\.?)$")
_INLINE_CHORD_RE = re.compile(r"\[([^\]]*)\]")
_LRC_STAMP_RE = re.compile(r"\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]")
_LRC_TAG_RE = re.compile(r"^\[(ar|ti|al|au|by|re|ve|length|offset|la|#):.*\]$", re.I)
_DIRECTIVE_RE = re.compile(r"^\{\s*([a-z_]+)\s*(?::\s*(.*?))?\s*\}$", re.I)
_PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass
class Pass:
    """One section as sung (or played), in order."""
    name: str | None = None          # as the text heads it: "Verse 1", "Chorus"
    kind: str | None = None          # an API section label: "verse", "chorus"...
    lines: list[str] = field(default_factory=list)
    chords: list[tuple[int, bool]] = field(default_factory=list)   # (root pitch class, minor)
    times: list[float] = field(default_factory=list)                # LRC: when each line starts
    group: str | None = None         # unnamed blocks found to be the same section

    @property
    def sung(self) -> bool:
        return bool(self.lines)

    @property
    def music(self) -> str:
        """What it should sound like: passes with the same music (Verse 1, Verse 2) share it."""
        if self.kind:
            return self.kind
        if self.name:
            return _norm(self.name)
        if self.group:
            return self.group
        return "|".join(_norm(line) for line in self.lines[:4]) or f"#{id(self)}"


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def kind_of(heading: str) -> str | None:
    """The section label a heading names ("Verse 2", "Refrain", "Chorus:"), or None."""
    text = heading.strip().strip("[]()*#").strip().rstrip(":").strip()
    text = re.sub(r"\s*(?:x|×)\s*\d+$", "", text, flags=re.I)        # "Chorus x2"
    for pattern, label in _KIND_RE:
        if pattern.match(text):
            return label
    return None


def chord_of(symbol: str) -> tuple[int, bool] | None:
    """(root pitch class, minor) of a chord symbol such as "F#m7/C#", or None."""
    symbol = symbol.strip().replace("♯", "#").replace("♭", "b")
    m = re.match(r"^([A-G])([#b]?)(.*)$", symbol)
    if not m:
        return None
    root = (_PITCH[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)) % 12
    rest = m.group(3).split("/")[0]
    minor = bool(re.match(r"^(m(?!aj)|min|-)", rest))
    return root, minor


def _is_chord_line(line: str) -> bool:
    tokens = [t for t in re.split(r"[\s|]+", line.strip()) if t and t not in {"-", "/", "%", "(", ")"}]
    return bool(tokens) and all(_CHORD_RE.match(t.strip("()")) for t in tokens)


def parse(text: str) -> list[Pass]:
    """The passes in `text` (LRC, ChordPro or plain text, found from the text itself), in order."""
    if not text or not text.strip():
        return []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stamped = sum(1 for line in lines if _LRC_STAMP_RE.match(line.strip()))
    if stamped >= 3 and stamped >= 0.5 * sum(1 for line in lines if line.strip()):
        return _parse_lrc(lines)
    return _parse_text(lines)


def _parse_text(lines: list[str]) -> list[Pass]:
    """ChordPro or plain text: blocks headed by a name ({c: Chorus}, "Verse 1", [Bridge]), or
    between environment directives ({start_of_chorus}), or between blank lines."""
    passes: list[Pass] = []
    current: Pass | None = None
    last_of_kind: dict[str, Pass] = {}
    in_env = False

    def close():
        nonlocal current
        if current is not None and (current.lines or current.chords or current.name):
            passes.append(current)
            if current.kind:
                last_of_kind[current.kind] = current
        current = None

    def heading(name: str | None, kind: str | None):
        nonlocal current
        close()
        current = Pass(name=name or None, kind=kind)

    for raw in lines:
        line = raw.strip()
        if not line:
            # A blank line ends a block (inside {start_of_…} it doesn't: the directive ends it).
            if current is not None and current.lines and not in_env:
                close()
            continue
        d = _DIRECTIVE_RE.match(line)
        if d:
            name, value = d.group(1).lower(), (d.group(2) or "").strip()
            label = re.match(r'^label\s*=\s*"?(.*?)"?$', value)
            if label:
                value = label.group(1)
            env = re.match(r"^(?:start_of_|so)(verse|chorus|bridge|tab|grid|v|c|b)$", name)
            if env:
                part = {"v": "verse", "c": "chorus", "b": "bridge"}.get(env.group(1), env.group(1))
                if part in ("verse", "chorus", "bridge"):
                    heading(value or None, kind_of(value) or part)
                    in_env = True
            elif re.match(r"^(?:end_of_|eo)", name):
                close()
                in_env = False
            elif name in ("c", "comment", "ci", "comment_italic", "cb", "comment_box"):
                kind = kind_of(value)
                if kind:
                    heading(value, kind)
            elif name == "chorus":
                # A recall of the last chorus: sung again.
                close()
                before = last_of_kind.get("chorus")
                passes.append(Pass(name=value or (before.name if before else "Chorus"), kind="chorus",
                                   lines=list(before.lines) if before else ["…"],
                                   chords=list(before.chords) if before else []))
            continue
        if line.startswith("#"):
            continue
        # A heading line: "Chorus", "Verse 2:", "[Bridge]", "(Refrain)".
        bare = line.strip("[]()").strip()
        kind = kind_of(line) if len(line) <= 40 and not _is_chord_line(bare) else None
        if kind:
            heading(line.strip("[]():* ").strip(), kind)
            continue
        if current is None:
            current = Pass()
        if _is_chord_line(line):
            current.chords += [c for c in (chord_of(t.strip("()")) for t in re.split(r"[\s|]+", line) if t) if c]
            continue
        chords = [chord_of(c) for c in _INLINE_CHORD_RE.findall(line)]
        current.chords += [c for c in chords if c]
        words = _INLINE_CHORD_RE.sub("", line).strip()
        if words:
            current.lines.append(words)
    close()
    _group(passes)
    return passes


def _group(passes: list[Pass]) -> None:
    """Unnamed blocks that share most of their lines are the same section (a chorus)."""
    for i, p in enumerate(passes):
        if p.kind or p.name or p.group:
            continue
        mine = {_norm(x) for x in p.lines}
        for q in passes[:i]:
            theirs = {_norm(x) for x in q.lines}
            if mine and theirs and len(mine & theirs) >= max(1.0, 0.6 * min(len(mine), len(theirs))):
                q.group = q.group or f"block {passes.index(q) + 1}"
                p.group = q.group
                break


def _parse_lrc(lines: list[str]) -> list[Pass]:
    """LRC: every stamp of every line (a repeated line may carry several), in time order; blocks at
    blank lines, empty stamped lines or long pauses."""
    timed: list[tuple[float, str]] = []
    offset = 0.0
    last = None
    for raw in lines:
        line = raw.strip()
        if not line:
            if last is not None:      # a blank line ends a block: an empty line just after the last
                timed.append((last + 1e-3, ""))
            continue
        tag = _LRC_TAG_RE.match(line)
        if tag:
            m = re.match(r"^\[offset:\s*([+-]?\d+)\s*\]$", line, re.I)
            if m:
                offset = int(m.group(1)) / 1000.0     # ms; positive = lyrics earlier
            continue
        stamps = []
        while True:
            m = _LRC_STAMP_RE.match(line)
            if not m:
                break
            stamps.append(int(m.group(1)) * 60 + float(m.group(2).replace(":", ".")))
            line = line[m.end():].lstrip()
        words = re.sub(r"<\d{1,3}:\d{1,2}(?:\.\d{1,3})?>", "", line).strip()   # word stamps (enhanced LRC)
        for at in stamps:
            timed.append((at - offset, words))
        if len(stamps) == 1:
            last = stamps[0] - offset
        elif stamps:
            last = None               # a line sung several times: blank lines after it say nothing
    timed.sort(key=lambda x: x[0])
    sung = [(t, w) for t, w in timed if w]
    if not sung:
        return []
    gaps = [b[0] - a[0] for a, b in zip(sung, sung[1:])]
    usual = sorted(gaps)[len(gaps) // 2] if gaps else 4.0
    passes: list[Pass] = []
    current = Pass()
    previous_t = None
    for t, words in timed:
        if not words:
            if current.lines:
                passes.append(current)
                current = Pass()
            continue
        kind = kind_of(words) if len(words) <= 40 else None
        if kind:                        # a heading written as a line
            if current.lines:
                passes.append(current)
            current = Pass(name=words.strip("[]():* "), kind=kind)
            continue
        if previous_t is not None and current.lines and t - previous_t > max(6.0, 2.2 * usual):
            passes.append(current)
            current = Pass()
        current.lines.append(words)
        current.times.append(t)
        previous_t = t
    if current.lines:
        passes.append(current)
    _group(passes)
    return passes


# --------------------------------------------------------------------------- #
# Placing the passes on the bars
# --------------------------------------------------------------------------- #

W_NOVELTY = 1.0      # a pass starting where the music changes
W_TEMPLATE = 1.0     # each bar sounding like the same bar of the other passes of its music
W_VOICE = 0.6        # sung passes have voice, the rest doesn't
W_CHORDS = 0.8       # each bar's chord is one of the pass's
W_TIME = 6.0         # a pass starting where the (fitted) LRC says
LENGTH_STEP = 0.6    # per bar longer or shorter than the other passes of its music
W_LINES = 2.0        # per doubling of bars per line away from the song's
GAP = 1.5            # an unsung stretch between passes (intro and outro: GAP_ENDS)
GAP_ENDS = 0.3
REPEAT = 2.5         # a pass sung once more than the text says
SKIP = 3.0           # a pass of the text not in the recording
MAX_PASS_BARS = 40
MAX_GAP_BARS = 16
NOVELTY_CAP = 2.5
SEED_STARTS = 8
SEED_LENGTHS = (4, 8, 16)


@dataclass
class _Placed:
    hint: int | None      # the pass (index in the text), None for a stretch between passes
    start: int            # bar
    bars: int


def align(passes: list[Pass], bars, novelty, downbeats, bar_len: float, voice=None,
          bar_chords: list[set] | None = None) -> list[dict] | None:
    """Sections from `passes` placed on the recording's bars, or None when they don't fit it.

    `bars` is one feature vector per bar (bar k from downbeats[k]), `novelty` how much the music
    changes at each bar line (about 1 for the song's usual clear change), `voice` 0-1 per bar when
    there is a vocal stem, `bar_chords` the chords (root, minor) heard in each bar when there are
    any. Every section starts on a downbeat; the first one at 0.
    """
    import numpy as np

    n, nb = len(passes), len(bars)
    if n == 0 or nb < 8 or n > nb:
        return None
    unit = bars / (np.linalg.norm(bars, axis=1, keepdims=True) + 1e-9)
    sim = unit @ unit.T
    base = float(np.median(sim[np.triu_indices(nb, 1)]))
    # A clear change counts, a huge one no more than a few: short sections make huge ones.
    nov = np.minimum(np.asarray(novelty, dtype=float), NOVELTY_CAP)
    voice = None if voice is None or float(np.max(voice)) < 0.2 else np.clip(np.asarray(voice, float), 0, 1)
    use_chords = bar_chords is not None and any(bar_chords) and any(p.chords for p in passes)
    ctx = dict(np=np, sim=sim, base=base, nov=nov, voice=voice, nb=nb,
               downbeats=np.asarray(downbeats, float), bar_len=bar_len,
               chords=bar_chords if use_chords else None)

    lines = sum(len(p.lines) for p in passes)
    priors: list[float | None] = [None]
    if lines:
        guess = 0.85 * nb / lines
        priors += sorted({round(guess, 2), 1.0, 2.0, 4.0})
    starts: list[tuple] = [(per_line, None) for per_line in priors]
    # Seeds: the section sung most often (the chorus, usually) taken to be at one of the clearest
    # changes, for a few lengths. Without them a section that follows itself (a chorus sung twice
    # in a row) can't be found: nothing changes where the second one starts.
    counts: dict[str, int] = {}
    for p in passes:
        counts[p.music] = counts.get(p.music, 0) + 1
    music, times_sung = max(counts.items(), key=lambda kv: kv[1])
    if times_sung >= 2:
        candidates = [0] + sorted(range(1, nb), key=lambda k: -nov[k])[:SEED_STARTS - 1]
        for a in candidates:
            for length in SEED_LENGTHS:
                if a + length <= nb:
                    seed = {i: [(a, length)] for i, p in enumerate(passes) if p.music == music}
                    starts.append((None, seed))
    # An LRC: its clock fitted to where the music changes, then every placement follows it.
    timed = _times_on_changes(passes, ctx)
    best = None
    for per_line, seed in starts:
        placed, score = _converge(passes, ctx, per_line, timed, seed=seed)
        if placed is not None and (best is None or score > best[1]):
            best = (placed, score)
    if best is None:
        return None
    placed = best[0]
    if timed is None:
        # Or fitted to the placement found without it.
        timed = _fit_times(passes, placed, ctx)
        if timed is not None:
            again, _ = _converge(passes, ctx, None, timed, placed)
            placed = again or placed
    if not _holds_up(passes, placed, ctx, timed is not None):
        return None
    return _sections(passes, placed, ctx)


def _converge(passes, ctx, per_line, times, start=None, seed=None):
    """Alternate placing the passes and learning what each music sounds like, until it settles."""
    placed = start
    score = float("-inf")
    for step in range(8):
        templates = _templates(passes, placed, ctx) if placed is not None else (seed if step == 0 else None)
        if placed is not None:
            per_line = _bars_per_line(passes, placed) or per_line
        result = _place(passes, ctx, per_line, templates, times)
        if result is None:
            return placed, score
        new, score = result
        if placed is not None and [(p.hint, p.start, p.bars) for p in new] == [(p.hint, p.start, p.bars) for p in placed]:
            break
        placed = new
    return placed, score


def _bars_per_line(passes, placed) -> float | None:
    """The song's bars per line, from where the sung passes are (the median)."""
    ratios = sorted(q.bars / len(passes[q.hint].lines) for q in placed
                    if q.hint is not None and passes[q.hint].lines)
    return ratios[len(ratios) // 2] if ratios else None


def _templates(passes, placed, ctx):
    """For each pass, the other placed passes of the same music (start bar, bars): leave-one-out,
    so a pass is never held in place by itself."""
    out: dict[int, list[tuple[int, int]]] = {}
    for i, p in enumerate(passes):
        out[i] = [(q.start, q.bars) for q in placed
                  if q.hint is not None and q.hint != i and passes[q.hint].music == p.music]
    # A pass sung twice (an extra chorus) is compared with its other placements too.
    return out


def _score_tables(passes, ctx, per_line, templates, times):
    """For each pass i, table[i][a, L-1]: how well it fits starting at bar a for L bars, and
    timing[i][a]: the LRC's say on a start at bar a (not for a pass sung once more)."""
    np, sim, base, nov, voice, nb = (ctx[k] for k in ("np", "sim", "base", "nov", "voice", "nb"))
    lengths = np.arange(1, MAX_PASS_BARS + 1)
    voice_cum = np.concatenate([[0.0], np.cumsum(voice - 0.5)]) if voice is not None else None
    tables, timing = [], []
    for i, p in enumerate(passes):
        table = np.full((nb, MAX_PASS_BARS), -np.inf)
        when = np.zeros(nb)
        others = templates.get(i) if templates else None
        if ctx["chords"] is not None and p.chords:
            mine = set(p.chords)
            match = np.array([0.0 if not c else (0.5 if c & mine else -0.5) for c in ctx["chords"]])
            chord_cum = np.concatenate([[0.0], np.cumsum(match)])
        else:
            chord_cum = None
        for a in range(nb):
            room = min(MAX_PASS_BARS, nb - a)
            row = np.full(room, W_NOVELTY * (NOVELTY_CAP if a == 0 else float(nov[a])))
            ls = lengths[:room]
            if others:
                fit = np.zeros(room)
                for b, other_len in others:
                    m = min(room, other_len, nb - b)
                    diag = sim[a + np.arange(m), b + np.arange(m)] - base
                    cum = np.concatenate([[0.0], np.cumsum(diag)])
                    fit += cum[np.minimum(ls, m)] - LENGTH_STEP * np.abs(ls - other_len)
                row += W_TEMPLATE * fit / len(others)
            if per_line and p.lines:
                # Bars follow lines: about the same number of bars per line all through a song.
                row -= W_LINES * np.abs(np.log2(ls / (per_line * len(p.lines))))
            if voice_cum is not None:
                v = voice_cum[a + ls] - voice_cum[a]
                row += W_VOICE * (v if p.sung else -v)
            if chord_cum is not None:
                row += W_CHORDS * (chord_cum[a + ls] - chord_cum[a])
            table[a, :room] = row
            if times is not None and times[i] is not None:
                off = (float(ctx["downbeats"][a]) - times[i]) / (0.35 * ctx["bar_len"])
                when[a] = W_TIME * (math.exp(-off * off) - 0.5)
        tables.append(table)
        timing.append(when)
    return tables, timing


def _place(passes, ctx, per_line, templates, times):
    """The best placement by dynamic programming over (bar, passes done)."""
    np, nb, voice = ctx["np"], ctx["nb"], ctx["voice"]
    n = len(passes)
    tables, timing = _score_tables(passes, ctx, per_line, templates, times)
    tables = [t.tolist() for t in tables]
    timing = [t.tolist() for t in timing]
    nov = ctx["nov"].tolist()
    voice_cum = np.concatenate([[0.0], np.cumsum(0.5 - voice)]).tolist() if voice is not None else None
    NEG = float("-inf")
    V = [[NEG] * (n + 1) for _ in range(nb + 1)]
    back: list[list[tuple | None]] = [[None] * (n + 1) for _ in range(nb + 1)]
    V[0][0] = 0.0
    for k in range(nb + 1):
        for i in range(n + 1):
            here = V[k][i]
            if here == NEG:
                continue
            if i < n:   # the text's next pass isn't in the recording
                cand = here - SKIP
                if cand > V[k][i + 1]:
                    V[k][i + 1], back[k][i + 1] = cand, (k, i, None, 0)
            if k == nb:
                continue
            # A stretch between passes (intro, instrumental, outro).
            for g in range(1, min(MAX_GAP_BARS, nb - k) + 1):
                cost = GAP_ENDS if (k == 0 or (k + g == nb and i == n)) else GAP
                s = here - cost + W_NOVELTY * (0.0 if k == 0 else 0.3 * nov[k])
                if voice_cum is not None:
                    s += W_VOICE * (voice_cum[k + g] - voice_cum[k])
                if s > V[k + g][i]:
                    V[k + g][i], back[k + g][i] = s, (k, i, -1, g)
            room = min(MAX_PASS_BARS, nb - k)
            if i < n:
                row, when = tables[i][k], here + timing[i][k]
                for length in range(1, room + 1):
                    s = when + row[length - 1]
                    if s > V[k + length][i + 1]:
                        V[k + length][i + 1], back[k + length][i + 1] = s, (k, i, i, length)
            if i > 0:   # the pass just sung, once more
                row = tables[i - 1][k]
                for length in range(1, room + 1):
                    s = here - REPEAT + row[length - 1]
                    if s > V[k + length][i]:
                        V[k + length][i], back[k + length][i] = s, (k, i, i - 1, length)
    if V[nb][n] == NEG:
        return None
    placed: list[_Placed] = []
    k, i = nb, n
    while (k, i) != (0, 0):
        pk, pi, what, length = back[k][i]
        if what is not None:
            placed.append(_Placed(None if what == -1 else what, pk, length))
        k, i = pk, pi
    placed.reverse()
    return placed, V[nb][n]


def _times_on_changes(passes, ctx):
    """An LRC's pass starts on the recording's clock: the offset and speed (within 15 %) that put
    them where the music changes most, or None when none does clearly."""
    np = ctx["np"]
    lrc = [p.times[0] for p in passes if p.times]
    if len(lrc) < 3:
        return None
    downbeats, nov = ctx["downbeats"], ctx["nov"]
    lrc = np.asarray(lrc)
    best = None
    for speed in np.arange(0.87, 1.1501, 0.005):
        for k in range(len(downbeats)):
            offset = downbeats[k] - speed * lrc[0]
            mapped = speed * lrc + offset
            nearest = np.clip(np.searchsorted(downbeats, mapped), 1, len(downbeats) - 1)
            nearest = np.where(np.abs(downbeats[nearest - 1] - mapped) < np.abs(downbeats[nearest] - mapped),
                               nearest - 1, nearest)
            close = np.abs(downbeats[nearest] - mapped) <= 0.35 * ctx["bar_len"]
            score = float(np.sum(np.where(close, np.where(nearest == 0, NOVELTY_CAP, nov[nearest]), 0.0)))
            if best is None or score > best[0]:
                best = (score, speed, offset)
    score, speed, offset = best
    if score < 0.8 * len(lrc):
        return None
    return [speed * p.times[0] + offset if p.times else None for p in passes]


def _fit_times(passes, placed, ctx):
    """Where each pass starts by the LRC's clock mapped onto the recording's (an offset and a speed:
    an LRC is often timed to another release), or None when no mapping fits the placement."""
    lrc = [(i, p.times[0]) for i, p in enumerate(passes) if p.times]
    if len(lrc) < 3:
        return None
    first = {}
    for q in placed:
        if q.hint is not None and q.hint not in first:
            first[q.hint] = float(ctx["downbeats"][q.start])
    pairs = [(t, first[i]) for i, t in lrc if i in first]
    if len(pairs) < 3:
        return None
    tolerance = 1.0 * ctx["bar_len"]
    candidates = [(1.0, 0.0)]
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            (t1, s1), (t2, s2) = pairs[a], pairs[b]
            if t2 - t1 > 1e-6:
                speed = (s2 - s1) / (t2 - t1)
                if 0.85 <= speed <= 1.15:
                    candidates.append((speed, s1 - speed * t1))
        candidates.append((1.0, pairs[a][1] - pairs[a][0]))
    best = None
    for speed, offset in candidates:
        inliers = [(t, s) for t, s in pairs if abs(speed * t + offset - s) <= tolerance]
        if best is None or len(inliers) > len(best[2]):
            best = (speed, offset, inliers)
    speed, offset, inliers = best
    if len(inliers) < max(3, 0.6 * len(pairs)):
        return None
    if len(inliers) >= 2:   # refine on the inliers
        ts = [t for t, _ in inliers]
        ss = [s for _, s in inliers]
        mt, ms = sum(ts) / len(ts), sum(ss) / len(ss)
        var = sum((t - mt) ** 2 for t in ts)
        if var > 0:
            refined = sum((t - mt) * (s - ms) for t, s in inliers) / var
            if 0.85 <= refined <= 1.15:
                speed, offset = refined, ms - refined * mt
    return [speed * p.times[0] + offset if p.times else None for p in passes]


def _holds_up(passes, placed, ctx, timed: bool) -> bool:
    """Whether the placement is worth more than the sections found without the text."""
    np, sim, base, nov = ctx["np"], ctx["sim"], ctx["base"], ctx["nov"]
    sung = [q for q in placed if q.hint is not None]
    hints = {q.hint for q in sung}
    if len(hints) < max(2, 0.7 * len(passes)):
        return False
    if timed:
        return True
    within, between = [], []
    for x in range(len(sung)):
        for y in range(x + 1, len(sung)):
            a, b = sung[x], sung[y]
            m = min(a.bars, b.bars)
            if m <= 0:
                continue
            value = float(np.mean(sim[a.start + np.arange(m), b.start + np.arange(m)]))
            (within if passes[a.hint].music == passes[b.hint].music else between).append(value)
    starts = [q.start for q in placed if q.start > 0]
    changes = float(np.mean([nov[k] for k in starts])) if starts else 0.0
    if within:
        w = float(np.mean(within))
        bt = float(np.mean(between)) if between else base
        return w - bt >= 0.1 and min(within) - base >= 0.1 and changes >= 0.8
    return changes >= 1.0


def _sections(passes, placed, ctx) -> list[dict]:
    # Passes the text didn't name: the music sung most often (twice or more) is the chorus.
    counts: dict[str, int] = {}
    for p in passes:
        if p.sung and not p.kind:
            counts[p.music] = counts.get(p.music, 0) + 1
    chorus = max(counts, key=lambda m: counts[m]) if counts and max(counts.values()) >= 2 \
        and not any(p.kind == "chorus" for p in passes) else None
    out = []
    last = len(placed) - 1
    for idx, q in enumerate(placed):
        start = 0.0 if q.start == 0 else round(float(ctx["downbeats"][q.start]), 3)
        if q.hint is None:
            label = "intro" if idx == 0 else "outro" if idx == last else "inst"
            out.append({"start": start, "label": label})
            continue
        p = passes[q.hint]
        label = p.kind or ("chorus" if p.music == chorus else "verse" if p.sung else "inst")
        section = {"start": start, "label": label, "hint": q.hint}
        if p.name:
            section["name"] = p.name[:80]
        out.append(section)
    return out
