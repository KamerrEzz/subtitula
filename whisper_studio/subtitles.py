from dataclasses import dataclass
from functools import lru_cache

from whisper_studio.models import Segment

PAUSE_BREAK_S = 0.6
SENTENCE_PAUSE_S = 0.3
SENTENCE_MIN_FILL = 0.25
CLAUSE_MIN_FILL = 0.6
CONNECTOR_MIN_FILL = 0.5
BACKTRACK_MIN_FILL = 0.4
MIN_SHRINK_S = 0.2
MIN_POSITIVE_S = 0.1

_SENTENCE_END = (".", "!", "?", "…")
_CLAUSE_END = (",", ";", ":")
_GLUE_START = ".,;:!?…)”»’'%"
_NO_LINE_START = ".,;:!?…)”»%"
_TRAILING_QUOTES = "\"'”»’)]"
_CONNECTORS = frozenset({"y", "o", "pero", "que", "porque", "cuando", "and", "but", "so", "because"})


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]


@dataclass
class _Token:
    text: str
    start: float
    end: float


def _clean(text: str) -> str:
    return " ".join(text.split())


def _ends_sentence(text: str) -> bool:
    return text.rstrip(_TRAILING_QUOTES).endswith(_SENTENCE_END)


def _ends_clause(text: str) -> bool:
    return text.rstrip(_TRAILING_QUOTES).endswith(_CLAUSE_END)


def _is_connector(text: str) -> bool:
    return text.strip(".,;:!?¿¡…\"'").lower() in _CONNECTORS


def _word_tokens(seg: Segment, tokens: list[_Token]) -> None:
    for word in seg.words or ():
        text = _clean(word.word)
        if not text:
            continue
        # Whisper emits trailing punctuation as its own word; it must stay attached to the previous one.
        if tokens and text[0] in _GLUE_START:
            tokens[-1].text += text
            tokens[-1].end = max(tokens[-1].end, word.end)
        else:
            tokens.append(_Token(text, word.start, word.end))


def _proportional_tokens(seg: Segment) -> list[_Token]:
    words = _clean(seg.text).split()
    if not words:
        return []
    total = sum(len(w) for w in words) + len(words) - 1
    span = seg.end - seg.start
    tokens: list[_Token] = []
    position = 0
    for word in words:
        start = seg.start + span * position / total
        position += len(word) + 1
        tokens.append(_Token(word, start, start))
    # Each token ends where the next begins so time is split by character length, spaces included.
    for token, nxt in zip(tokens, tokens[1:]):
        token.end = nxt.start
    tokens[-1].end = seg.end
    return tokens


def _has_word_timing(seg: Segment) -> bool:
    return any(_clean(w.word) for w in seg.words or ())


@lru_cache(maxsize=4096)
def _wrap_cached(words: tuple[str, ...], max_chars: int, max_lines: int) -> tuple[str, ...] | None:
    joined = " ".join(words)
    if len(joined) <= max_chars or len(words) == 1:
        return (joined,)
    n = len(words)
    inf = float("inf")
    for lines in range(2, min(max_lines, n) + 1):
        ideal = len(joined) / lines
        best = [[inf] * (n + 1) for _ in range(lines + 1)]
        cut = [[0] * (n + 1) for _ in range(lines + 1)]
        best[0][0] = 0.0
        for j in range(1, lines + 1):
            for i in range(j, n + 1):
                for k in range(j - 1, i):
                    if best[j - 1][k] == inf or (k > 0 and words[k][0] in _NO_LINE_START):
                        continue
                    chunk = words[k:i]
                    length = len(" ".join(chunk))
                    if length > max_chars and len(chunk) > 1:
                        continue
                    cost = (length - ideal) ** 2
                    if chunk[-1].rstrip(_TRAILING_QUOTES).endswith(_SENTENCE_END + _CLAUSE_END):
                        cost -= 4
                    elif _is_connector(chunk[-1]):
                        cost += 4
                    if j == 1:
                        cost += 0.01 * length  # ties favor a shorter top line (bottom-heavy)
                    total = best[j - 1][k] + cost
                    if total < best[j][i]:
                        best[j][i] = total
                        cut[j][i] = k
        if best[lines][n] != inf:
            out: list[str] = []
            i = n
            for j in range(lines, 0, -1):
                k = cut[j][i]
                out.append(" ".join(words[k:i]))
                i = k
            return tuple(reversed(out))
    return None


def _wrap(tokens: list[_Token], max_chars: int, max_lines: int) -> list[str] | None:
    result = _wrap_cached(tuple(t.text for t in tokens), max_chars, max_lines)
    return list(result) if result is not None else None


def _group(tokens: list[_Token], max_chars: int, max_lines: int, max_duration: float) -> list[list[_Token]]:
    capacity = max_chars * max_lines

    def fill(buf: list[_Token]) -> float:
        return (sum(len(t.text) for t in buf) + len(buf) - 1) / capacity

    def fits(buf: list[_Token]) -> bool:
        if len(buf) > 1 and buf[-1].end - buf[0].start > max_duration:
            return False
        return _wrap(buf, max_chars, max_lines) is not None

    def best_cut(buf: list[_Token]) -> int:
        best_score, best_index = 0, len(buf)
        for i in range(1, len(buf)):
            if fill(buf[:i]) < BACKTRACK_MIN_FILL:
                continue
            head_end = buf[i - 1].text
            if _ends_sentence(head_end):
                score = 3
            elif _ends_clause(head_end):
                score = 2
            elif _is_connector(buf[i].text):
                score = 1
            else:
                continue
            if score >= best_score:
                best_score, best_index = score, i
        return best_index

    groups: list[list[_Token]] = []
    buf: list[_Token] = []
    for token in tokens:
        gap = token.start - buf[-1].end if buf else 0.0
        if buf and (gap > PAUSE_BREAK_S or (gap >= SENTENCE_PAUSE_S and _ends_sentence(buf[-1].text))):
            groups.append(buf)
            buf = []
        if buf and _is_connector(token.text) and fill(buf) >= CONNECTOR_MIN_FILL:
            groups.append(buf)
            buf = []
        while buf and not fits(buf + [token]):
            cut = best_cut(buf)
            groups.append(buf[:cut])
            buf = buf[cut:]
        buf.append(token)
        if (_ends_sentence(token.text) and fill(buf) >= SENTENCE_MIN_FILL) or (
            _ends_clause(token.text) and fill(buf) >= CLAUSE_MIN_FILL
        ):
            groups.append(buf)
            buf = []
    if buf:
        groups.append(buf)
    return groups


def _fix_timing(cues: list[Cue], min_duration: float, min_gap: float) -> None:
    for cue in cues:
        if cue.end <= cue.start:
            cue.end = cue.start + MIN_POSITIVE_S
    for prev, cur in zip(cues, cues[1:]):
        earliest = prev.end + min_gap
        if cur.start >= earliest:
            continue
        shortened = cur.start - min_gap
        if shortened - prev.start >= MIN_SHRINK_S:
            prev.end = shortened
        else:
            cur.start = earliest
            cur.end = max(cur.end, cur.start + MIN_POSITIVE_S)
    for index, cue in enumerate(cues):
        limit = cues[index + 1].start - min_gap if index + 1 < len(cues) else float("inf")
        target = cue.start + min_duration
        if cue.end < target:
            cue.end = max(cue.end, min(target, limit))


def build_cues(
    segments: list[Segment],
    *,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    max_duration: float = 7.0,
    min_duration: float = 1.0,
    min_gap: float = 0.08,
) -> list[Cue]:
    cues: list[Cue] = []

    def emit(tokens: list[_Token]) -> None:
        for group in _group(tokens, max_chars_per_line, max_lines, max_duration):
            lines = _wrap(group, max_chars_per_line, max_lines)
            if lines:
                cues.append(Cue(group[0].start, group[-1].end, lines))

    stream: list[_Token] = []
    for seg in segments:
        if _has_word_timing(seg):
            _word_tokens(seg, stream)
            continue
        emit(stream)
        stream = []
        emit(_proportional_tokens(seg))
    emit(stream)
    _fix_timing(cues, min_duration, min_gap)
    return cues
