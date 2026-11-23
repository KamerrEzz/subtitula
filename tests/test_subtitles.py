import pytest

from whisper_studio.models import Segment, Word
from whisper_studio.subtitles import Cue, build_cues


def words_segment(text: str, start: float = 0.0, per_word: float = 0.3, gap_after: dict[int, float] | None = None) -> Segment:
    gap_after = gap_after or {}
    t = start
    words: list[Word] = []
    for i, w in enumerate(text.split()):
        words.append(Word(t, t + per_word, " " + w, 0.9))
        t += per_word + gap_after.get(i, 0.0)
    return Segment(words[0].start, words[-1].end, " " + text, words)


def flat(cue: Cue) -> str:
    return " ".join(cue.lines)


def check_invariants(cues: list[Cue], min_gap: float = 0.08) -> None:
    for cue in cues:
        assert cue.end > cue.start
        assert 1 <= len(cue.lines) <= 2
        assert all(line and line == line.strip() for line in cue.lines)
    for prev, cur in zip(cues, cues[1:]):
        assert cur.start >= prev.end + min_gap - 1e-9


def test_short_segment_is_one_line_one_cue() -> None:
    cues = build_cues([words_segment("Hola mundo, esto es una prueba.")])
    assert len(cues) == 1
    assert cues[0].lines == ["Hola mundo, esto es una prueba."]


def test_line_length_and_line_count_limits() -> None:
    text = " ".join(["palabra"] * 60)
    cues = build_cues([words_segment(text, per_word=0.2)])
    assert len(cues) > 1
    check_invariants(cues)
    for cue in cues:
        assert all(len(line) <= 42 for line in cue.lines)
    assert " ".join(flat(c) for c in cues) == text


def test_two_line_cue_is_balanced() -> None:
    cues = build_cues([words_segment("uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece")])
    assert len(cues) == 1
    first, second = cues[0].lines
    assert abs(len(first) - len(second)) <= 8


def test_wrap_never_splits_words_or_starts_with_closing_punctuation() -> None:
    seg = words_segment("esta frase es bastante larga , y la coma suelta no debe abrir línea nueva ok")
    for cue in build_cues([seg]):
        for line in cue.lines:
            assert line[0] not in ".,;:!?…)"


def test_prefers_sentence_end_over_filling_the_cue() -> None:
    text = "Esta es la primera frase terminada bien. Segunda frase que sigue y que es bastante más larga todavía"
    cues = build_cues([words_segment(text, per_word=0.25)])
    assert flat(cues[0]).endswith("bien.")


def test_prefers_comma_when_it_must_break() -> None:
    text = "Cuando llegamos a la casa vieja del pueblo, empezamos a buscar todas las llaves perdidas ayer por la tarde"
    cues = build_cues([words_segment(text, per_word=0.2)])
    assert len(cues) >= 2
    assert flat(cues[0]).endswith("pueblo,")


def test_breaks_before_connector_when_cue_is_half_full() -> None:
    text = "tenemos muchas cosas importantes que hacer hoy mismo pero nadie quiere empezar con nada de nada"
    cues = build_cues([words_segment(text, per_word=0.25)])
    assert len(cues) >= 2
    assert flat(cues[1]).startswith("pero")


def test_breaks_on_long_pause_between_words() -> None:
    seg = words_segment("hola a todos bienvenidos de nuevo", gap_after={2: 1.5})
    cues = build_cues([seg])
    assert len(cues) == 2
    assert flat(cues[0]) == "hola a todos"
    assert flat(cues[1]) == "bienvenidos de nuevo"


def test_max_duration_forces_a_break() -> None:
    seg = words_segment("a b c d e f g h i j k l m n o p q r s t", per_word=0.9)
    cues = build_cues([seg], max_duration=7.0)
    assert len(cues) >= 3
    assert all(cue.end - cue.start <= 7.0 + 1e-9 for cue in cues)


def test_min_duration_extends_into_the_gap_without_overlap() -> None:
    first = Segment(0.0, 0.4, "Sí.", [Word(0.0, 0.4, " Sí.", 0.9)])
    second = Segment(1.2, 3.0, "Claro que sí.", [Word(1.2, 1.6, " Claro", 0.9), Word(1.6, 2.0, " que", 0.9), Word(2.0, 3.0, " sí.", 0.9)])
    cues = build_cues([first, second], min_duration=1.0, min_gap=0.08)
    check_invariants(cues)
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(1.0) or cues[0].end <= cues[1].start - 0.08 + 1e-9
    assert cues[0].end - cues[0].start >= 0.7


def test_min_duration_is_capped_by_the_next_cue() -> None:
    a = Segment(0.0, 0.3, "Hola.", [Word(0.0, 0.3, " Hola.", 0.9)])
    b = Segment(1.0, 3.0, "Adiós amigo mío.", [Word(1.0, 1.5, " Adiós", 0.9), Word(1.5, 2.0, " amigo", 0.9), Word(2.0, 3.0, " mío.", 0.9)])
    cues = build_cues([a, b], min_duration=1.0, min_gap=0.08)
    assert len(cues) == 2
    assert cues[0].end == pytest.approx(cues[1].start - 0.08)
    check_invariants(cues)


def test_min_gap_is_enforced_between_touching_cues() -> None:
    a = Segment(0.0, 2.0, "Primera frase completa.", [Word(0.0, 1.0, " Primera", 0.9), Word(1.0, 1.5, " frase", 0.9), Word(1.5, 2.0, " completa.", 0.9)])
    b = Segment(2.0, 4.0, "Segunda frase completa.", [Word(2.0, 3.0, " Segunda", 0.9), Word(3.0, 3.5, " frase", 0.9), Word(3.5, 4.0, " completa.", 0.9)])
    cues = build_cues([a, b], max_chars_per_line=25)
    assert len(cues) == 2
    assert cues[1].start - cues[0].end >= 0.08 - 1e-9
    check_invariants(cues)


def test_fallback_without_words_splits_text_and_distributes_time() -> None:
    text = "Esta frase no tiene marcas por palabra pero es larga. Y esta es otra frase bastante larga también."
    cues = build_cues([Segment(10.0, 20.0, text)])
    assert len(cues) >= 2
    check_invariants(cues)
    assert cues[0].start == pytest.approx(10.0)
    assert cues[-1].end <= 20.0 + 1e-9
    assert " ".join(flat(c) for c in cues) == text
    lengths = [len(flat(c)) for c in cues]
    durations = [c.end - c.start for c in cues]
    assert durations[lengths.index(max(lengths))] >= durations[lengths.index(min(lengths))]


def test_fallback_short_segment_keeps_one_cue_with_its_timing() -> None:
    cues = build_cues([Segment(1.0, 3.0, "  Hola   mundo ")])
    assert cues == [Cue(1.0, 3.0, ["Hola mundo"])]


def test_empty_and_whitespace_segments_are_dropped() -> None:
    cues = build_cues([Segment(0, 1, "   "), Segment(1, 2, ""), Segment(2, 3, "  ", [Word(2, 3, "  ", 0.5)]), Segment(3, 4, "Real")])
    assert [flat(c) for c in cues] == ["Real"]


def test_very_long_word_stays_intact_on_its_own_line() -> None:
    giant = "x" * 60
    cues = build_cues([words_segment(f"mira {giant} ahora")])
    lines = [line for cue in cues for line in cue.lines]
    assert giant in lines
    assert all(len(line) <= 42 for line in lines if line != giant)
    check_invariants(cues)


def test_punctuation_only_tokens_glue_to_the_previous_word() -> None:
    seg = Segment(0, 1.5, "Hola , mundo", [Word(0, 0.5, " Hola", 0.9), Word(0.5, 0.6, ",", 0.9), Word(0.6, 1.5, " mundo", 0.9)])
    assert build_cues([seg])[0].lines == ["Hola, mundo"]


def test_zero_duration_words_still_yield_positive_cues() -> None:
    seg = Segment(5.0, 5.0, "a b", [Word(5.0, 5.0, " a", 0.9), Word(5.0, 5.0, " b", 0.9)])
    cues = build_cues([seg])
    check_invariants(cues)


def test_text_is_preserved_across_a_long_transcript() -> None:
    sentences = [f"Frase número {i} con algo de contenido para leer, y más texto después." for i in range(40)]
    segs, t = [], 0.0
    for s in sentences:
        seg = words_segment(s, start=t, per_word=0.28)
        segs.append(seg)
        t = seg.end + 0.2
    cues = build_cues(segs)
    check_invariants(cues)
    assert " ".join(flat(c) for c in cues) == " ".join(sentences)


def test_short_sentence_followed_by_a_pause_gets_its_own_cue() -> None:
    seg = words_segment("Funciona bien. Ahora seguimos con otra frase distinta", gap_after={1: 0.4})
    cues = build_cues([seg])
    assert flat(cues[0]) == "Funciona bien."
    assert flat(cues[1]).startswith("Ahora")
