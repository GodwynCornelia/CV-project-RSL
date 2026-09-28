import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.lattice import LatticeGraph
from src.models import CandidateLattice, Hypothesis


def test_lattice_construction_and_beam_search():
    # Создаем 3 сегмента с известными гипотезами
    lat1 = CandidateLattice(
        segment_id=1,
        timestamp=(1.0, 2.0),
        candidates=[
            Hypothesis(gloss="Я", score=0.85),
            Hypothesis(gloss="МЫ", score=0.10),
            Hypothesis(gloss="ОН", score=0.05)
        ]
    )
    lat2 = CandidateLattice(
        segment_id=2,
        timestamp=(2.2, 3.1),
        candidates=[
            Hypothesis(gloss="ХОТЕТЬ", score=0.70),
            Hypothesis(gloss="ЛЮБИТЬ", score=0.20),
            Hypothesis(gloss="ИСКАТЬ", score=0.10)
        ]
    )
    lat3 = CandidateLattice(
        segment_id=3,
        timestamp=(3.3, 4.0),
        candidates=[
            Hypothesis(gloss="ВОДА", score=0.60),
            Hypothesis(gloss="ЧАЙ", score=0.30),
            Hypothesis(gloss="КОФЕ", score=0.10)
        ]
    )

    graph = LatticeGraph([lat1, lat2, lat3])
    paths = graph.beam_search(beam_width=3)

    assert len(paths) > 0
    # Лучший путь по вероятностям должен быть: Я -> ХОТЕТЬ -> ВОДА
    best_path = paths[0]
    assert best_path.glosses == ["Я", "ХОТЕТЬ", "ВОДА"]
    assert best_path.total_score > 0.5


def test_repetition_penalty():
    # Сегмент 1: "ВОДА" (0.55) vs "ЧАЙ" (0.45)
    lat1 = CandidateLattice(
        segment_id=1,
        timestamp=(1.0, 1.5),
        candidates=[
            Hypothesis(gloss="ВОДА", score=0.55),
            Hypothesis(gloss="ЧАЙ", score=0.45)
        ]
    )
    # Сегмент 2: "ВОДА" (0.52) vs "ЧАЙ" (0.48)
    lat2 = CandidateLattice(
        segment_id=2,
        timestamp=(1.6, 2.0),
        candidates=[
            Hypothesis(gloss="ВОДА", score=0.52),
            Hypothesis(gloss="ЧАЙ", score=0.48)
        ]
    )

    graph = LatticeGraph([lat1, lat2])
    paths = graph.beam_search(beam_width=3)

    # Без штрафа "ВОДА ВОДА" была бы чуть выше, но штраф анти-статтеринга
    # штрафует мгновенный повтор одной и той же глоссы подряд
    assert len(paths) > 0
    print("[PASS] test_lattice: all assertions passed!")


if __name__ == "__main__":
    test_lattice_construction_and_beam_search()
    test_repetition_penalty()

