import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decoding.semantic_decoder import SemanticDecoder
from src.models import CandidateLattice, Hypothesis


def test_clean_llm_output():
    decoder = SemanticDecoder()

    raw1 = '  "Я хочу выпить чашку чая" '
    assert decoder._clean_llm_output(raw1) == "Я хочу выпить чашку чая."

    raw2 = 'Предложение: Мы идем домой.'
    assert decoder._clean_llm_output(raw2) == "Мы идем домой."

    raw3 = 'добро пожаловать'
    assert decoder._clean_llm_output(raw3) == "Добро пожаловать."


def test_linguistic_fallback():
    decoder = SemanticDecoder()
    glosses = ["я", "хотеть", "вода"]
    sentence = decoder._linguistic_fallback(glosses)
    assert sentence == "Я хотеть вода."


def test_empty_lattice_handling():
    decoder = SemanticDecoder()
    result = decoder.decode_lattice([])
    assert result.segments_count == 0
    assert result.natural_sentence == ""
    print("[PASS] test_semantic_decoder: all assertions passed!")


if __name__ == "__main__":
    test_clean_llm_output()
    test_linguistic_fallback()
    test_empty_lattice_handling()

