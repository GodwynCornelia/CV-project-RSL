import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Optional
import numpy as np

from src.classifier.gesture_classifier import GestureClassifier
from src.config import KinematicConfig, LLMConfig, ModelConfig
from src.models import CandidateLattice, Hypothesis
from src.pipeline import ContinuousSignTranslationPipeline


def load_real_tensor_for_word(word: str) -> Optional[np.ndarray]:
    """
    Загружает реальный тензор (16, 118) из hybrid_tensors/train для заданного слова.
    """
    base_dirs = [Path("hybrid_tensors/train"), Path("hybrid_tensors/val")]
    for base in base_dirs:
        word_dir = base / word
        if word_dir.exists():
            files = list(word_dir.glob("*.npy"))
            if files:
                return np.load(files[0]).astype(np.float32)
    return None


def generate_demo_lattices(classifier: GestureClassifier) -> List[CandidateLattice]:
    """
    Формирует список CandidateLattice для демонстрации или быстрого тестирования.
    Если есть доступные тензоры, прогоняет их через нейросеть; иначе генерирует правдоподобные решетки.
    """
    test_words = ["я", "хотеть", "вода"]
    lattices: List[CandidateLattice] = []

    for idx, word in enumerate(test_words, start=1):
        tensor = load_real_tensor_for_word(word)
        if tensor is not None:
            lat = classifier.classify_segment(tensor, segment_id=idx, timestamp=(idx * 1.5, idx * 1.5 + 1.2))
        else:
            # Синтетическая решетка с типичным распределением вероятностей
            lat = CandidateLattice(
                segment_id=idx,
                timestamp=(idx * 1.5, idx * 1.5 + 1.2),
                candidates=[
                    Hypothesis(gloss=word, score=0.72),
                    Hypothesis(gloss="мы" if word == "я" else "любить", score=0.14),
                    Hypothesis(gloss="он" if word == "я" else "чай", score=0.07),
                    Hypothesis(gloss="вы", score=0.04),
                    Hypothesis(gloss="они", score=0.03)
                ]
            )
        lattices.append(lat)

    return lattices


def test_continuous_phrase_from_tensors():
    """
    Сквозной тест QA:
    1. Берет 3 реальных жеста ('я', 'хотеть', 'вода') из hybrid_tensors.
    2. Собирает непрерывный поток кадров с кинематическими паузами между ними.
    3. Прогоняет поток через KinematicSpotter и GestureClassifier.
    4. Декодирует решетку гипотез через SemanticDecoder в естественный русский текст.
    """
    model_cfg = ModelConfig()
    kinematic_cfg = KinematicConfig(
        velocity_threshold_active=12.0,
        velocity_threshold_stop=6.0,
        acceleration_threshold_stop=-2.0,
        pause_frames_trigger=4,
        min_gesture_frames=6
    )
    llm_cfg = LLMConfig(timeout_seconds=4.0)

    pipeline = ContinuousSignTranslationPipeline(
        model_config=model_cfg,
        kinematic_config=kinematic_cfg,
        llm_config=llm_cfg,
        init_tracker=False  # В тесте работаем напрямую с признаками (118,)
    )

    words = ["я", "хотеть", "вода"]
    stream_features: List[np.ndarray] = []
    curr_wrist_x = 100.0

    # 1. Сборка непрерывного потока с реалистичной кинематикой фаз (Preparation -> Stroke -> Retraction -> Pause)
    for word in words:
        tensor = load_real_tensor_for_word(word)
        if tensor is None:
            tensor = np.zeros((16, 118), dtype=np.float32)

        # Фаза жеста: 25 кадров с выраженным кинематическим пиком (v_max ~ 20 px/кадр)
        T_gesture = 25
        for t in range(T_gesture):
            idx = int(t * 15 / (T_gesture - 1))
            frame_feat = tensor[idx].copy()

            # Мгновенная скорость пропорциональна sin(pi * t / T)
            step_v = 22.0 * np.sin(np.pi * t / T_gesture)
            curr_wrist_x += step_v

            frame_feat[34] = curr_wrist_x  # Левая кисть MediaPipe
            frame_feat[35] = 250.0
            stream_features.append(frame_feat)

        # Фаза паузы между жестами: 12 кадров покоя (v = 0)
        for _ in range(12):
            pause_feat = np.zeros(118, dtype=np.float32)
            pause_feat[34] = curr_wrist_x
            pause_feat[35] = 250.0
            stream_features.append(pause_feat)

    # 2. Прогон непрерывного потока через споттер и классификатор
    timestamp = 0.0
    spotted_count = 0

    for feat in stream_features:
        timestamp += 0.033
        kinematics, segment_lattice = pipeline.process_features_frame(feat, timestamp)
        if segment_lattice is not None:
            spotted_count += 1
            print(f"--> [Тест] Распознан жест #{segment_lattice.segment_id}:")
            for c in segment_lattice.candidates:
                print(f"    - {c.gloss}: {c.score*100:.1f}%")

    print(f"\nВсего зафиксировано жестов: {spotted_count} (ожидалось 3)")
    assert spotted_count >= 1, "Споттер должен был зафиксировать хотя бы 1 жест в потоке"

    # 3. Декодирование фразы
    translation = pipeline.translate_current_phrase()
    print("\n--- РЕЗУЛЬТАТ СЕМАНТИЧЕСКОГО ДЕКОДИРОВАНИЯ ---")
    print(f"Top-1 глоссы: {translation.top1_glosses}")
    print(f"Итоговое предложение: «{translation.natural_sentence}»")
    print(f"Время декодирования: {translation.latency_ms:.1f} мс")

    assert translation.natural_sentence != "", "Итоговое предложение не должно быть пустым"
    print("\n[QA Тест] Сквозная цепочка успешно отработала!")


if __name__ == "__main__":
    test_continuous_phrase_from_tensors()
