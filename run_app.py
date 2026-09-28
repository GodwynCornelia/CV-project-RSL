import sys
from PyQt6.QtWidgets import QApplication

from src.config import KinematicConfig, LLMConfig, ModelConfig, UIConfig
from src.pipeline import ContinuousSignTranslationPipeline
from src.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RSL Machine Translation (Patent Demo)")

    model_config = ModelConfig()
    kinematic_config = KinematicConfig()
    llm_config = LLMConfig()
    ui_config = UIConfig()

    # Инициализация сквозного пайплайна
    pipeline = ContinuousSignTranslationPipeline(
        model_config=model_config,
        kinematic_config=kinematic_config,
        llm_config=llm_config,
        init_tracker=True
    )

    window = MainWindow(pipeline, config=ui_config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
