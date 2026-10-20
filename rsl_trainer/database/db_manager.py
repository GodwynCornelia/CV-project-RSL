import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


RUSSIAN_ALPHABET = [
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
    'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф',
    'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'
]

DEFAULT_LEVELS = [
    {
        "level_id": 1,
        "title": "Уровень 1: Знакомство и вежливость",
        "description": "Базовые фразы вежливости и приветствия",
        "target_phrase": ["Привет!", "Добро пожаловать!"],
        "is_unlocked": 1,
        "stars": 0
    },
    {
        "level_id": 2,
        "title": "Уровень 2: Базовые потребности",
        "description": "Составление фразы «Я хочу воду»",
        "target_phrase": ["я", "хотеть", "вода"],
        "is_unlocked": 0,
        "stars": 0
    },
    {
        "level_id": 3,
        "title": "Уровень 3: Дружба и дом",
        "description": "Составление фразы «Мой друг любит дом»",
        "target_phrase": ["друг", "любить", "дом"],
        "is_unlocked": 0,
        "stars": 0
    },
    {
        "level_id": 4,
        "title": "Уровень 4: Время и планы",
        "description": "Динамические фразы о времени",
        "target_phrase": ["время", "хорошо", "день"],
        "is_unlocked": 0,
        "stars": 0
    }
]


from contextlib import contextmanager


class DatabaseManager:
    """
    Менеджер локальной базы данных SQLite (userdata.db).
    Хранит профиль пользователя, опыт (XP), ударный режим (Streak),
    прогресс изучения дактиля и динамических жестов.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.db_path = str(base_dir / "userdata.db")
        else:
            self.db_path = db_path

        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        pass

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Таблица пользователя
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                streak_days INTEGER DEFAULT 1,
                total_xp INTEGER DEFAULT 0,
                last_active_date TEXT
            )
            """)

            # 2. Таблица прогресса по дактилю
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dactyl_progress (
                letter TEXT PRIMARY KEY,
                status TEXT DEFAULT 'locked',  -- locked, learning, mastered
                best_score REAL DEFAULT 0.0,
                attempts INTEGER DEFAULT 0
            )
            """)

            # 3. Таблица прогресса по динамическим жестам
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS gesture_progress (
                gesture_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'learning',
                best_score REAL DEFAULT 0.0,
                last_trained TEXT
            )
            """)

            # 4. Таблица уровней
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS levels (
                level_id INTEGER PRIMARY KEY,
                title TEXT,
                description TEXT,
                target_phrase TEXT,
                is_unlocked INTEGER DEFAULT 0,
                stars INTEGER DEFAULT 0
            )
            """)

            conn.commit()

            # Инициализация дефолтного пользователя
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                today = datetime.date.today().isoformat()
                cursor.execute(
                    "INSERT INTO users (user_id, streak_days, total_xp, last_active_date) VALUES (1, 1, 0, ?)",
                    (today,)
                )

            # Инициализация 33 букв алфавита
            cursor.execute("SELECT COUNT(*) FROM dactyl_progress")
            if cursor.fetchone()[0] == 0:
                for idx, letter in enumerate(RUSSIAN_ALPHABET):
                    # Первые 3 буквы открыты сразу, остальные locked
                    status = 'learning' if idx < 3 else 'locked'
                    cursor.execute(
                        "INSERT INTO dactyl_progress (letter, status, best_score, attempts) VALUES (?, ?, 0.0, 0)",
                        (letter, status)
                    )

            # Инициализация уровней
            cursor.execute("SELECT COUNT(*) FROM levels")
            if cursor.fetchone()[0] == 0:
                for lvl in DEFAULT_LEVELS:
                    cursor.execute(
                        "INSERT INTO levels (level_id, title, description, target_phrase, is_unlocked, stars) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            lvl["level_id"],
                            lvl["title"],
                            lvl["description"],
                            json.dumps(lvl["target_phrase"], ensure_ascii=False),
                            lvl["is_unlocked"],
                            lvl["stars"]
                        )
                    )

            conn.commit()

    def get_user_stats(self) -> Dict[str, Any]:
        """Возвращает текущие показатели пользователя (XP, Streak)."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = 1").fetchone()
            if row:
                return {
                    "user_id": row["user_id"],
                    "streak_days": row["streak_days"],
                    "total_xp": row["total_xp"],
                    "last_active_date": row["last_active_date"]
                }
            return {"user_id": 1, "streak_days": 1, "total_xp": 0, "last_active_date": ""}

    def add_xp(self, amount: int) -> int:
        """Добавляет очки опыта пользователю и обновляет активность."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET total_xp = total_xp + ? WHERE user_id = 1",
                (amount,)
            )
            conn.commit()
            return self.get_user_stats()["total_xp"]

    def update_streak(self) -> int:
        """Обновляет стрик дней активности."""
        today = datetime.date.today()
        today_str = today.isoformat()
        with self._get_connection() as conn:
            row = conn.execute("SELECT last_active_date, streak_days FROM users WHERE user_id = 1").fetchone()
            if row and row["last_active_date"]:
                last_date = datetime.date.fromisoformat(row["last_active_date"])
                diff = (today - last_date).days
                if diff == 1:
                    new_streak = row["streak_days"] + 1
                elif diff == 0:
                    new_streak = row["streak_days"]
                else:
                    new_streak = 1
            else:
                new_streak = 1

            conn.execute(
                "UPDATE users SET streak_days = ?, last_active_date = ? WHERE user_id = 1",
                (new_streak, today_str)
            )
            conn.commit()
            return new_streak

    def get_dactyl_letters(self) -> List[Dict[str, Any]]:
        """Возвращает список всех 33 букв с их текущим прогрессом."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM dactyl_progress").fetchall()
            return [dict(row) for row in rows]

    def update_dactyl_progress(self, letter: str, score: float) -> bool:
        """
        Обновляет прогресс сдачи буквы.
        Если score >= 0.70, буква помечается как 'mastered', и открывается следующая буква.
        """
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM dactyl_progress WHERE letter = ?", (letter,)).fetchone()
            if not row:
                return False

            best_score = max(row["best_score"], score)
            attempts = row["attempts"] + 1
            status = "mastered" if best_score >= 0.70 else ("learning" if row["status"] == "locked" else row["status"])

            conn.execute(
                "UPDATE dactyl_progress SET status = ?, best_score = ?, attempts = ? WHERE letter = ?",
                (status, best_score, attempts, letter)
            )

            # Если буква успешно сдана, разблокируем следующую по алфавиту
            if status == "mastered":
                try:
                    curr_idx = RUSSIAN_ALPHABET.index(letter)
                    if curr_idx + 1 < len(RUSSIAN_ALPHABET):
                        next_letter = RUSSIAN_ALPHABET[curr_idx + 1]
                        conn.execute(
                            "UPDATE dactyl_progress SET status = 'learning' WHERE letter = ? AND status = 'locked'",
                            (next_letter,)
                        )
                except ValueError:
                    pass

            conn.commit()
            return True

    def get_gesture_progress(self, gesture_name: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM gesture_progress WHERE gesture_id = ?", (gesture_name,)).fetchone()
            if row:
                return dict(row)
            return {"gesture_id": gesture_name, "status": "learning", "best_score": 0.0, "last_trained": ""}

    def update_gesture_progress(self, gesture_name: str, score: float, *, is_passed: bool = False) -> None:
        today = datetime.date.today().isoformat()
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM gesture_progress WHERE gesture_id = ?", (gesture_name,)).fetchone()
            if row:
                best = max(row["best_score"], score)
                status = "mastered" if is_passed or row["status"] == "mastered" else "learning"
                conn.execute(
                    "UPDATE gesture_progress SET status = ?, best_score = ?, last_trained = ? WHERE gesture_id = ?",
                    (status, best, today, gesture_name)
                )
            else:
                status = "mastered" if is_passed else "learning"
                conn.execute(
                    "INSERT INTO gesture_progress (gesture_id, status, best_score, last_trained) VALUES (?, ?, ?, ?)",
                    (gesture_name, status, score, today)
                )
            conn.commit()

    def get_levels(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM levels ORDER BY level_id ASC").fetchall()
            levels = []
            for row in rows:
                item = dict(row)
                item["target_phrase"] = json.loads(item["target_phrase"])
                levels.append(item)
            return levels

    def complete_level(self, level_id: int, stars: int = 3) -> bool:
        with self._get_connection() as conn:
            conn.execute("UPDATE levels SET stars = MAX(stars, ?) WHERE level_id = ?", (stars, level_id))
            # Открываем следующий уровень
            conn.execute("UPDATE levels SET is_unlocked = 1 WHERE level_id = ?", (level_id + 1,))
            conn.commit()
            return True
