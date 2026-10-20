import os
import tempfile
import pytest

from rsl_trainer.database.db_manager import DatabaseManager, RUSSIAN_ALPHABET


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    yield db
    db.close()
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def test_initial_db_state(temp_db):
    stats = temp_db.get_user_stats()
    assert stats["user_id"] == 1
    assert stats["total_xp"] == 0
    assert stats["streak_days"] >= 1

    letters = temp_db.get_dactyl_letters()
    assert len(letters) == 33

    # Первые 3 буквы открыты ('learning'), остальные 'locked'
    assert letters[0]["status"] == "learning"
    assert letters[1]["status"] == "learning"
    assert letters[2]["status"] == "learning"
    assert letters[3]["status"] == "locked"


def test_add_xp_and_streak(temp_db):
    new_xp = temp_db.add_xp(25)
    assert new_xp == 25

    new_xp = temp_db.add_xp(50)
    assert new_xp == 75

    streak = temp_db.update_streak()
    assert streak >= 1


def test_update_dactyl_progress(temp_db):
    # Попытка сдать букву 'А' с баллом 0.85 (порог 0.70)
    res = temp_db.update_dactyl_progress("А", 0.85)
    assert res is True

    letters = {row["letter"]: row for row in temp_db.get_dactyl_letters()}
    assert letters["А"]["status"] == "mastered"
    assert letters["А"]["best_score"] == 0.85
    assert letters["А"]["attempts"] == 1

    # Следующая буква 'Б' уже была learning, а если сдать 'В', то 'Г' должна разблокироваться
    temp_db.update_dactyl_progress("Б", 0.90)
    temp_db.update_dactyl_progress("В", 0.75)

    letters = {row["letter"]: row for row in temp_db.get_dactyl_letters()}
    assert letters["В"]["status"] == "mastered"
    # Буква 'Г' (индекс 3) должна перейти из 'locked' в 'learning'
    assert letters["Г"]["status"] == "learning"


def test_levels_and_completion(temp_db):
    levels = temp_db.get_levels()
    assert len(levels) >= 4
    assert levels[0]["level_id"] == 1
    assert levels[0]["is_unlocked"] == 1
    assert levels[1]["is_unlocked"] == 0

    # Прохождение уровня 1
    temp_db.complete_level(1, stars=3)

    levels_after = temp_db.get_levels()
    assert levels_after[0]["stars"] == 3
    # Уровень 2 должен быть разблокирован
    assert levels_after[1]["is_unlocked"] == 1


@pytest.mark.parametrize("days_ago,expected", [(1, 5), (3, 1), (0, 4)])
def test_reward_does_not_erase_previous_activity_date(temp_db, days_ago, expected):
    import datetime
    previous = (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()
    with temp_db._get_connection() as conn:
        conn.execute("UPDATE users SET last_active_date = ?, streak_days = 4", (previous,))
    temp_db.add_xp(25)
    assert temp_db.update_streak() == expected
    # Multiple successful exercises on one day must not increment it again.
    temp_db.add_xp(30)
    assert temp_db.update_streak() == expected


def test_mastery_uses_verification_result_and_survives_later_failure(temp_db):
    temp_db.update_gesture_progress("вода", 0.31, is_passed=True)
    assert temp_db.get_gesture_progress("вода")["status"] == "mastered"
    temp_db.update_gesture_progress("вода", 0.10, is_passed=False)
    progress = temp_db.get_gesture_progress("вода")
    assert progress["status"] == "mastered"
    assert progress["best_score"] == 0.31
    temp_db.update_gesture_progress("дом", 0.65, is_passed=False)
    assert temp_db.get_gesture_progress("дом")["status"] == "learning"
