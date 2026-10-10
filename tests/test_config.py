from src.config import RANDOM_STATE


def test_random_state_exists():
    assert RANDOM_STATE is not None


def test_random_state_is_integer():
    assert isinstance(RANDOM_STATE, int)


def test_random_state_value():
    assert RANDOM_STATE == 42