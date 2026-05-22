import pytest

from fixed_noise_diffusion.checkpoints import (
    parse_int_list,
    parse_nonnegative_int_list,
    parse_positive_int_list,
)


def test_parse_int_list_accepts_comma_separated_values():
    assert parse_int_list("1, 5,10", name="--epochs") == [1, 5, 10]


def test_parse_int_list_ignores_empty_items_between_commas():
    assert parse_int_list("1,, 2,", name="--epochs") == [1, 2]


@pytest.mark.parametrize("raw", ["", " , "])
def test_parse_int_list_rejects_empty_lists(raw):
    with pytest.raises(ValueError, match="--epochs"):
        parse_int_list(raw, name="--epochs")


@pytest.mark.parametrize("raw", ["1.5", "abc", "1,abc"])
def test_parse_int_list_rejects_non_integer_items(raw):
    with pytest.raises(ValueError, match="integer values"):
        parse_int_list(raw, name="--epochs")


@pytest.mark.parametrize("raw", ["0", "-1", "1,0"])
def test_parse_positive_int_list_rejects_nonpositive_values(raw):
    with pytest.raises(ValueError, match="at least 1"):
        parse_positive_int_list(raw, name="--epochs")


def test_parse_positive_int_list_accepts_positive_values():
    assert parse_positive_int_list("1,2", name="--epochs") == [1, 2]


def test_parse_nonnegative_int_list_accepts_zero_timestep():
    assert parse_nonnegative_int_list("0,25", name="--timesteps") == [0, 25]


def test_parse_nonnegative_int_list_rejects_negative_timestep():
    with pytest.raises(ValueError, match="at least 0"):
        parse_nonnegative_int_list("-1", name="--timesteps")
