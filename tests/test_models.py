"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean, daily_max, daily_min

@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            [[0, 0],
             [0, 0],
             [0, 0]],
            [0, 0],
        ),
        (
            [[1, 2],
             [3, 4],
             [5, 6]],
            [3, 4],
        ),
    ],
)
def test_daily_mean(test_input, expected):
    """Test mean function for zeroes and positive integers."""
    npt.assert_array_equal(
        daily_mean(np.array(test_input)),
        np.array(expected),
    )

@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [0, 0, 0],
        ),
        (
            [[4, 2, 5],
             [1, 6, 2],
             [4, 1, 9]],
            [4, 6, 9],
        ),
        (
            [[4, -2, 5],
             [1, -6, 2],
             [-4, -1, 9]],
            [4, -1, 9],
        ),
    ],
)
def test_daily_max(test_input, expected):
    """Test max function for zeroes and positive and negative integers."""
    npt.assert_array_equal(
        daily_max(np.array(test_input)),
        np.array(expected),
    )

@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [0, 0, 0],
        ),
        (
            [[4, 2, 5],
             [1, 6, 2],
             [4, 1, 9]],
            [1, 1, 2],
        ),
        (
            [[4, -2, 5],
             [1, -6, 2],
             [-4, -1, 9]],
            [-4, -6, 2],
        ),
    ],
)
def test_daily_min(test_input, expected):
    """Test min function for zeroes and positive and negative integers."""
    npt.assert_array_equal(
        daily_min(np.array(test_input)),
        np.array(expected),
    )


def test_daily_min_string():
    """Test that daily_min raises TypeError for string input."""

    with pytest.raises(TypeError):
        daily_min([["Hello", "there"], ["General", "Kenobi"]])

