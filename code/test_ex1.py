from code import fitpoly
import numpy
import pytest
import os


@pytest.fixture
def exercise_results():
    return fitpoly.exercise_1('data/womens100.csv', 'figures/womens100-best-fit.pdf')


def test_ex1_w(exercise_results):
    target = [4.09241546e+01, -1.50718122e-02]
    w = exercise_results
    assert numpy.allclose(w, target)


def test_womens100_figure_exists():
    assert os.path.exists('figures/womens100-best-fit.pdf')
