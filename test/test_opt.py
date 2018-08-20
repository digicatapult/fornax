import itertools
import unittest
import fornax.opt as opt
import fornax.model as model
import fornax.select as select
from test_base import TestCaseDB
from sqlalchemy.orm import Query, aliased

from sqlalchemy import or_, and_, literal
import numpy as np


class TestProximity(unittest.TestCase):

    def setUp(self):
        self.h = 2
        self.alpha = .3

    def test_zero(self):
        proximities = opt._proximity(self.h, self.alpha, np.array([0]))
        self.assertListEqual([1.0], proximities.tolist())

    def test_one(self):
        proximities = opt._proximity(self.h, self.alpha, np.array([1]))
        self.assertListEqual([self.alpha], proximities.tolist())

    def test_pow(self):
        proximities = opt._proximity(self.h, self.alpha, np.array([2, 3, 4]))
        self.assertListEqual([self.alpha**2, 0.0, 0.0], proximities.tolist())

    def test_assert_h(self):
        self.assertRaises(ValueError, opt._proximity, -1, 0.3, np.array([0]))

    def test_assert_alpha_big(self):
        self.assertRaises(ValueError, opt._proximity, 2, 1.1, np.array([0]))

    def test_assert_alpha_small(self):
        self.assertRaises(ValueError, opt._proximity, 2, -.1, np.array([0]))


class TestDeltaPlus(unittest.TestCase):

    def test_greater(self):
        self.assertListEqual(
            opt._delta_plus(np.array([2, 4, 6]), np.array([1, 2, 3])).tolist(),
            [1, 2, 3]
        )
        
    def test_less(self):
        self.assertListEqual(
            opt._delta_plus(np.array([1, 2, 3]), np.array([2, 4, 6])).tolist(),
            [0, 0, 0]
        )

class TestOpt(unittest.TestCase):
    """Reproduce the scenario set out in figure 4 of the paper"""

    def setUp(self):
        self.h = 2
        self.alpha = .3
        self.records = [
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 1), (1, 1, 1, 4, 0, 1, 0, 0, 0, 1), (1, 1, 2, 5, 1, 2, 0, 0, 0, 1), 
            (1, 1, 3, 3, 1, 1, 0, 0, 0, 1), (1, 1, 3, 6, 1, 2, 0, 0, 0, 1), (1, 1, 4, 7, 2, 2, 0, 0, 0, 1), 
            (1, 4, 1, 1, 0, 1, 0, 0, 0, 1), (1, 4, 1, 4, 0, 0, 0, 0, 0, 1), (1, 4, 1, 8, 0, 2, 0, 0, 0, 1), 
            (1, 4, 2, 5, 1, 1, 0, 0, 0, 1), (1, 4, 3, 3, 1, 2, 0, 0, 0, 1), (1, 4, 3, 6, 1, 1, 0, 0, 0, 1), 
            (1, 4, 4, 7, 2, 2, 0, 0, 0, 1), (1, 8, 1, 4, 0, 2, 0, 0, 0, 1), (1, 8, 1, 8, 0, 0, 0, 0, 0, 1), 
            (1, 8, 2, 9, 1, 1, 0, 0, 0, 1), (1, 8, 3, 6, 1, 1, 0, 0, 0, 1), (1, 8, 3, 12, 1, 1, 0, 0, 0, 1), 
            (1, 8, 4, 10, 2, 2, 0, 0, 0, 1), (2, 5, 1, 1, 1, 2, 0, 0, 0, 1), (2, 5, 1, 4, 1, 1, 0, 0, 0, 1), 
            (2, 5, 2, 5, 0, 0, 0, 0, 0, 1), (2, 5, 3, 3, 2, 2, 0, 0, 0, 1), (2, 5, 3, 6, 2, 2, 0, 0, 0, 1), 
            (2, 5, 4, 7, 1, 1, 0, 0, 0, 1), (2, 5, 4, 10, 1, 2, 0, 0, 0, 1), (2, 5, 5, None, 2, None, 0, 0, 0, 1), 
            (2, 9, 1, 8, 1, 1, 0, 0, 0, 1), (2, 9, 2, 9, 0, 0, 0, 0, 0, 1), (2, 9, 3, 6, 2, 2, 0, 0, 0, 1), 
            (2, 9, 3, 12, 2, 2, 0, 0, 0, 1), (2, 9, 4, 7, 1, 2, 0, 0, 0, 1), (2, 9, 4, 10, 1, 1, 0, 0, 0, 1), 
            (2, 9, 5, 11, 2, 2, 0, 0, 0, 1), (3, 3, 1, 1, 1, 1, 0, 0, 0, 1), (3, 3, 1, 4, 1, 2, 0, 0, 0, 1), 
            (3, 3, 2, 5, 2, 2, 0, 0, 0, 1), (3, 3, 3, 3, 0, 0, 0, 0, 0, 1), (3, 6, 1, 1, 1, 2, 0, 0, 0, 1), 
            (3, 6, 1, 4, 1, 1, 0, 0, 0, 1), (3, 6, 1, 8, 1, 1, 0, 0, 0, 1), (3, 6, 2, 5, 2, 2, 0, 0, 0, 1), 
            (3, 6, 2, 9, 2, 2, 0, 0, 0, 1), (3, 6, 3, 6, 0, 0, 0, 0, 0, 1), (3, 6, 3, 12, 0, 2, 0, 0, 0, 1), 
            (3, 12, 1, 8, 1, 1, 0, 0, 0, 1), (3, 12, 2, 9, 2, 2, 0, 0, 0, 1), (3, 12, 3, 6, 0, 2, 0, 0, 0, 1), 
            (3, 12, 3, 12, 0, 0, 0, 0, 0, 1), (3, 12, 3, 13, 0, 2, 0, 0, 0, 1), (3, 13, 1, None, 1, None, 0, 0, 0, 1), 
            (3, 13, 2, None, 2, None, 0, 0, 0, 1), (3, 13, 3, 12, 0, 2, 0, 0, 0, 1), (3, 13, 3, 13, 0, 0, 0, 0, 0, 1), 
            (4, 7, 1, 1, 2, 2, 0, 0, 0, 1), (4, 7, 1, 4, 2, 2, 0, 0, 0, 1), (4, 7, 2, 5, 1, 1, 0, 0, 0, 1), 
            (4, 7, 2, 9, 1, 2, 0, 0, 0, 1), (4, 7, 4, 7, 0, 0, 0, 0, 0, 1), (4, 7, 4, 10, 0, 1, 0, 0, 0, 1), 
            (4, 7, 5, 11, 1, 2, 0, 0, 0, 1), (4, 10, 1, 8, 2, 2, 0, 0, 0, 1), (4, 10, 2, 5, 1, 2, 0, 0, 0, 1), 
            (4, 10, 2, 9, 1, 1, 0, 0, 0, 1), (4, 10, 4, 7, 0, 1, 0, 0, 0, 1), (4, 10, 4, 10, 0, 0, 0, 0, 0, 1), 
            (4, 10, 5, 11, 1, 1, 0, 0, 0, 1), (5, 11, 2, 9, 2, 2, 0, 0, 0, 1), (5, 11, 4, 7, 1, 2, 0, 0, 0, 1), 
            (5, 11, 4, 10, 1, 1, 0, 0, 0, 1), (5, 11, 5, 11, 0, 0, 0, 0, 0, 1)
        ]
        
    def test_optimal_matches(self):
        _, optimum_match = opt.optimise(5, self.h, self.alpha, self.records)
        self.assertSequenceEqual(
            list((a, b) for (a, b, c) in optimum_match),
            list([
                (1, 8),
                (2, 9),
                (3, 6),
                (4, 10),
                (5, 11)
            ])
        )

    def test_score_bounds(self):
        scores, _ = opt.optimise(5, self.h, self.alpha, self.records)
        self.assertDictEqual(
            {key: 0 <= score < 1.1 for key, score in scores.items()},
            {key: True for key, score in scores.items()}
        )


if __name__ == '__main__':
    unittest.main()
