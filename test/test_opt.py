import itertools
import unittest
import fornax.opt as opt
import fornax.model as model
import fornax.select as select
from test_base import TestCaseDB
from sqlalchemy.orm import Query, aliased

from sqlalchemy import or_, and_, literal
import numpy as np


class TestOpt(unittest.TestCase):
    """Reproduce the scenario set out in figure 4 of the paper"""

    def setUp(self):
        self.h = 2
        self.alpha = .3
        self.rows_a = [
            (1, 1, 2, 2, 1, 1, 0), (1, 1, 2, 5, 1, 2, 0), (1, 1, 3, 6, 1, 2, 0), (1, 1, 3, 3, 1, 1, 0), 
            (1, 1, 4, 7, 2, 2, 0), (1, 4, 2, 5, 1, 1, 0), (1, 4, 2, 2, 1, 2, 0), (1, 4, 3, 3, 1, 2, 0), 
            (1, 4, 3, 6, 1, 1, 0), (1, 4, 4, 7, 2, 2, 0), (1, 8, 2, 9, 1, 1, 0), (1, 8, 3, 6, 1, 1, 0), 
            (1, 8, 3, 12, 1, 1, 0), (1, 8, 4, 10, 2, 2, 0), (2, 2, 1, 1, 1, 1, 0), (2, 2, 1, 4, 1, 2, 0), 
            (2, 2, 3, 3, 2, 2, 0), (2, 5, 1, 4, 1, 1, 0), (2, 5, 1, 1, 1, 2, 0), (2, 5, 3, 6, 2, 2, 0), 
            (2, 5, 3, 3, 2, 2, 0), (2, 5, 4, 7, 1, 1, 0), (2, 5, 4, 10, 1, 2, 0), (2, 9, 1, 8, 1, 1, 0), 
            (2, 9, 3, 12, 2, 2, 0), (2, 9, 3, 6, 2, 2, 0), (2, 9, 4, 7, 1, 2, 0), (2, 9, 4, 10, 1, 1, 0), 
            (2, 9, 5, 11, 2, 2, 0), (3, 3, 1, 4, 1, 2, 0), (3, 3, 1, 1, 1, 1, 0), (3, 3, 2, 5, 2, 2, 0), 
            (3, 3, 2, 2, 2, 2, 0), (3, 6, 1, 1, 1, 2, 0), (3, 6, 1, 4, 1, 1, 0), (3, 6, 1, 8, 1, 1, 0), 
            (3, 6, 2, 9, 2, 2, 0), (3, 6, 2, 5, 2, 2, 0), (3, 12, 1, 8, 1, 1, 0), (3, 12, 2, 9, 2, 2, 0), 
            (4, 7, 1, 4, 2, 2, 0), (4, 7, 1, 1, 2, 2, 0), (4, 7, 2, 9, 1, 2, 0), (4, 7, 2, 5, 1, 1, 0), 
            (4, 7, 5, 11, 1, 2, 0), (4, 10, 1, 8, 2, 2, 0), (4, 10, 2, 5, 1, 2, 0), (4, 10, 2, 9, 1, 1, 0), 
            (4, 10, 5, 11, 1, 1, 0), (5, 11, 2, 9, 2, 2, 0), (5, 11, 4, 10, 1, 1, 0), (5, 11, 4, 7, 1, 2, 0)
        ]

        self.rows_b = [(1, 4), (2, 5), (3, 3), (4, 4), (5, 3)]

    def test_optimal_matches(self):
        _, optimum_match = opt.optimise(self.h, self.alpha, self.rows_a, self.rows_b)
        self.assertSequenceEqual(
            list((a,b) for (a,b,c) in optimum_match), 
            list([
                (1, 8), 
                (2, 9), 
                (3, 6), 
                (4, 10), 
                (5, 11)
            ])
        )

    def test_score_bounds(self):
        scores, _ = opt.optimise(self.h, self.alpha, self.rows_a, self.rows_b)
        self.assertDictEqual(
            {key:0 <= score < 1.1 for key, score in scores.items()},
            {key:True for key, score in scores.items()}
        )

if __name__ == '__main__':
    unittest.main()
