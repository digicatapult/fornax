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
            [1, 1, 1, 0], [1, 4, 1, 0], [1, 8, 1, 0], 
            [2, 2, 2, 0], [2, 5, 2, 0], [2, 9, 2, 0], 
            [3, 3, 3, 0], [3, 6, 3, 0], [3, 12, 3, 0], 
            [3, 13, 3, 0], [4, 7, 4, 0], [4, 10, 4, 0], 
            [5, 11, 5, 0], [1, 1, 2, 1], [1, 1, 3, 1], 
            [1, 4, 2, 1], [1, 4, 3, 1], [1, 8, 2, 1], 
            [1, 8, 3, 1], [2, 2, 1, 1], [2, 2, 4, 1], 
            [2, 5, 1, 1], [2, 5, 4, 1], [2, 9, 1, 1], 
            [2, 9, 4, 1], [3, 3, 1, 1], [3, 6, 1, 1], 
            [3, 12, 1, 1], [3, 13, 1, 1], [4, 7, 5, 1], 
            [4, 7, 2, 1], [4, 10, 5, 1], [4, 10, 2, 1], 
            [5, 11, 4, 1]
        ]


        self.rows_b = [
            [1, 1, 1, 0], [1, 4, 4, 0], [1, 8, 8, 0], 
            [2, 2, 2, 0], [2, 5, 5, 0], [2, 9, 9, 0], 
            [3, 3, 3, 0], [3, 6, 6, 0], [3, 12, 12, 0], 
            [3, 13, 13, 0], [4, 7, 7, 0], [4, 10, 10, 0], 
            [5, 11, 11, 0], [1, 1, 4, 1], [1, 1, 2, 1], 
            [1, 1, 3, 1], [1, 4, 6, 1], [1, 4, 5, 1], 
            [1, 4, 1, 1], [1, 8, 9, 1], [1, 8, 6, 1], 
            [1, 8, 12, 1], [2, 2, 1, 1], [2, 5, 4, 1], 
            [2, 5, 7, 1], [2, 9, 10, 1], [2, 9, 8, 1], 
            [3, 3, 7, 1], [3, 3, 1, 1], [3, 6, 8, 1], 
            [3, 6, 4, 1], [3, 12, 11, 1], [3, 12, 8, 1], 
            [3, 13, 11, 1], [4, 7, 5, 1], [4, 7, 3, 1], 
            [4, 7, 10, 1], [4, 10, 9, 1], [4, 10, 11, 1], 
            [4, 10, 7, 1], [5, 11, 10, 1], [5, 11, 12, 1], 
            [5, 11, 13, 1]
        ]



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
