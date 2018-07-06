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
        self.rows = [
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

    def test_opt(self):
        scores, optimum_match = opt.optimise(self.h, self.alpha, self.rows)
        self.assertSequenceEqual(
            list(itertools.chain(*optimum_match)), 
            list(itertools.chain(
                (1, 8, 0), 
                (2, 9, 0), 
                (3, 6, 0), 
                (4, 10, 0), 
                (5, 11, 0)
            ))
        )
      

if __name__ == '__main__':
    unittest.main()
