import unittest
import fornax.opt_
import numpy as np

class TestNeighbourHoodMatchingCosts(unittest.TestCase):

    def setUp(self):
        self.records = [[1, 2, 3, 4, 5], [2, 4, 6, 7, 8]]
        self.costs = fornax.opt_.NeighbourHoodMatchingCosts(self.records)

    def test_get_v(self):
        self.assertListEqual(list(self.costs.v), [1, 2])
        self.assertEqual(self.costs[0].v, 1)

    def test_get_u(self):
        self.assertListEqual(list(self.costs.u), [2, 4])
        self.assertEqual(self.costs[0].u, 2)

    def test_type(self):
        self.assertTrue(isinstance(self.costs[:1], fornax.opt_.NeighbourHoodMatchingCosts))


class TestOpt(unittest.TestCase):

    def setUp(self):
        self.records = ([
            (1, 1, 1, 1, 0, 0, 1), (1, 1, 1, 4, 0, 1, 1), (1, 1, 2, 5, 1, 2, 1), 
            (1, 1, 3, 3, 1, 1, 1), (1, 1, 3, 6, 1, 2, 1), (1, 1, 4, 7, 2, 2, 1), 
            (1, 4, 1, 1, 0, 1, 1), (1, 4, 1, 4, 0, 0, 1), (1, 4, 1, 8, 0, 2, 1), 
            (1, 4, 2, 5, 1, 1, 1), (1, 4, 3, 3, 1, 2, 1), (1, 4, 3, 6, 1, 1, 1), 
            (1, 4, 4, 7, 2, 2, 1), (1, 8, 1, 4, 0, 2, 1), (1, 8, 1, 8, 0, 0, 1), 
            (1, 8, 2, 9, 1, 1, 1), (1, 8, 3, 6, 1, 1, 1), (1, 8, 3, 12, 1, 1, 1), 
            (1, 8, 4, 10, 2, 2, 1), (2, 5, 1, 1, 1, 2, 1), (2, 5, 1, 4, 1, 1, 1), 
            (2, 5, 2, 5, 0, 0, 1), (2, 5, 3, 3, 2, 2, 1), (2, 5, 3, 6, 2, 2, 1), 
            (2, 5, 4, 7, 1, 1, 1), (2, 5, 4, 10, 1, 2, 1), (2, 5, 5, None, 2, None, 1), 
            (2, 9, 1, 8, 1, 1, 1), (2, 9, 2, 9, 0, 0, 1), (2, 9, 3, 6, 2, 2, 1), 
            (2, 9, 3, 12, 2, 2, 1), (2, 9, 4, 7, 1, 2, 1), (2, 9, 4, 10, 1, 1, 1), 
            (2, 9, 5, 11, 2, 2, 1), (3, 3, 1, 1, 1, 1, 1), (3, 3, 1, 4, 1, 2, 1), 
            (3, 3, 2, 5, 2, 2, 1), (3, 3, 3, 3, 0, 0, 1), (3, 6, 1, 1, 1, 2, 1), 
            (3, 6, 1, 4, 1, 1, 1), (3, 6, 1, 8, 1, 1, 1), (3, 6, 2, 5, 2, 2, 1), 
            (3, 6, 2, 9, 2, 2, 1), (3, 6, 3, 6, 0, 0, 1), (3, 6, 3, 12, 0, 2, 1), 
            (3, 12, 1, 8, 1, 1, 1), (3, 12, 2, 9, 2, 2, 1), (3, 12, 3, 6, 0, 2, 1), 
            (3, 12, 3, 12, 0, 0, 1), (3, 12, 3, 13, 0, 2, 1), (3, 13, 1, None, 1, None, 1), 
            (3, 13, 2, None, 2, None, 1), (3, 13, 3, 12, 0, 2, 1), (3, 13, 3, 13, 0, 0, 1), 
            (4, 7, 1, 1, 2, 2, 1), (4, 7, 1, 4, 2, 2, 1), (4, 7, 2, 5, 1, 1, 1), 
            (4, 7, 2, 9, 1, 2, 1), (4, 7, 4, 7, 0, 0, 1), (4, 7, 4, 10, 0, 1, 1), 
            (4, 7, 5, 11, 1, 2, 1), (4, 10, 1, 8, 2, 2, 1), (4, 10, 2, 5, 1, 2, 1), 
            (4, 10, 2, 9, 1, 1, 1), (4, 10, 4, 7, 0, 1, 1), (4, 10, 4, 10, 0, 0, 1), 
            (4, 10, 5, 11, 1, 1, 1), (5, 11, 2, 9, 2, 2, 1), (5, 11, 4, 7, 1, 2, 1), 
            (5, 11, 4, 10, 1, 1, 1), (5, 11, 5, 11, 0, 0, 1)
        ])
        self.query_result = fornax.opt_.QueryResult([tuple(item if item is not None else -1 for item in tup) for tup in self.records])


    def test_missed(self):
        _, misses = fornax.opt_.missed(self.query_result)
        self.assertEqual(misses[(3, 13)], 2)
        self.assertEqual(misses[(2, 5)], 1)

    def test_totals(self):
        totals, _ = fornax.opt_.missed(self.query_result)
        self.assertEqual(totals[(3, 13)], 1)
        self.assertEqual(totals[(2, 5)], 4)

    def test_get_matching_costs(self):
        matching_costs = fornax.opt_.get_matching_costs(self.query_result)
        self.assertFalse(True)

    def test_solve(self):
        fornax.opt_.solve(self.query_result)