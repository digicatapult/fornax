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