import unittest
import fornax.opt as opt
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


class TestNeighbourHoodMatchingCosts(unittest.TestCase):

    def setUp(self):
        self.records = [[1, 2, 3, 4, 5], [2, 4, 6, 7, 8]]
        self.costs = opt.NeighbourHoodMatchingCosts(self.records)

    def test_get_v(self):
        self.assertListEqual(list(self.costs.v), [1, 2])
        self.assertEqual(self.costs[0].v, 1)

    def test_get_u(self):
        self.assertListEqual(list(self.costs.u), [2, 4])
        self.assertEqual(self.costs[0].u, 2)

    def test_type(self):
        self.assertTrue(isinstance(self.costs[:1], opt.NeighbourHoodMatchingCosts))


class TestOpt(unittest.TestCase):
    """Reproduce the scenario set out in figure 4 of the paper"""

    def setUp(self):
        self.h = 2
        self.alpha = .3
        self.records = ([
            (1, 1, 1, 1, 0, 0, 1), (1, 1, 1, 4, 0, 1, 1), (1, 1, 3, 3, 1, 1, 1),  
            (1, 4, 1, 1, 0, 1, 1), (1, 4, 1, 4, 0, 0, 1), (1, 4, 2, 5, 1, 1, 1), 
            (1, 4, 3, 6, 1, 1, 1), (1, 8, 1, 8, 0, 0, 1), (1, 8, 2, 9, 1, 1, 1), 
            (1, 8, 3, 6, 1, 1, 1), (1, 8, 3, 12, 1, 1, 1), (2, 5, 1, 4, 1, 1, 1), 
            (2, 5, 2, 5, 0, 0, 1), (2, 5, 4, 7, 1, 1, 1), (2, 9, 1, 8, 1, 1, 1), 
            (2, 9, 2, 9, 0, 0, 1), (2, 9, 4, 10, 1, 1, 1), (3, 3, 1, 1, 1, 1, 1),
            (3, 3, 3, 3, 0, 0, 1), (3, 6, 1, 4, 1, 1, 1), (3, 6, 1, 8, 1, 1, 1),
            (3, 6, 3, 6, 0, 0, 1), (3, 12, 1, 8, 1, 1, 1), (3, 12, 3, 12, 0, 0, 1), 
            (3, 13, 1, None, 1, None, 1), (3, 13, 3, 13, 0, 0, 1), (4, 7, 2, 5, 1, 1, 1), 
            (4, 7, 4, 7, 0, 0, 1), (4, 7, 4, 10, 0, 1, 1),  (4, 10, 2, 9, 1, 1, 1), 
            (4, 10, 4, 7, 0, 1, 1), (4, 10, 4, 10, 0, 0, 1), (4, 10, 5, 11, 1, 1, 1),
            (5, 11, 4, 10, 1, 1, 1), (5, 11, 5, 11, 0, 0, 1)
        ])

    def test_optimal_matches(self):

        graphs, scores = opt.solve(self.records, hopping_distance=1)
        perfect = [graph for graph, score in zip(graphs, scores) if score == 0]

        self.assertSequenceEqual(
            perfect[0],
            [(1, 8), (2, 9), (3, 6), (4, 10), (5, 11)]
        )

        self.assertSequenceEqual(
            perfect[1],
            [(1, 8), (2, 9), (3, 12), (4, 10), (5, 11)]
        )