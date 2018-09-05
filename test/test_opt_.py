import unittest
import fornax.opt_
import numpy as np


class Dummy(fornax.opt_.Frame):

    names = 'a b c'.split()

    def __init__(self, records):
        super().__init__(records, self.names)

    @property
    def a(self):
        return self.records.a

    @property
    def b(self):
        return self.records.b


class TestFrame(unittest.TestCase):

    def setUp(self):
        self.names = 'a b'
        self.records = [[1, 2, 3], [2, 4, 6]]
        self.dummy = Dummy(self.records)

    def test_get_item(self):
        self.assertEqual(self.dummy[0].a, self.records[0][0])
        self.assertEqual(self.dummy[0].b, self.records[0][1])
        self.assertEqual(self.dummy[1].a, self.records[1][0])

    def test_len(self):
        self.assertEqual(len(self.dummy), 2)

    def test_slice(self):
        self.assertTrue(isinstance(self.dummy[:1], Dummy))
        

class TestNeighbourHoodMatchingCosts(unittest.TestCase):

    def setUp(self):
        self.records = [[1, 2, 3, 4, 5], [2, 4, 6, 7, 8]]
        self.costs = fornax.opt_.NeighbourHoodMatchingCosts(self.records)

    def test_get_u(self):
        self.assertListEqual(list(self.costs.u), [1, 2])
        self.assertEqual(self.costs[0].u, 1)