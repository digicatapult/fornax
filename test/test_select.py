import unittest
import collections
import numpy as np
import fornax.model as model
from sqlalchemy.sql.expression import literal
from test_base import TestCaseDB

import fornax.select


class TestTable(unittest.TestCase):

    def setUp(self):
        self.labels = ['id', 'label', 'type']
        self.tuples = [
            (0, 'a', 0),
            (1, 'b', 0)
        ]
        self.table = fornax.select.Table(self.labels, self.tuples)

    def test_length(self):
        self.assertEqual(len(self.table), 2)

    def test_first_row(self):
        self.assertEqual(self.tuples[0], tuple(self.table[0]))

    def test_last_row(self):
        self.assertEqual(self.tuples[-1], tuple(self.table[-1]))

    def test_slice_front(self):
        self.assertEqual(self.tuples[0:], [tuple(row) for row in self.table[0:]])

    def test_slice_back(self):
        self.assertEqual(self.tuples[:-1], [tuple(row) for row in self.table[:-1]])

    def test_to_frame(self):
        self.assertEqual(
            self.table.to_frame(), 
            fornax.select.Frame(self.table.fields(), [(0, 1), ('a', 'b'), (0, 0)])
        )

    def test_fields(self):
        first = self.table[0]
        self.assertEqual(
            [item for item in first],
            [getattr(first, field) for field in self.table.fields()]
        )

    def test_join_product(self):
        table = self.table.join(lambda x: True, self.table)
        self.assertEqual(len(table), len(self.tuples)**2)

    def test_join_predicate(self):
        table = self.table.join(lambda x: x[0].id == x[1].id, self.table)
        self.assertEqual(len(table), len(self.tuples))

    def test_join_empty(self):
        table = self.table.join(lambda x: False, self.table)
        self.assertListEqual(
            list(table.fields()), 
            [l+'_left' for l in self.labels] +
            [r+'_right' for r in self.labels] 
        )


class TestFrame(unittest.TestCase):

    def setUp(self):
        self.labels = ['first', 'second']
        self.columns = [[1, 2, 3], [4, 5, 6]]
        self.frame = fornax.select.Frame(self.labels, self.columns)

    def test_length(self):
        self.assertEqual(len(self.frame), len(self.columns[0]))

    def test_get_item(self):
        self.assertListEqual(list(self.frame.first), self.columns[0])
        self.assertListEqual(list(self.frame.second), self.columns[1])

    def test_numpy(self):
        self.assertIsInstance(self.frame.first, np.ndarray)
        self.assertIsInstance(self.frame.second, np.ndarray)

    def test_assert_length(self):
        self.assertRaises(
            ValueError, 
            fornax.select.Frame, self.labels, 
            [self.columns[:-1], self.columns[1]]
        )


if __name__ == '__main__':
    unittest.main()