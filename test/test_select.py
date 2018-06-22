import unittest
import collections
import numpy as np
import fornax.model as model
from sqlalchemy.sql import func
from sqlalchemy.orm import Query
from sqlalchemy.sql.expression import literal
from test_base import TestCaseDB

import fornax.select

DummyNode = collections.namedtuple(
    'DummyNode', 'id, label, type, search_term'
)

class TestSelect(TestCaseDB):

    def setUp(self):
        """ populate the database with edge and node types """
        super().setUp()
        new_node_type = model.NodeType(id=0, description="A node type for testing")
        new_edge_type = model.EdgeType(id=0, description="An edge type for testing")
        self.session.add(new_node_type)
        self.session.add(new_edge_type)
        self.session.commit()

        # Insert some labelled nodes
        labels = ['Matt', 'Dom', 'Callum', 'David', 'Anthony']
        for id_, label in enumerate(labels):
            new_node = model.Node(id=id_, label=label, type=0)
            self.session.add(new_node)
            self.session.commit()

        self.session.add(model.Edge(start=0, end=1, type=0, weight=1))
        self.session.add(model.Edge(start=2, end=3, type=0, weight=1))


    def test_get_candidate(self):
        """ test get candidate """
        query = fornax.select.get_candidate(0.7, 'Mat')
        rows = query.with_session(self.session).all()
        self.assertEqual(
            sorted([row.label for row in rows]), 
            sorted(['Matt'])
        )

    def test_get_candidates_empty(self):
        """ test get_candidates with empty string """
        query = fornax.select.get_candidate(0.7, [])
        rows = query.with_session(self.session).all()
        self.assertEqual([row.label for row in rows], [])

    def test_get_candidates_negative_distance(self):
        """ test get candidate with negative distance"""
        self.assertRaises(ValueError, fornax.select.get_candidate, -1, ['Matt'])

    def test_get_candidates_distance_gt_one(self):
        """ test get candidate with distance .gt. one"""
        self.assertRaises(ValueError, fornax.select.get_candidate, 1.1, ['Matt'])

    def test_get_neighbours(self):
        """ test get neighbours """
        query = fornax.select.get_candidate(0.7, 'Matt')
        query = fornax.select.get_neighbours(query)
        rows = query.with_session(self.session).all()
        self.assertEqual([row.label for row in rows], ['Dom'])

    def test_get_many_neighbours(self):
        """ test get many neighbours """
        query = self.session.query(model.Node, literal("").label('search_label'))
        query = query.filter(model.Node.id < 2)
        query = fornax.select.get_neighbours(query)
        rows = query.with_session(self.session).all()
        self.assertEqual([row.label for row in rows], ['Dom'])

    def test_distance(self):
        """ test distance"""
        query = self.session.query(model.Node, literal("").label('search_label'))
        query = query.filter(model.Node.id < 2)
        query = fornax.select.get_neighbours(query)
        row = query.with_session(self.session).first()
        self.assertIsNotNone(row)
        self.assertEqual(row.distance, 1)

    def test_get_neighbours_reccursive(self):
        """ get next nearest neighbours """
        #TODO: Move reccursion outside of testing 
        query = fornax.select.get_candidate(0.7, 'Matt')
        query = fornax.select.get_neighbours(query).union(
            fornax.select.get_neighbours(
                fornax.select.get_neighbours(query)
            )
        )
        rows = query.with_session(self.session).all()
        self.assertEqual([row.label for row in rows], ['Dom'])

    def test_get_all_neighbours(self):
        """ get all the neighbours for a set of fuzzy labels """
        queries = []
        for label in ['Mat', 'Calum']:
            # Fuzzy match each label
            query = fornax.select.get_candidate(0.7, label)
            # Get all the neighbours of all the matches
            query = fornax.select.get_neighbours(query)
            queries.append(query)
        # Get the union of all the results
        query = queries[0].union_all(*queries[1:])
        rows = query.with_session(self.session).all()
        self.assertEqual(
            sorted([row.label for row in rows]),
            sorted(['Dom', 'David'])
        )

    def test_rows_to_table(self):
        query = fornax.select.get_candidate(0.7, 'Mat')
        query = fornax.select.get_neighbours(query)
        rows = query.with_session(self.session).all()
        table = fornax.select.Table(rows[0]._fields, rows)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0].id, 1)


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