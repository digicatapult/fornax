import unittest
import numpy as np
import fornax.model as model
from sqlalchemy.sql import func
from sqlalchemy.orm import Query
from test_base import TestCaseDB

import fornax.select

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
        query = self.session.query(model.Node).filter(model.Node.id < 2)
        query = fornax.select.get_neighbours(query)
        rows = query.with_session(self.session).all()
        self.assertEqual([row.label for row in rows], ['Dom'])

    def test_distance(self):
        """ test distance"""
        query = Query(model.Node)
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

    def test_to_dict(self):
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
        d = fornax.select.to_dict(rows)
        target = {
            'id': np.array([1, 3]),
            'label': np.array(['Dom', 'David']),
            'distance': np.array([1, 1]),
            'parent': np.array([0, 2]),
            'type': [0, 0]
        }
        for key in target:
            self.assertEqual(
                sorted(d[key]), 
                sorted(target[key]),
                'for key {}'.format(key)
            )

