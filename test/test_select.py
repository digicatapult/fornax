import unittest
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
        nodes = query.with_session(self.session).all()
        self.assertEqual(
            sorted([node.label for node in nodes]), 
            sorted(['Matt'])
        )

    def test_get_candidates_empty(self):
        """ test get_candidates with empty string """
        query = fornax.select.get_candidate(0.7, [])
        nodes = query.with_session(self.session).all()
        self.assertEqual([node.label for node in nodes], [])

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
        nodes = query.with_session(self.session).all()
        self.assertEqual([node.label for node in nodes], ['Dom'])

    def test_get_many_neighbours(self):
        """ test get many neighbours """
        query = self.session.query(model.Node).filter(model.Node.id < 2)
        query = fornax.select.get_neighbours(query)
        nodes = query.with_session(self.session).all()
        self.assertEqual([node.label for node in nodes], ['Dom'])

    def test_get_neighbours_reccursive(self):
        """ get next nearest neighbours """
        #TODO: Move reccursion outside of testing 
        query = fornax.select.get_candidate(0.7, 'Matt')
        query = fornax.select.get_neighbours(query).union(
            fornax.select.get_neighbours(
                fornax.select.get_neighbours(query)
            )
        )
        nodes = query.with_session(self.session).all()
        self.assertEqual([node.label for node in nodes], ['Dom'])

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
        query = Query(model.Node).limit(0).union(*queries)
        neighbour_nodes = query.with_session(self.session).all()
        self.assertEqual(
            sorted([node.label for node in neighbour_nodes]),
            sorted(['Dom', 'David'])
        )
