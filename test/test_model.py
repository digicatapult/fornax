import unittest
import fornax.model as model
from test_base import TestCaseDB


class TestNode(TestCaseDB):

    def test_query_node_round_trip(self):
        """ node round trip """
        new_node = model.QueryNode()
        self.session.add(new_node)
        self.session.commit()

        row = self.session.query(model.QueryNode).first()
        self.assertIsNotNone(row)

    def test_target_node_round_trip(self):
        """ node round trip """
        new_node = model.QueryNode()
        self.session.add(new_node)
        self.session.commit()

        row = self.session.query(model.QueryNode).first()
        self.assertIsNotNone(row)


class TestEdge(TestCaseDB):

    def setUp(self):
        super().setUp()

        new_nodes = [model.QueryNode(id=id_) for id_ in range(2)]
        new_nodes += [model.TargetNode(id=id_) for id_ in range(2)]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.QueryEdge(start=0, end=1), 
            model.TargetEdge(start=0, end=1),
            model.MatchingEdge(start=0, end=0, weight=1.)
        ]
        self.session.add_all(new_edges)
        self.session.commit()

    def test_query_edge_round_trip(self):
        """ edge round trip """
        row = self.session.query(model.QueryEdge).first()
        self.assertIsNotNone(row)
        self.assertEqual(0, row.start)

    def test_target_edge_round_trip(self):
        row = self.session.query(model.TargetEdge).first()
        self.assertIsNotNone(row)
        self.assertEqual(0, row.start)

    def test_query_edge_join_start(self):
        """ find a node by joining on the start of an edge """
        query = self.session.query(model.QueryNode)
        query = query.join(model.QueryEdge, model.QueryNode.id==model.QueryEdge.start)
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.id, 0)

    def test_target_edge_join_start(self):
        """ find a node by joining on the start of an edge """
        query = self.session.query(model.TargetNode)
        query = query.join(model.TargetEdge, model.TargetNode.id==model.TargetEdge.start)
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.id, 0)

    def test_query_edge_join_end(self):
        """ find a node by joining on the end of an edge """
        query = self.session.query(model.QueryNode)
        query = query.join(model.QueryEdge, model.QueryNode.id==model.QueryEdge.end)
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.id, 1)

    def test_target_edge_join_end(self):
        """ find a node by joining on the end of an edge """
        query = self.session.query(model.TargetNode)
        query = query.join(model.TargetEdge, model.TargetNode.id==model.TargetEdge.end)
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.id, 1)

    def test_match_edge_join_start(self):
        """ find a node by joining on the start of an edge """
        query = self.session.query(model.QueryNode)
        query = query.join(model.MatchingEdge, model.QueryNode.id==model.MatchingEdge.start)
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.id, 0)

    def test_match_edge_join_end(self):
        """ find a node by joining on the end of an edge """
        query = self.session.query(model.TargetNode)
        query = query.join(model.MatchingEdge, model.TargetNode.id==model.MatchingEdge.end)
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.id, 0)


class TestNeighboursQuery(TestCaseDB):

    def setUp(self):
        super().setUp()
        new_nodes = [model.QueryNode(id=id_) for id_ in range(4)]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.QueryEdge(start=0, end=1),
            model.QueryEdge(start=0, end=2), 
            model.QueryEdge(start=2, end=3), 
        ]
        self.session.add_all(new_edges)
        self.session.commit()

    def test_neighbours(self):
        query = self.session.query(model.QueryNode)
        query = query.filter(model.QueryNode.id == 0)
        node = query.first()
        self.assertListEqual(
            [n.id for n in node.neighbours],
            [1, 2]
        )

    def test_next_neighbours(self):
        query = self.session.query(model.QueryNode)
        query = query.filter(model.QueryNode.id == 0)
        node = query.first()
        self.assertListEqual(
            [n2.id for n1 in node.neighbours for n2 in n1.neighbours],
            [3]
        )


class TestNeighboursTarget(TestCaseDB):

    def setUp(self):
        super().setUp()
        new_nodes = [model.TargetNode(id=id_) for id_ in range(5)]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.TargetEdge(start=0, end=1),
            model.TargetEdge(start=0, end=2), 
            model.TargetEdge(start=2, end=3),
            model.TargetEdge(start=3, end=4),
            model.TargetEdge(start=3, end=0) 
        ]
        self.session.add_all(new_edges)
        self.session.commit()

    def test_neighbours(self):
        query = self.session.query(model.TargetNode)
        query = query.filter(model.TargetNode.id == 0)
        node = query.first()
        self.assertListEqual(
            [n.id for n in node.neighbours],
            [1, 2]
        )

    def test_next_neighbours(self):
        query = self.session.query(model.TargetNode)
        query = query.filter(model.TargetNode.id == 0)
        node = query.first()
        self.assertListEqual(
            [n2.id for n1 in node.neighbours for n2 in n1.neighbours],
            [3]
        )


if __name__ == '__main__':
    unittest.main()
