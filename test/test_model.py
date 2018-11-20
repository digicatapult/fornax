import sqlalchemy
import unittest
import fornax.model as model
from test_base import TestCaseDB


class TestNode(TestCaseDB):

    def test_node_round_trip(self):
        """ node round trip """
        new_graph = model.Graph(graph_id=0)
        self.session.add(new_graph)
        self.session.commit()
        new_node = model.Node(node_id=0, graph_id=0)
        self.session.add(new_node)
        self.session.commit()

        row = self.session.query(model.Node).first()
        self.assertIsNotNone(row)


class TestEdge(TestCaseDB):

    def setUp(self):
        super().setUp()
        new_graphs = [model.Graph(graph_id=0), model.Graph(graph_id=1)]
        self.session.add_all(new_graphs)
        self.session.commit()
        new_nodes = [model.Node(node_id=id_, graph_id=0) for id_ in range(2)]
        new_nodes += [model.Node(node_id=id_, graph_id=1) for id_ in range(2)]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.Edge(start=0, end=1, graph_id=0),
            model.Edge(start=1, end=0, graph_id=1),
        ]
        self.session.add_all(new_edges)
        self.session.commit()

    def test_query_edge_round_trip(self):
        """ edge round trip """
        row = self.session.query(model.Edge).first()
        self.assertIsNotNone(row)
        self.assertEqual(0, row.start)

    def test_edge_join_start(self):
        """ find a node by joining on the start of an edge """
        query = self.session.query(model.Node)
        query = query.join(
            model.Edge,
            sqlalchemy.and_(
                model.Node.node_id == model.Edge.start,
                model.Node.graph_id == model.Edge.graph_id
            )
        )
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.node_id, 0)

    def test_edge_join_end(self):
        """ find a node by joining on the end of an edge """
        query = self.session.query(model.Node)
        query = query.join(
            model.Edge,
            sqlalchemy.and_(
                model.Node.node_id == model.Edge.end,
                model.Node.graph_id == model.Edge.graph_id
            )
        )
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.node_id, 1)


class TestNeighbours(TestCaseDB):

    def setUp(self):
        super().setUp()
        new_graph = model.Graph(graph_id=0)
        self.session.add(new_graph)
        self.session.commit()
        new_nodes = [model.Node(node_id=id_, graph_id=0) for id_ in range(4)]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.Edge(start=0, end=1, graph_id=0),
            model.Edge(start=0, end=2, graph_id=0),
            model.Edge(start=2, end=3, graph_id=0),
        ]

        self.session.add_all(new_edges)
        self.session.commit()

    def test_neighbours(self):
        query = self.session.query(model.Node)
        query = query.filter(model.Node.node_id == 0)
        node = query.first()
        self.assertListEqual(
            [n.node_id for n in node.neighbours()],
            [1, 2]
        )

    def test_next_neighbours(self):
        query = self.session.query(model.Node)
        query = query.filter(model.Node.node_id == 0)
        node = query.first()
        self.assertListEqual(
            [n2.node_id for n1 in node.neighbours() for n2 in n1.neighbours()],
            [3]
        )


class TestMatch(TestCaseDB):

    def setUp(self):
        super().setUp()
        new_graphs = [model.Graph(graph_id=0), model.Graph(graph_id=1)]
        self.session.add_all(new_graphs)
        self.session.commit()
        new_query = model.Query(query_id=0, start_graph_id=0, end_graph_id=1)
        self.session.add(new_query)
        self.session.commit()

        new_nodes = [
            model.Node(node_id=0, graph_id=0),
            model.Node(node_id=0, graph_id=1),
            model.Node(node_id=1, graph_id=0),
            model.Node(node_id=1, graph_id=1)
        ]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.Match(start=0, end=0, weight=1, start_graph_id=0,
                        end_graph_id=1, query_id=0),
            model.Match(start=1, end=0, weight=1, start_graph_id=0,
                        end_graph_id=1, query_id=0),
            model.Match(start=1, end=1, weight=1, start_graph_id=0,
                        end_graph_id=1, query_id=0)
        ]
        self.session.add_all(new_edges)
        self.session.commit()

    def test_select_by_match_left(self):
        query = self.session.query(model.Node)
        query = query.filter(
            model.Node.start_matches.any(
                model.Match.start == 0
            )
        )
        nodes = query.all()
        self.assertLessEqual(
            [node.node_id for node in nodes],
            [0, 1]
        )

    def test_test_min_check(self):
        self.session.add(model.Match(start=0, end=0, weight=1.1,
                                     start_graph_id=0, end_graph_id=0))
        self.assertRaises(
            sqlalchemy.exc.IntegrityError,
            self.session.commit
        )

    def test_test_max_check(self):
        self.session.add(model.Match(start=0, end=0, weight=0,
                                     start_graph_id=0, end_graph_id=0))
        self.assertRaises(
            sqlalchemy.exc.IntegrityError,
            self.session.commit
        )


if __name__ == '__main__':
    unittest.main()
