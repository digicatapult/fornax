import sqlalchemy
import unittest
import fornax.model as model
from test_base import TestCaseDB


class TestNode(TestCaseDB):

    def test_node_round_trip(self):
        """ node round trip """
        new_node = model.Node(id=0, gid=0)
        self.session.add(new_node)
        self.session.commit()

        row = self.session.query(model.Node).first()
        self.assertIsNotNone(row)


class TestEdge(TestCaseDB):

    def setUp(self):
        super().setUp()

        new_nodes = [model.Node(id=id_, gid=0) for id_ in range(2)]
        new_nodes += [model.Node(id=id_, gid=1) for id_ in range(2)]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.Edge(start=0, end=1, gid=0), 
            model.Edge(start=1, end=0, gid=1),
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
                model.Node.id==model.Edge.start,
                model.Node.gid==model.Edge.gid
            )
        )
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.id, 0)

    def test_edge_join_end(self):
        """ find a node by joining on the end of an edge """
        query = self.session.query(model.Node)
        query = query.join(
            model.Edge, 
            sqlalchemy.and_(
                model.Node.id==model.Edge.end,
                model.Node.gid==model.Edge.gid
            )
        )
        row = query.first()
        self.assertIsNotNone(row)
        self.assertEqual(row.id, 1)


class TestNeighbours(TestCaseDB):

    def setUp(self):
        super().setUp()
        new_nodes = [model.Node(id=id_, gid=0) for id_ in range(4)]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.Edge(start=0, end=1, gid=0),
            model.Edge(start=0, end=2, gid=0), 
            model.Edge(start=2, end=3, gid=0), 
        ]

        self.session.add_all(new_edges)
        self.session.commit()

    def test_neighbours(self):
        query = self.session.query(model.Node)
        query = query.filter(model.Node.id == 0)
        node = query.first()
        self.assertListEqual(
            [n.id for n in node.neighbours()],
            [1, 2]
        )

    def test_next_neighbours(self):
        query = self.session.query(model.Node)
        query = query.filter(model.Node.id == 0)
        node = query.first()
        self.assertListEqual(
            [n2.id for n1 in node.neighbours() for n2 in n1.neighbours()],
            [3]
        )


class TestMatch(TestCaseDB):

    def setUp(self):
        super().setUp()
        new_nodes = [
            model.Node(id=0, gid=0),
            model.Node(id=0, gid=1),
            model.Node(id=1, gid=0),
            model.Node(id=1, gid=1)
        ]
        self.session.add_all(new_nodes)
        self.session.commit()

        new_edges = [
            model.Match(start=0, end=0, weight=1, start_gid=0, end_gid=0),
            model.Match(start=1, end=0, weight=1, start_gid=0, end_gid=0),
            model.Match(start=1, end=1, weight=1, start_gid=0, end_gid=0)
        ]
        self.session.add_all(new_edges)
        self.session.commit()

    def test_select_by_match_left(self):
        query = self.session.query(model.Node)
        query = query.filter(
            model.Node.matches.any(
                model.Match.start == 0
            )
        )
        nodes = query.all()
        self.assertLessEqual(
            [node.id for node in nodes],
            [0, 1]
        )

    def test_test_min_check(self):
        self.session.add(model.Match(start=0, end=0, weight=1.1, start_gid=0, end_gid=0))
        self.assertRaises(
            sqlalchemy.exc.IntegrityError,
            self.session.commit
        )    

    def test_test_max_check(self):
        self.session.add(model.Match(start=0, end=0, weight=0, start_gid=0, end_gid=0))
        self.assertRaises(
            sqlalchemy.exc.IntegrityError,
            self.session.commit
        )

if __name__ == '__main__':
    unittest.main()
