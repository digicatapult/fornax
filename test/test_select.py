import unittest
import fornax.model as model
import fornax.select as select
from test_base import TestCaseDB
from sqlalchemy.orm import Query
from sqlalchemy import literal, func, and_


class TestSelect(TestCaseDB):
    """Reproduce the scenario set out in figure 4 of the paper"""

    def setUp(self):
        super().setUp()

        new_graphs = [
            model.Graph(graph_id=0),
            model.Graph(graph_id=1)
        ]
        self.session.add_all(new_graphs)
        self.session.commit()

        # Create the query graph from figure 4
        new_nodes = [
            model.Node(node_id=id_+1, graph_id=0)
            for id_, label in enumerate('abcde')
        ]

        start_finish = [(1, 2), (1, 3), (2, 4), (4, 5)]

        new_edges = [
            model.Edge(start=start, end=end, graph_id=0)
            for start, end in start_finish
        ]
        new_edges += [
            model.Edge(start=end, end=start, graph_id=0)
            for start, end in start_finish
        ]

        self.session.add_all(new_nodes)
        self.session.add_all(new_edges)
        self.session.commit()

        # Create the target graph from figure 4 (ids are offset by 100)
        labels = 'abcabcdabdecc'
        start_finish = [
            (1, 2), (1, 3), (1, 4), (3, 7), (4, 5), (4, 6), (5, 7),
            (6, 8), (8, 9), (8, 12), (9, 10), (10, 7),
            (10, 11), (11, 12), (11, 13)
        ]
        new_nodes = [
            model.Node(node_id=id_+1, graph_id=1)
            for id_, label in enumerate(labels)
        ]

        new_edges = [
            model.Edge(start=start, end=end, graph_id=1)
            for start, end in start_finish
        ]
        new_edges += [
            model.Edge(start=end, end=start, graph_id=1)
            for start, end in start_finish
        ]

        self.session.add_all(new_nodes)
        self.session.commit()
        self.session.add_all(new_edges)
        self.session.commit()

        new_query = model.Query(query_id=0, start_graph_id=0, end_graph_id=1)
        self.session.add(new_query)
        self.session.commit()

        self.session.add_all(
            [
                model.Match(start=1, end=1, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=1, end=4, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=1, end=8, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=2, end=2, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=2, end=5, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=2, end=9, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=3, end=3, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=3, end=6, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=3, end=12, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=3, end=13, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=4, end=7, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=4, end=10, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
                model.Match(start=5, end=11, weight=1,
                            start_graph_id=0, end_graph_id=1, query_id=0),
            ]
        )
        self.session.commit()

    def test__neighbours_1(self):

        nodes = Query([
            model.Match.start.label('match'),
            model.Match.start_graph_id.label('graph_id'),
            model.Node.node_id.label('neighbour'),
            literal(0).label('distance')
        ]).join(
            model.Node,
            and_(
                model.Node.node_id == model.Match.start,
                model.Node.graph_id == model.Match.start_graph_id
            )
        )
        query = select._neighbours(nodes, 1, 1)
        records = query.with_session(self.session).all()
        self.assertListEqual(
            sorted(records),
            sorted([
                (1, 0, 2, 1), (1, 0, 3, 1), (2, 0, 1, 1),
                (2, 0, 4, 1), (3, 0, 1, 1), (4, 0, 2, 1),
                (4, 0, 5, 1), (5, 0, 4, 1)
            ])
        )

    def test__neighbours_2(self):

        nodes = Query([
            model.Match.start.label('match'),
            model.Match.start_graph_id.label('graph_id'),
            model.Node.node_id.label('neighbour'),
            literal(0).label('distance')
        ]).join(
            model.Node,
            and_(
                model.Node.node_id == model.Match.start,
                model.Node.graph_id == model.Match.start_graph_id
            )
        )
        query = select._neighbours(nodes, 1, 2)
        records = query.with_session(self.session).all()
        self.assertListEqual(
            sorted(records),
            sorted([
                (1, 0, 1, 2), (1, 0, 2, 1), (1, 0, 3, 1),
                (1, 0, 4, 2), (2, 0, 2, 2), (2, 0, 1, 1),
                (2, 0, 3, 2), (2, 0, 4, 1), (2, 0, 5, 2),
                (3, 0, 3, 2), (3, 0, 1, 1), (3, 0, 2, 2),
                (4, 0, 4, 2), (4, 0, 1, 2), (4, 0, 2, 1),
                (4, 0, 5, 1), (5, 0, 2, 2), (5, 0, 4, 1),
                (5, 0, 5, 2),
            ])
        )

    def test_neighbours_1(self):
        query = select.neighbours(1, True)
        records = query.with_session(self.session).all()
        self.assertListEqual(
            sorted(records),
            sorted([
                (1, 1, 0), (1, 2, 1), (1, 3, 1),
                (2, 2, 0), (2, 1, 1), (2, 4, 1),
                (3, 3, 0), (3, 1, 1), (4, 4, 0),
                (4, 2, 1), (4, 5, 1), (5, 5, 0),
                (5, 4, 1)
            ])
        )

    def test_neighbours_2(self):
        query = select.neighbours(2, True)
        records = query.with_session(self.session).all()
        self.assertListEqual(
            sorted(records),
            sorted([
                (1, 1, 0), (1, 2, 1), (1, 3, 1),
                (1, 4, 2), (2, 2, 0), (2, 1, 1),
                (2, 3, 2), (2, 4, 1), (2, 5, 2),
                (3, 3, 0), (3, 1, 1), (3, 2, 2),
                (4, 4, 0), (4, 1, 2), (4, 2, 1),
                (4, 5, 1), (5, 2, 2), (5, 4, 1),
                (5, 5, 0)
            ])
        )

    def test_join(self):

        # delete a match to simulate misses in this query
        match = self.session.query(model.Match)
        match = match.filter(model.Match.start == 2)
        match = match.filter(model.Match.end == 2)
        match.delete()
        self.session.commit()

        query = select.join(0, 1)
        records = query.with_session(self.session).all()
        self.assertListEqual(
            sorted(filter(lambda x: x[0] == x[1] == 1, records)),
            sorted([
                (1, 1, 1, 1, 0, 0, 1.0),
                (1, 1, 1, 4, 0, 1, 1.0),
                # <- Node 2 has no correspondences
                (1, 1, 2, None, 1, None, 1.0),
                (1, 1, 3, 3, 1, 1, 1.0),  # in the target graph
            ])
        )

    def test_join_batch(self):
        """Test batching queries with hopping distance h=1
            batch size = 1
        """
        query = select.join(0, 1)
        records = query.with_session(self.session).all()

        # keep getting batches until nothing comes back
        batched_records, i, batch_size, finished = [], 0, 1, False
        while not finished:
            query = select.join(0, 1, [i, i+batch_size])
            next_batch = query.with_session(self.session).all()
            batched_records += next_batch

            if len(next_batch) == 0:
                finished = True

            i += batch_size

        self.assertListEqual(
            sorted(records),
            sorted(batched_records)
        )

    def test_join_batch_h2(self):
        """Test batching queries with hopping distance h=2
            use larger batch size for performance
        """

        query = select.join(0, 2)
        records = query.with_session(self.session).all()

        # keep getting batches until nothing comes back
        batched_records, i, batch_size, finished = [], 0, 20, False
        while not finished:
            query = select.join(0, 2, [i, i+batch_size])
            next_batch = query.with_session(self.session).all()
            batched_records += next_batch

            if len(next_batch) == 0:
                finished = True

            i += batch_size

        self.assertListEqual(
            sorted(records),
            sorted(batched_records)
        )

    def test_join_val_error(self):
        self.assertRaises(ValueError, select.join, 0, 1, offsets=[1])


if __name__ == '__main__':
    unittest.main()
