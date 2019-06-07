import unittest
import json
import fornax.api
import fornax.model
from test_base import TestCaseDB
from sqlalchemy.orm.session import Session
from unittest import TestCase


class DummyException(Exception):
    pass


class TestConnection(TestCaseDB):

    def test_rollback(self):
        """ Test than the connection rolls back
        transactions if theres an exception
        """
        dburl = 'sqlite://'
        try:
            # raise an exception afer creatiion
            # then raise an exception
            with fornax.Connection(dburl) as conn:
                graph = fornax.GraphHandle.create(conn)
                names = ['adam', 'ben', 'chris']
                graph.add_nodes(name=names)
                raise DummyException
        except DummyException:
            pass
        finally:
            # everything should be gone
            with fornax.Connection(dburl) as conn:
                n_nodes = conn.session.query(fornax.model.Node).count()
                n_graphs = conn.session.query(fornax.model.Graph).count()
            self.assertEqual(n_nodes, 0)
            self.assertEqual(n_graphs, 0)


class TestGraph(TestCaseDB):

    def run(self, result=None):
        with fornax.Connection('sqlite://') as conn:
            self.conn = conn
            self.conn.make_session = lambda: Session(self._connection)
            super().run(result)

    def test_init_raises(self):
        """ raise an ValueError if a hadle to a graph
        is constructed that does not exist
        """
        self.assertRaises(ValueError, fornax.GraphHandle, self.conn, 0)
        self.assertRaises(ValueError, fornax.GraphHandle.read, self.conn, 0)

    def test_create(self):
        """first graph has id zero
        """
        graph = fornax.GraphHandle.create(self.conn)
        self.assertEqual(graph.graph_id, 0)

    def test_create_two(self):
        """auto increment graph id
        """
        _ = fornax.GraphHandle.create(self.conn)
        second = fornax.GraphHandle.create(self.conn)
        self.assertEqual(second.graph_id, 1)

    def test_read(self):
        """get a graph handle using graph id
        """
        graph = fornax.GraphHandle.create(self.conn)
        graph_id = graph.graph_id
        same_graph = fornax.GraphHandle.read(self.conn, graph_id)
        self.assertEqual(same_graph.graph_id, graph_id)

    def test_delete(self):
        """getting a deleted graph should raise a value error
        """
        graph = fornax.GraphHandle.create(self.conn)
        graph.delete()
        self.assertRaises(ValueError, fornax.GraphHandle.read, self.conn, 0)

        graph = fornax.GraphHandle.create(self.conn)
        graph.add_nodes(id_src=[0, 1, 2])
        graph.add_edges([0, 2], [1, 1])
        graph.delete()

    def test_add_nodes(self):
        """meta data is stored on a node
        """
        graph = fornax.GraphHandle.create(self.conn)
        names = ['adam', 'ben', 'chris']
        graph.add_nodes(name=names)
        nodes = self.conn.session.query(fornax.model.Node).filter(
            fornax.model.Node.graph_id == 0).all()
        nodes = sorted(nodes, key=lambda node: node.node_id)
        self.assertListEqual(
            names, [json.loads(node.meta)['name'] for node in nodes])

    def test_add_nodes_more_meta(self):
        """multiple metadata is stored on a node
        """
        graph = fornax.GraphHandle.create(self.conn)
        names = ['adam', 'ben', 'chris']
        ages = [9, 10, 11]
        graph.add_nodes(name=names, age=ages)
        nodes = self.conn.session.query(fornax.model.Node).filter(
            fornax.model.Node.graph_id == 0).all()
        nodes = sorted(nodes, key=lambda node: node.node_id)
        self.assertListEqual(
            names, [json.loads(node.meta)['name'] for node in nodes])
        self.assertListEqual(
            ages, [json.loads(node.meta)['age'] for node in nodes])

    def test_missing_attribute(self):
        """Null values for metadata must be explicit
        """
        graph = fornax.GraphHandle.create(self.conn)
        names = ['adam', 'ben', 'chris']
        ages = [9, 10]
        self.assertRaises(TypeError, graph.add_nodes, name=names, age=ages)

    def test_assign_id(self):
        """assigning node id is forbidden
        """
        graph = fornax.GraphHandle.create(self.conn)
        ids = range(3)
        self.assertRaises(ValueError, graph.add_nodes, id=ids)

    def test_no_metadata(self):
        """Nodes must have some metadata associated with them
        """
        graph = fornax.GraphHandle.create(self.conn)
        self.assertRaises(ValueError, graph.add_nodes)

    def test_add_edges(self):
        """store metadata on edges
        """
        graph = fornax.GraphHandle.create(self.conn)
        names = ['adam', 'ben', 'chris']
        ages = [9, 10, 11]
        graph.add_nodes(name=names, age=ages)
        relationships = ['is_friend', 'is_foe']
        graph.add_edges([0, 0], [1, 2], relationship=relationships)
        edges = self.conn.session.query(
            fornax.model.Edge
        ).filter(
            fornax.model.Edge.graph_id == graph.graph_id
        ).filter(
            fornax.model.Edge.start < fornax.model.Edge.end
        ).all()
        edges = sorted(edges, key=lambda edge: (edge.start, edge.end))
        self.assertListEqual(relationships, [json.loads(
            edge.meta)['relationship'] for edge in edges])

    def test_add_edges_more_meta(self):
        """store multiple metadata on edges
        """
        graph = fornax.GraphHandle.create(self.conn)
        names = ['adam', 'ben', 'chris']
        ages = [9, 10, 11]
        graph.add_nodes(name=names, age=ages)
        relationships = ['is_friend', 'is_foe']
        types = [0, 1]
        graph.add_edges(
            [0, 0], [1, 2], relationship=relationships, type_=types)
        edges = self.conn.session.query(
            fornax.model.Edge
        ).filter(
            fornax.model.Edge.graph_id == graph.graph_id
        ).filter(
            fornax.model.Edge.start < fornax.model.Edge.end
        ).all()
        edges = sorted(edges, key=lambda edge: (edge.start, edge.end))
        self.assertListEqual(relationships, [json.loads(
            edge.meta)['relationship'] for edge in edges])
        self.assertListEqual(
            types, [json.loads(edge.meta)['type_'] for edge in edges])

    def test_simple_graph(self):
        """Test for a simple graph.
        A simple graph is a graph with no loops.
        A loop is an edge that connects a vertex to itself
        """
        graph = fornax.GraphHandle.create(self.conn)
        names = ['adam', 'ben', 'chris']
        ages = [9, 10, 11]
        graph.add_nodes(name=names, age=ages)
        self.assertRaises(fornax.api.InvalidEdgeError, graph.add_edges, [
                          1, 0], [1, 2], relationship=['is_friend', 'is_foe'])

    def test_add_nodes_id_src(self):
        graph = fornax.GraphHandle.create(self.conn)
        graph.add_nodes(id_src=['a', 'b', 'c', 'd'])
        graph.add_edges(['a', 'b'], ['b', 'c'])
        nodes = self.conn.session.query(fornax.model.Node).all()
        self.assertEqual(
            [n.node_id for n in nodes],
            [self.conn._hash(item) for item in ('a', 'b', 'c', 'd')]
        )

    def test_add_nodes_id_src_meta(self):
        graph = fornax.GraphHandle.create(self.conn)
        graph.add_nodes(id_src=['a', 'b', 'c', 'd'])
        graph.add_edges(['a', 'b'], ['b', 'c'])
        nodes = self.conn.session.query(fornax.model.Node).all()
        self.assertEqual(
            [json.loads(n.meta)['id_src'] for n in nodes],
            ['a', 'b', 'c', 'd']
        )

    def test_add_edges_id_src(self):
        graph = fornax.GraphHandle.create(self.conn)
        graph.add_nodes(id_src=['a', 'b', 'c', 'd'])
        graph.add_edges(['a', 'b'], ['b', 'c'])
        edges = self.conn.session.query(
            fornax.model.Edge
        ).filter(
            fornax.model.Edge.start < fornax.model.Edge.end
        ).all()
        self.assertEqual(
            sorted([e.start, e.end] for e in edges),
            sorted(
                sorted([self.conn._hash(start), self.conn._hash(end)])
                for start, end in [('a', 'b'), ('b', 'c')]
            )
        )          


class TestQuery(TestCaseDB):

    def run(self, result=None):
        with fornax.Connection('sqlite://') as conn:
            self.conn = conn
            self.conn.make_session = lambda: Session(self._connection)
            super().run(result)

    def test_init_query_raises(self):
        self.assertRaises(ValueError, fornax.QueryHandle, self.conn, 0)

    def test_init_read_raises(self):
        self.assertRaises(ValueError, fornax.QueryHandle.read, self.conn, 0)

    def test_create(self):
        query_graphs = [fornax.GraphHandle.create(self.conn) for _ in range(3)]
        target_graphs = [
            fornax.GraphHandle.create(self.conn) for _ in range(3)
        ]
        queries = [fornax.QueryHandle.create(
            self.conn, q, t) for q, t in zip(query_graphs, target_graphs)]
        self.assertEqual([q.query_id for q in queries], [0, 1, 2])

    def test_create_query_target(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        query_db = self.conn.session.query(fornax.model.Query).filter(
            fornax.model.Query.query_id == query.query_id).first()
        self.assertEqual(query_db.start_graph_id, query_graph.graph_id)
        self.assertEqual(query_db.end_graph_id, target_graph.graph_id)

    def test_read(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        q1 = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        q2 = fornax.QueryHandle.read(self.conn, q1.query_id)
        self.assertEqual(q1.query_id, q2.query_id)

    def test_delete(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        query_graph.add_nodes(id_src=[0, 1])
        query_graph.add_edges([0], [1])
        target_graph = fornax.GraphHandle.create(self.conn)
        target_graph.add_nodes(id_src=[1, 2])
        target_graph.add_edges([2], [1])
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        query.add_matches([0, 1], [2, 1], [1, 1])
        query_id = query.query_id
        query.delete()
        query_exists = self.conn.session.query(fornax.model.Query).filter(
            fornax.model.Query.query_id == query_id).scalar()
        matches_exists = self.conn.session.query(fornax.model.Match).filter(
            fornax.model.Match.query_id == query_id).scalar()
        self.assertFalse(query_exists)
        self.assertFalse(matches_exists)

    def test_get_query_graph(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        self.assertEqual(query.query_graph(), query_graph)

    def test_get_target_graph(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        self.assertEqual(query.target_graph(), target_graph)

    def test_query_nodes(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        q_uids = [0, 1, 2]
        t_uids = [3, 4, 5]
        query_graph.add_nodes(uid=q_uids)
        target_graph.add_nodes(uid=t_uids)
        query_nodes = query._query_nodes()
        self.assertListEqual(
            [fornax.api.Node(i, 'query', {'uid': uid})
             for i, uid in enumerate(q_uids)],
            query_nodes
        )

    def test_query_edges(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        q_uids = [0, 1, 2]
        t_uids = [3, 4, 5]
        query_graph.add_nodes(uid=q_uids)
        query_graph.add_edges([0, 0], [1, 2], my_id=['a', 'b'])
        target_graph.add_nodes(uid=t_uids)
        query_edges = query._query_edges()
        self.assertListEqual(
            [fornax.api.Edge(0, 1, 'query', {'my_id': 'a'}), fornax.api.Edge(
                0, 2, 'query', {'my_id': 'b'})],
            query_edges
        )

    def test_target_nodes(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        q_uids = [0, 1, 2]
        t_uids = [3, 4, 5]
        query_graph.add_nodes(id_src=q_uids)
        target_graph.add_nodes(id_src=t_uids)
        #  no target nodes will appear if there are no matches
        query.add_matches(q_uids, t_uids, [1, 1, 1])
        target_nodes = query._target_nodes()
        self.assertListEqual(
            [
                fornax.api.Node(uid, 'target', {'id_src': uid})
                for uid in t_uids
            ],
            target_nodes
        )

    def test_undirected_edges(self):
        """Each edge needs to be stored in both directions
        """
        graph = fornax.GraphHandle.create(self.conn)
        graph.add_nodes(myid=[1, 2, 3])
        graph.add_edges([0], [1])
        src = [
            (e.start, e.end)
            for e in self.conn.session.query(fornax.model.Edge).all()
        ]
        self.assertListEqual(sorted(src), [(0, 1), (1, 0)])

    def test_target_edges(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        uids = [0, 1]
        query_graph.add_nodes(uid=range(3))
        target_graph.add_nodes(uid=range(3))
        target_graph.add_edges([0, 1], [1, 2])
        sources = [0, 0]
        targets = [1, 2]
        weights = [1, 1]
        query.add_matches(sources, targets, weights, my_id=uids)
        _, _, _, _, target_edges_arr = query._optimise(2, 10, None)
        target_edges = query._target_edges(
            query._target_nodes(), target_edges_arr)
        self.assertListEqual(
            [
                #  no matches end on node 0
                #  fornax.api.Edge(0, 1, 'target', dict()),
                fornax.api.Edge(1, 2, 'target', dict())
            ],
            target_edges
        )

    def test_add_matches(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        uids = [0, 1]
        query_graph.add_nodes(uid=range(3))
        target_graph.add_nodes(uid=range(3))
        sources = [0, 0]
        targets = [1, 2]
        weights = [1, 1]
        query.add_matches(sources, targets, weights, my_id=uids)
        matches = self.conn.session.query(fornax.model.Match).filter(
            fornax.model.Match.query_id == query.query_id
        ).order_by(fornax.model.Match.end.asc())
        self.assertEqual(sources, [m.start for m in matches])
        self.assertEqual(targets, [m.end for m in matches])
        self.assertEqual(weights, [m.weight for m in matches])
        self.assertEqual(uids, [json.loads(m.meta)['my_id'] for m in matches])

    def test_execute_raises(self):
        query_graph = fornax.GraphHandle.create(self.conn)
        target_graph = fornax.GraphHandle.create(self.conn)
        query = fornax.QueryHandle.create(self.conn, query_graph, target_graph)
        self.assertRaises(ValueError, query.execute)


class TestNode(TestCase):

    def setUp(self):
        self.node = fornax.api.Node(0, 'query', {'a': 1})

    def test_id(self):
        self.assertEqual(self.node.id, 0)

    def test_meta(self):
        self.assertEqual(self.node.meta, {'a': 1})

    def test_eq(self):
        self.assertEqual(self.node, fornax.api.Node(0, 'query', {'a': 1}))
        self.assertNotEqual(self.node, fornax.api.Node(1, 'query', {'a': 1}))
        self.assertNotEqual(self.node, fornax.api.Node(0, 'query', {'a': 0}))
        self.assertNotEqual(self.node, fornax.api.Node(1, 'query', {'a': 0}))
        self.assertNotEqual(self.node, fornax.api.Node(0, 'target', {'a': 1}))

    def test_node_raises(self):
        self.assertRaises(ValueError, fornax.api.Node, 0, 'a', {})


class TestEdge(TestCase):

    def setUp(self):
        self.edge = fornax.api.Edge(0, 1, 'query', {'a': 1})

    def test_start(self):
        self.assertEqual(self.edge.start, 0)

    def test_end(self):
        self.assertEqual(self.edge.end, 1)

    def test_meta(self):
        self.assertEqual(self.edge.meta, {'a': 1})

    def test_eq(self):
        self.assertEqual(self.edge, fornax.api.Edge(0, 1, 'query', {'a': 1}))
        self.assertNotEqual(
            self.edge, fornax.api.Edge(1, 1, 'query', {'a': 1}))
        self.assertNotEqual(
            self.edge, fornax.api.Edge(0, 0, 'query', {'a': 1}))
        self.assertNotEqual(
            self.edge, fornax.api.Edge(0, 1, 'query', {'a': 2}))
        self.assertNotEqual(
            self.edge, fornax.api.Edge(0, 1, 'target', {'a': 1}))

    def test_edge_raises(self):
        self.assertRaises(ValueError, fornax.api.Edge, 0, 1, 'a', {})


class TestExample(TestCaseDB):

    @classmethod
    def setUp(self):
        """trick fornax into using the test database setup
        """
        super().setUp(self)
        self.maxsize = fornax.Connection.SQLITE_MAX_SIZE
        with fornax.Connection('sqlite://') as conn:
            conn.make_session = lambda: Session(self._connection)
            query_graph = fornax.GraphHandle.create(conn)
            query_graph.add_nodes(my_id=range(1, 6))
            starts, ends = zip(*[(1, 3), (1, 2), (2, 4), (4, 5)])
            query_graph.add_edges(
                [i - 1 for i in starts],
                [i - 1 for i in ends]
            )

            target_graph = fornax.GraphHandle.create(conn)
            target_graph.add_nodes(my_id=range(1, 14))
            starts, ends = zip(*[
                (1, 2), (1, 3), (1, 4),
                (3, 7), (4, 5), (4, 6),
                (5, 7), (6, 8), (7, 10),
                (8, 9), (8, 12), (9, 10),
                (10, 11), (11, 12), (11, 13),
            ])
            target_graph.add_edges(
                [s - 1 for s in starts],
                [e - 1 for e in ends]
            )

            query = fornax.QueryHandle.create(
                conn,
                query_graph,
                target_graph
            )
            starts, ends, weights = zip(*[
                (1, 1, 1), (1, 4, 1), (1, 8, 1),
                (2, 2, 1), (2, 5, 1), (2, 9, 1),
                (3, 3, 1), (3, 6, 1), (3, 12, 1), (3, 13, 1),
                (4, 7, 1), (4, 10, 1),
                (5, 11, 1)
            ])

            query.add_matches(
                [s - 1 for s in starts],
                [e - 1 for e in ends],
                weights
            )

            self.payload = query.execute(n=2)

    def test_iters(self):
        self.assertEqual(self.payload['max_iters'], 10)

    def test_hopping_distance(self):
        self.assertEqual(self.payload['hopping_distance'], 2)

    def test_first_graph_cost(self):
        graph = self.payload['graphs'][0]
        self.assertEqual(graph['cost'], 0)

    def test_first_graph_nodes(self):
        graph = self.payload['graphs'][1]
        nodes = [
            {"id": 0, "type": "query", "my_id": 1},
            {"id": 1, "type": "query", "my_id": 2},
            {"id": 2, "type": "query", "my_id": 3},
            {"id": 3, "type": "query", "my_id": 4},
            {"id": 4, "type": "query", "my_id": 5},
            {"id": 7, "type": "target", "my_id": 8},
            {"id": 8, "type": "target", "my_id": 9},
            {"id": 9, "type": "target", "my_id": 10},
            {"id": 10, "type": "target", "my_id": 11},
            {"id": 11, "type": "target", "my_id": 12}
        ]
        for node in nodes:
            node['id'] = fornax.api._hash(
                (node['id'], node['type']),
                self.maxsize
            )
        self.assertListEqual(
            graph['nodes'],
            nodes
        )

    def test_first_graph_links(self):

        graph = self.payload['graphs'][1]

        matches = [
            {"source": 0, "target": 7, "type": "match", "weight": 1.0},
            {"source": 1, "target": 8, "type": "match", "weight": 1.0},
            {"source": 2, "target": 11, "type": "match", "weight": 1.0},
            {"source": 3, "target": 9, "type": "match", "weight": 1.0},
            {"source": 4, "target": 10, "type": "match", "weight": 1.0},
            {"source": 0, "target": 1, "type": "query", "weight": 1.0},
            {"source": 0, "target": 2, "type": "query", "weight": 1.0},
            {"source": 1, "target": 3, "type": "query", "weight": 1.0},
            {"source": 3, "target": 4, "type": "query", "weight": 1.0},
            {"source": 7, "target": 8, "type": "target", "weight": 1.0},
            {"source": 7, "target": 11, "type": "target", "weight": 1.0},
            {"source": 8, "target": 9, "type": "target", "weight": 1.0},
            {"source": 9, "target": 10, "type": "target", "weight": 1.0},
            {"source": 10, "target": 11, "type": "target", "weight": 1.0},
        ]

        for match in matches:

            if match['type'] == 'query' or match['type'] == 'target':

                match['source'] = fornax.api._hash(
                    (match['source'], match['type']),
                    self.maxsize
                )

                match['target'] = fornax.api._hash(
                    (match['target'], match['type']),
                    self.maxsize
                )

            else:

                match['source'] = fornax.api._hash(
                    (match['source'], 'query'),
                    self.maxsize
                )

                match['target'] = fornax.api._hash(
                    (match['target'], 'target'),
                    self.maxsize
                )

        self.assertListEqual(graph['links'], matches)

    def test_second_graph_cost(self):
        graph = self.payload['graphs'][1]
        self.assertEqual(graph['cost'], 0)

    def test_second_graph_nodes(self):
        graph = self.payload['graphs'][0]
        nodes = [
            {"id": 0, "type": "query", "my_id": 1},
            {"id": 1, "type": "query", "my_id": 2},
            {"id": 2, "type": "query", "my_id": 3},
            {"id": 3, "type": "query", "my_id": 4},
            {"id": 4, "type": "query", "my_id": 5},
            {"id": 5, "type": "target", "my_id": 6},
            {"id": 7, "type": "target", "my_id": 8},
            {"id": 8, "type": "target", "my_id": 9},
            {"id": 9, "type": "target", "my_id": 10},
            {"id": 10, "type": "target", "my_id": 11},
        ]
        for node in nodes:
            node['id'] = fornax.api._hash(
                (node['id'], node['type']),
                self.maxsize
            )

        self.assertListEqual(
            graph['nodes'],
            nodes
        )

    def test_second_graph_links(self):
        graph = self.payload['graphs'][0]
        matches = [
            {"source": 0, "target": 7, "type": "match", "weight": 1.0},
            {"source": 1, "target": 8, "type": "match", "weight": 1.0},
            {"source": 2, "target": 5, "type": "match", "weight": 1.0},
            {"source": 3, "target": 9, "type": "match", "weight": 1.0},
            {"source": 4, "target": 10, "type": "match", "weight": 1.0},
            {"source": 0, "target": 1, "type": "query", "weight": 1.0},
            {"source": 0, "target": 2, "type": "query", "weight": 1.0},
            {"source": 1, "target": 3, "type": "query", "weight": 1.0},
            {"source": 3, "target": 4, "type": "query", "weight": 1.0},
            {"source": 5, "target": 7, "type": "target", "weight": 1.0},
            {"source": 7, "target": 8, "type": "target", "weight": 1.0},
            {"source": 8, "target": 9, "type": "target", "weight": 1.0},
            {"source": 9, "target": 10, "type": "target", "weight": 1.0},
        ]

        for match in matches:

            if match['type'] == 'query' or match['type'] == 'target':

                match['source'] = fornax.api._hash(
                    (match['source'], match['type']),
                    self.maxsize
                )

                match['target'] = fornax.api._hash(
                    (match['target'], match['type']),
                    self.maxsize
                )

            elif match['type'] == 'match':

                match['source'] = fornax.api._hash(
                    (match['source'], 'query'),
                    self.maxsize
                )

                match['target'] = fornax.api._hash(
                    (match['target'], 'target'),
                    self.maxsize
                )

        self.assertListEqual(graph['links'], matches)
