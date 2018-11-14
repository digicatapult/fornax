import unittest
import json
import fornax.api
import fornax.model
from test_base import TestCaseDB
from sqlalchemy.orm.session import Session


class TestGraph(TestCaseDB):

    @classmethod
    def setUp(self):
        """trick fornax into using the test database setup
        """
        super().setUp(self)
        fornax.api.Session = lambda: Session(self._connection)

    def test_init_raises(self):
        """ raise an ValueError if a hadle to a graph is constructed that does not exist """
        self.assertRaises(ValueError, fornax.GraphHandle, 0)
        self.assertRaises(ValueError, fornax.GraphHandle.read, 0)

    def test_create(self):
        """first graph has id zero
        """
        graph = fornax.GraphHandle.create()
        self.assertEqual(graph.graph_id, 0)

    def test_create_two(self):
        """auto increment graph id
        """
        _ = fornax.api.GraphHandle.create()
        second = fornax.GraphHandle.create()
        self.assertEqual(second.graph_id, 1)

    def test_read(self):
        """get a graph handle using graph id
        """
        graph = fornax.api.GraphHandle.create()
        graph_id = graph.graph_id
        same_graph = fornax.GraphHandle.read(graph_id)
        self.assertEqual(same_graph.graph_id, graph_id)

    def test_delete(self):
        """getting a deleted graph should raise a value error
        """
        graph = fornax.GraphHandle.create()
        graph.delete()
        self.assertRaises(ValueError, fornax.api.GraphHandle.read, 0)

    def test_add_nodes(self):
        """meta data is stored on a node
        """
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        graph.add_nodes(name=names)
        nodes = self.session.query(fornax.model.Node).filter(fornax.model.Node.graph_id==0).all()
        nodes = sorted(nodes, key=lambda node: node.node_id)
        self.assertListEqual(names, [json.loads(node.meta)['name'] for node in nodes])

    def test_add_nodes_more_meta(self):
        """multiple metadata is stored on a node
        """
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        ages = [9, 10 ,11]
        graph.add_nodes(name=names, age=ages)
        nodes = self.session.query(fornax.model.Node).filter(fornax.model.Node.graph_id==0).all()
        nodes = sorted(nodes, key=lambda node: node.node_id)
        self.assertListEqual(names, [json.loads(node.meta)['name'] for node in nodes])
        self.assertListEqual(ages, [json.loads(node.meta)['age'] for node in nodes])
        
    def test_missing_attribute(self):
        """Null values for metadata must be explicit
        """
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        ages = [9, 10]
        self.assertRaises(TypeError, graph.add_nodes, name=names, age=ages)

    def test_assign_id(self):
        """assigning node id is forbidden
        """
        graph = fornax.GraphHandle.create()
        ids = range(3)
        self.assertRaises(ValueError, graph.add_nodes, id=ids)

    def test_no_metadata(self):
        """Nodes must have some metadata associated with them
        """
        graph = fornax.GraphHandle.create()
        self.assertRaises(ValueError, graph.add_nodes)

    def test_add_edges(self):
        """store metadata on edges
        """
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        ages = [9, 10 ,11]
        graph.add_nodes(name=names, age=ages)
        relationships = ['is_friend', 'is_foe']
        graph.add_edges([0, 0], [1, 2], relationship=relationships)
        edges = self.session.query(
            fornax.model.Edge
        ).filter(
            fornax.model.Edge.graph_id==graph.graph_id
        ).filter(
            fornax.model.Edge.start < fornax.model.Edge.end
        ).all()
        edges = sorted(edges, key=lambda edge: (edge.start, edge.end))
        self.assertListEqual(relationships, [json.loads(edge.meta)['relationship'] for edge in edges])

    def test_add_edges_more_meta(self):
        """store multiple metadata on edges
        """
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        ages = [9, 10 ,11]
        graph.add_nodes(name=names, age=ages)
        relationships = ['is_friend', 'is_foe']
        types = [0 , 1]
        graph.add_edges([0, 0], [1, 2], relationship=relationships, type_=types)
        edges = self.session.query(
            fornax.model.Edge
        ).filter(
            fornax.model.Edge.graph_id==graph.graph_id
        ).filter(
            fornax.model.Edge.start < fornax.model.Edge.end
        ).all()
        edges = sorted(edges, key=lambda edge: (edge.start, edge.end))
        self.assertListEqual(relationships, [json.loads(edge.meta)['relationship'] for edge in edges])
        self.assertListEqual(types, [json.loads(edge.meta)['type'] for edge in edges])

    def test_simple_graph(self):
        """Test for a simple graph.
        A simple graph is a graph with no loops.
        A loop is an edge that connects a vertex to itself 
        """
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        ages = [9, 10 ,11]
        graph.add_nodes(name=names, age=ages)
        self.assertRaises(ValueError, graph.add_edges, [1, 0], [1, 2], relationship=['is_friend', 'is_foe'])

    def test_bad_edge_offset(self):
        """Edges a specicified by integer offsetse into the list of nodes
        """
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        ages = [9, 10 ,11]
        graph.add_nodes(name=names, age=ages)
        self.assertRaises(ValueError, graph.add_edges, ['adam', 'adam'], ['ben', 'chris'], relationship=['is_friend', 'is_foe'])


class TestQuery(TestCaseDB):

    @classmethod
    def setUp(self):
        """trick fornax into using the test database setup
        """
        super().setUp(self)
        fornax.api.Session = lambda: Session(self._connection)
    
    def test_init_query_raises(self):
        self.assertRaises(ValueError, fornax.QueryHandle, 0)

    def test_init_read_raises(self):
        self.assertRaises(ValueError, fornax.QueryHandle.read, 0)

    def test_create(self):
        query_graphs = [fornax.GraphHandle.create() for _ in range(3)]
        target_graphs = [fornax.GraphHandle.create() for _ in range(3)]
        queries = [fornax.QueryHandle.create(q, t) for q, t in zip(query_graphs, target_graphs)]
        self.assertEqual([q.query_id for q in queries], [0, 1, 2])
    
    def test_create_query_target(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        query_db = self.session.query(fornax.model.Query).filter(fornax.model.Query.query_id==query.query_id).first()
        self.assertEqual(query_db.start_graph_id, query_graph.graph_id)
        self.assertEqual(query_db.end_graph_id, target_graph.graph_id)
    
    def test_read(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        q1 = fornax.QueryHandle.create(query_graph, target_graph)
        q2 = fornax.QueryHandle.read(q1.query_id)
        self.assertEqual(q1.query_id, q2.query_id)

    def test_delete(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        query_id = query.query_id
        query.delete()
        query_exists = self.session.query(fornax.model.Query).filter(fornax.model.Query.query_id==query_id).scalar()
        matches_exists = self.session.query(fornax.model.Match).filter(fornax.model.Match.query_id==query_id).scalar()
        self.assertFalse(query_exists)
        self.assertFalse(matches_exists)
    
    def test_get_query_graph(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        self.assertEqual(query.query_graph(), query_graph)

    def test_get_target_graph(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        self.assertEqual(query.target_graph(), target_graph)

    def test_query_nodes(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        q_uids = [0, 1, 2]
        t_uids = [3, 4, 5]
        query_graph.add_nodes(uid=q_uids)
        target_graph.add_nodes(uid=t_uids)
        query_nodes = query._query_nodes()
        self.assertListEqual(
            [fornax.QueryHandle.Node(i, {'uid':uid}) for i, uid in enumerate(q_uids)], 
            query_nodes
        )

    def test_query_edges(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        q_uids = [0, 1, 2]
        t_uids = [3, 4, 5]
        query_graph.add_nodes(uid=q_uids)
        query_graph.add_edges([0, 0], [1, 2], my_id=['a', 'b'])
        target_graph.add_nodes(uid=t_uids)
        query_edges = query._query_edges()
        self.assertListEqual(
            [fornax.QueryHandle.Edge(0, 1, {'my_id':'a'}), fornax.QueryHandle.Edge(0, 2, {'my_id':'b'})], 
            query_edges    
        )

    def test_target_nodes(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        q_uids = [0, 1, 2]
        t_uids = [3, 4, 5]
        query_graph.add_nodes(uid=q_uids)
        target_graph.add_nodes(uid=t_uids)
        target_nodes = query._target_nodes()
        self.assertListEqual(
            [fornax.QueryHandle.Node(i, {'uid':uid}) for i, uid in enumerate(t_uids)], 
            target_nodes
        )

    def test_add_matches(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        uids = [0, 1]
        query_graph.add_nodes(uid=range(3))
        target_graph.add_nodes(uid=range(3))
        sources = [0, 0]
        targets = [1, 2]
        weights = [1, 1]
        query.add_matches(sources, targets, weights, my_id=uids)
        matches = self.session.query(fornax.model.Match).filter(
            fornax.model.Match.query_id==query.query_id
        ).order_by(fornax.model.Match.end.asc())
        self.assertEqual(sources, [m.start for m in matches])
        self.assertEqual(targets, [m.end for m in matches])
        self.assertEqual(weights, [m.weight for m in matches])
        self.assertEqual(uids, [json.loads(m.meta)['my_id'] for m in matches])

    def test_execute_raises(self):
        query_graph, target_graph = fornax.GraphHandle.create(), fornax.GraphHandle.create()
        query = fornax.QueryHandle.create(query_graph, target_graph)
        self.assertRaises(ValueError, query.execute)



