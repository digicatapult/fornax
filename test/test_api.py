import unittest
import json
import fornax.api
import fornax.model
from test_base import TestCaseDB
from sqlalchemy.orm.session import Session


class TestGraph(TestCaseDB):

    @classmethod
    def setUp(self):
        # trick fornax into using the test database setup
        super().setUp(self)
        fornax.api.Session = lambda: Session(self._connection)

    def test_init_raises(self):
        """ raise an ValueError if a hadle to a graph is constructed that does not exist """
        self.assertRaises(ValueError, fornax.GraphHandle, 0)
        self.assertRaises(ValueError, fornax.GraphHandle.read, 0)

    def test_create(self):
        graph = fornax.GraphHandle.create()
        self.assertEqual(graph.graph_id, 0)

    def test_create_two(self):
        _ = fornax.api.GraphHandle.create()
        second = fornax.GraphHandle.create()
        self.assertEqual(second.graph_id, 1)

    def test_read(self):
        graph = fornax.api.GraphHandle.create()
        graph_id = graph.graph_id
        same_graph = fornax.GraphHandle.read(graph_id)
        self.assertEqual(same_graph.graph_id, graph_id)

    def test_delete(self):
        graph = fornax.GraphHandle.create()
        graph.delete()
        self.assertRaises(ValueError, fornax.api.GraphHandle.read, 0)

    def test_add_nodes(self):
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        graph.add_nodes(name=names)
        nodes = self.session.query(fornax.model.Node).filter(fornax.model.Node.graph_id==0).all()
        self.assertListEqual(names, [json.loads(node.meta)['name'] for node in nodes])

    def test_add_nodes_more_meta(self):
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        ages = [9, 10 ,11]
        graph.add_nodes(name=names, age=ages)
        nodes = self.session.query(fornax.model.Node).filter(fornax.model.Node.graph_id==0).all()
        self.assertListEqual(names, [json.loads(node.meta)['name'] for node in nodes])
        self.assertListEqual(ages, [json.loads(node.meta)['age'] for node in nodes])
        
    def test_missing_attribute(self):
        graph = fornax.GraphHandle.create()
        names = ['adam', 'ben', 'chris']
        ages = [9, 10]
        self.assertRaises(TypeError, graph.add_nodes, name=names, age=ages)

    def test_assign_id(self):
        graph = fornax.GraphHandle.create()
        ids = range(3)
        self.assertRaises(ValueError, graph.add_nodes, id=ids)