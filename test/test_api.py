import unittest
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
        self.assertRaises(ValueError, fornax.api.GraphHandle, 0)
        self.assertRaises(ValueError, fornax.api.GraphHandle.read, 0)

    def test_create(self):
        graph = fornax.api.GraphHandle.create()
        self.assertEqual(graph.graph_id, 0)

    def test_create_two(self):
        _ = fornax.api.GraphHandle.create()
        second = fornax.api.GraphHandle.create()
        self.assertEqual(second.graph_id, 1)

    def test_read(self):
        graph = fornax.api.GraphHandle.create()
        graph_id = graph.graph_id
        same_graph = fornax.api.GraphHandle.read(graph_id)
        self.assertEqual(same_graph.graph_id, graph_id)

    def test_delete(self):
        graph = fornax.api.GraphHandle.create()
        graph.delete()
        self.assertRaises(ValueError, fornax.api.GraphHandle.read, 0)
        
        