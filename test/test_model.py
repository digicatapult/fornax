import unittest
import fornax.model as model
from sqlalchemy.sql import func
from test_base import TestCaseDB


class TestNode(TestCaseDB):

    def test_node_round_trip(self):
        """ node round trip """
        new_node_type = model.NodeType(id=0, description="A node type for testing")
        new_node = model.Node(label='Ed Jones', type=0)

        self.session.add(new_node_type)
        self.session.commit()
        self.session.add(new_node)
        self.session.commit()

        found = self.session.query(model.Node).first()
        self.assertIsNotNone(found)
        self.assertEqual(new_node.label, found.label)
        self.assertEqual(new_node.type, 0)

    def test_trgm(self):
        """ Assert that pg_trgm is enabled """
        new_node_type = model.NodeType(id=0, description="A node type for testing")
        self.session.add(new_node_type)
        self.session.commit()

        # Insert some labelled nodes
        labels = ['Matt', 'Dom', 'Callum', 'David', 'Anthony']
        for label in labels:
            new_node = model.Node(label=label, type=0)
            self.session.add(new_node)
            self.session.commit()

        # Find a node using a label with an edit distance of at least one
        # from any node label
        query = self.session.query(model.Node)
        query = query.order_by(model.Node.label.op('<->')('Calum'))
        found = query.first()

        self.assertIsNotNone(found)
        self.assertEqual(found.label, 'Callum')


class TestEdge(TestCaseDB):

    def test_edge_round_trip(self):
        """ edge round trip """
        new_edge_type = model.EdgeType(id=0, description="An edge type for testing")
        new_node_type = model.NodeType(id=0, description="A node type for testing")
        self.session.add(new_edge_type)
        self.session.add(new_node_type)
        self.session.commit()

        new_nodes = [
            model.Node(id=id_, label=label, type=type_) 
            for id_, label, type_ in [(0, "Greg", 0), (1, "Sue", 0)]
        ]

        for new_node in new_nodes:
            self.session.add(new_node)
        self.session.commit()
        
        new_edge = model.Edge(start=0, end=1, type=0, weight=1.)
        self.session.add(new_edge)
        self.session.commit()

        found = self.session.query(model.Edge).first()
        self.assertIsNotNone(found)
        self.assertEqual(new_edge.start, found.start)

        query = self.session.query(model.Node)
        query = query.join(model.Edge, model.Node.id==model.Edge.start)
        found = query.first()
        self.assertIsNotNone(found)
        self.assertEqual(found.id, 0)

        query = self.session.query(model.Node)
        query = query.join(model.Edge, model.Node.id==model.Edge.end)
        found = query.first()
        self.assertIsNotNone(found)
        self.assertEqual(found.id, 1)


if __name__ == '__main__':
    unittest.main()
