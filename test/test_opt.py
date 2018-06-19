import unittest
import fornax.opt as opt
import fornax.model as model
import fornax.select as select
from test_base import TestCaseDB
from sqlalchemy.orm import Query


class TestOpt(TestCaseDB):
    """Reproduce the scenario set out in figure 4 of the paper"""

    def setUp(self):
        super().setUp()

        new_node_types = [
            model.NodeType(id=0, description="query node type"),
            model.NodeType(id=1, description="target node type")
        ]

        new_edge_types = [
            model.EdgeType(id=0, description="query edge type"),
            model.EdgeType(id=1, description="target edge type")
        ]

        for node_type in new_node_types:
            self.session.add(node_type)

        for edge_type in new_edge_types:
            self.session.add(edge_type)

        self.session.commit()
    

        # Create the query graph from figure 4

        new_nodes = [
            model.Node(id=id_+1, label=label, type=0)
            for id_, label in enumerate('abcde')
        ]

        start_finish =  [(1,2), (1,3), (2,4), (4,5)]

        new_edges = [
            model.Edge(start=start, end=end, type=0,  weight=1.) 
            for start, end in start_finish
        ]
        new_edges += [
            model.Edge(start=end, end=start, type=0, weight=1.) 
            for start, end in start_finish
        ]

        self.session.add_all(new_nodes)
        self.session.commit()
        self.session.add_all(new_edges)
        self.session.commit()

        # Create the target graph from figure 4 (ids are offset by 100)
        labels = 'abcabcdabdecc'
        start_finish = [
            (1,2), (1,3), (1,4), (3,7), (4,5), (4,6), (5,7),           
            (6,8), (8,9), (8, 12), (9,10), (10,11), (11,12), (11,13)
        ]
        new_nodes = [
            model.Node(id=100 + id_+1, label=label, type=1)
            for id_, label in enumerate(labels)
        ]

        new_edges = [
            model.Edge(start=start+100, end=end+100, type=1,  weight=1.) 
            for start, end in start_finish
        ]
        new_edges += [
            model.Edge(start=end+100, end=start+100, type=1, weight=1.) 
            for start, end in start_finish
        ]

        self.session.add_all(new_nodes)
        self.session.commit()
        self.session.add_all(new_edges)
        self.session.commit()


    def test_me(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
