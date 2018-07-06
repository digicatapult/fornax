import unittest
import fornax.model as model
import fornax.select as select
from test_base import TestCaseDB
from sqlalchemy.orm import Query


class TestOpt(TestCaseDB):
    """Reproduce the scenario set out in figure 4 of the paper"""

    def setUp(self):
        super().setUp()  

        # Create the query graph from figure 4
        new_nodes = [
            model.QueryNode(id=id_+1)
            for id_, label in enumerate('abcde')
        ]

        start_finish =  [(1,2), (1,3), (2,4), (4,5)]

        new_edges = [
            model.QueryEdge(start=start, end=end) 
            for start, end in start_finish
        ]
        new_edges += [
            model.QueryEdge(start=end, end=start) 
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
            (6,8), (8,9), (8, 12), (9,10), (10,7), (10,11), (11,12), (11,13)
        ]
        new_nodes = [
            model.TargetNode(id=id_+1) 
            for id_, label in enumerate(labels)
        ]

        new_edges = [
            model.TargetEdge(start=start, end=end) 
            for start, end in start_finish
        ]
        new_edges += [
            model.TargetEdge(start=end, end=start) 
            for start, end in start_finish
        ]

        self.session.add_all(new_nodes)
        self.session.commit()
        self.session.add_all(new_edges)
        self.session.commit()

        self.session.add_all(
            [
                model.Match(start=1, end=1, weight=1),
                model.Match(start=1, end=4, weight=1),
                model.Match(start=1, end=8, weight=1),
                model.Match(start=2, end=2, weight=1),
                model.Match(start=2, end=5, weight=1),
                model.Match(start=2, end=9, weight=1),
                model.Match(start=3, end=3, weight=1),
                model.Match(start=3, end=6, weight=1),
                model.Match(start=3, end=12, weight=1),
                model.Match(start=3, end=13, weight=1),
                model.Match(start=4, end=7, weight=1),
                model.Match(start=4, end=10, weight=1),
                model.Match(start=5, end=11, weight=1),
            ]
        )
        self.session.commit()


    def test_query_match_nearest_neighbours_h_1(self):
        query = select.match_nearest_neighbours(model.QueryNode, h=1)
        query = query.with_session(self.session)
        rows = query.all()
        self.assertListEqual(
            sorted(filter(lambda x: x[0] == x[1] == 2, rows)), 
            sorted([(2, 2, 1, 1), (2, 2, 2, 0), (2, 2, 4, 1)])
        )

    def test_target_match_nearest_neighbours_h_1(self):
        query = select.match_nearest_neighbours(model.TargetNode, h=1)
        query = query.with_session(self.session)
        query = query.filter(model.Match.start == 2)
        query = query.filter(model.Match.end == 2)
        rows = query.all()
        self.assertListEqual(
            sorted(filter(lambda x: x[0] == x[1] == 2, rows)), 
            sorted([(2, 2, 1, 1), (2, 2, 2, 0)])
        )

    def test_query_match_nearest_neighbours_h_2(self):
        query = select.match_nearest_neighbours(model.QueryNode, h=2)
        query = query.with_session(self.session)
        rows = sorted(query.all())
        self.assertListEqual(
            sorted(filter(lambda x: x[0] == x[1] == 2, rows)), 
            sorted([(2, 2, 1, 1), (2, 2, 2, 0), (2, 2, 3, 2), (2, 2, 4, 1),  (2, 2, 5, 2)])
        )

    def test_target_match_nearest_neighbours_h_2(self):
        query = select.match_nearest_neighbours(model.TargetNode, h=2)
        query = query.with_session(self.session)
        query = query.filter(model.Match.start == 2)
        query = query.filter(model.Match.end == 2)
        rows = query.all()
        self.assertListEqual(
            sorted(filter(lambda x: x[0] == x[1] == 2, rows)), 
            sorted([(2, 2, 1, 1), (2, 2, 2, 0), (2, 2, 3, 2), (2, 2, 4, 2)])
        )

    def test_generate_query_h_1(self):
        query = select.generate_query(1)
        rows = query.with_session(self.session).all()
        self.assertListEqual(
            sorted(filter(lambda x: x[:2] == (2,5), rows)),
            sorted([
                (2, 5, 1, 4, 1, 1, 0),
                (2, 5, 4, 7, 1, 1, 0),
            ])
        )
            
    def test_generate_query_h_2(self):
        query = select.generate_query(2)
        rows = query.with_session(self.session).all()
        self.assertListEqual(
            sorted(filter(lambda x: x[:2] == (2,5), rows)),
            sorted([
                (2, 5, 1, 1, 1, 2, 0),
                (2, 5, 1, 4, 1, 1, 0),
                (2, 5, 3, 3, 2, 2, 0),
                (2, 5, 3, 6, 2, 2, 0),
                (2, 5, 4, 7, 1, 1, 0),
                (2, 5, 4, 10, 1, 2, 0),
            ])
        )
    
if __name__ == '__main__':
    unittest.main()
