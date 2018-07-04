import itertools
import unittest
import fornax.opt as opt
import fornax.model as model
import fornax.select as select
from test_base import TestCaseDB
from sqlalchemy.orm import Query, aliased

from sqlalchemy import or_, and_, literal
import numpy as np


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

    def test_opt(self):
        h = 2
        alpha = .3
        query = select.generate_query(h)
        rows = query.with_session(self.session).all()
        d, optimum_match = opt.optimise(h, alpha, rows)
        self.assertDictEqual(optimum_match, {(1,8):0, (2,9):0, (3,6):0, (4,10):0, (5,11):0,})

#     def test_query_match_nearest_neighbours_h_1(self):
#         query = select.match_nearest_neighbours(model.QueryNode, h=1)
#         query = query.with_session(self.session)
#         query = query.filter(model.Match.start == 2)
#         query = query.filter(model.Match.end == 2)
#         rows = query.all()
#         self.assertListEqual(
#             sorted(rows), 
#             sorted([(2, 2, 1, 1), (2, 2, 2, 0), (2, 2, 4, 1)])
#         )

#     def test_target_match_nearest_neighbours_h_1(self):
#         query = select.match_nearest_neighbours(model.TargetNode, h=1)
#         query = query.with_session(self.session)
#         query = query.filter(model.Match.start == 2)
#         query = query.filter(model.Match.end == 2)
#         rows = query.all()
#         self.assertListEqual(
#             sorted(rows), 
#             sorted([(2, 2, 1, 1), (2, 2, 2, 0)])
#         )

#     def test_query_match_nearest_neighbours_h_2(self):
#         query = select.match_nearest_neighbours(model.QueryNode, h=2)
#         query = query.with_session(self.session)
#         query = query.filter(model.Match.start == 2)
#         query = query.filter(model.Match.end == 2)
#         rows = query.all()
#         self.assertListEqual(
#             sorted(rows), 
#             sorted([(2, 2, 1, 1), (2, 2, 2, 0), (2, 2, 3, 2), (2, 2, 4, 1),  (2, 2, 5, 2)])
#         )

#     def test_target_match_nearest_neighbours_h_2(self):
#         query = select.match_nearest_neighbours(model.TargetNode, h=2)
#         query = query.with_session(self.session)
#         query = query.filter(model.Match.start == 2)
#         query = query.filter(model.Match.end == 2)
#         rows = query.all()
#         self.assertListEqual(
#             sorted(rows), 
#             sorted([(2, 2, 1, 1), (2, 2, 2, 0), (2, 2, 3, 2), (2, 2, 4, 2)])
#         )

#     def test_me(self):
#         self.assertTrue(True)
#         return
#         queries = [
#             self.session.query(
#                 model.Match.start.label('match_start'),
#                 model.Match.end.label('match_end'),
#                 model.QueryNode.id.label('query_node_id'), 
#                 literal(0).label('query_distance')
#             ).filter(model.QueryNode.id == model.Match.start),
#             self.session.query(
#                 model.Match.start.label('match_start'),
#                 model.Match.end.label('match_end'),
#                 model.QueryNode.id.label('query_node_id'), 
#                 literal(1).label('query_distance')
#             ).filter(
#                 and_(
#                     model.QueryNode.neighbours.any(model.QueryNode.id == model.Match.start),
#                     model.QueryNode.id != model.Match.start
#                 )
#             ),
#             self.session.query(
#                 model.Match.start.label('match_start'),
#                 model.Match.end.label('match_end'),
#                 model.QueryNode.id.label('query_node_id'), 
#                 literal(2).label('query_distance')
#             ).filter(
#                 and_(
#                     model.QueryNode.id != model.Match.start,
#                     model.QueryNode.neighbours.any(model.QueryNode.id != model.Match.start),
#                     model.QueryNode.neighbours.any(model.QueryNode.neighbours.any(model.QueryNode.id == model.Match.start))
#                 )
#             )
#         ]
        
#         targets = [
#             self.session.query(
#                 model.Match.start.label('match_start'),
#                 model.Match.end.label('match_end'),
#                 model.TargetNode.id.label('target_node_id'), 
#                 literal(0).label('target_distance')
#             ).filter(model.TargetNode.id == model.Match.end),
#             self.session.query(
#                 model.Match.start.label('match_start'),
#                 model.Match.end.label('match_end'),
#                 model.TargetNode.id.label('target_node_id'), 
#                 literal(1).label('target_distance')
#             ).filter(
#                 and_
#                 (
#                     model.TargetNode.neighbours.any(model.TargetNode.id == model.Match.end),
#                     model.TargetNode.id != model.Match.end
#                 )
#             ),
#             self.session.query(
#                 model.Match.start.label('match_start'),
#                 model.Match.end.label('match_end'),
#                 model.TargetNode.id.label('target_node_id'), 
#                 literal(2).label('target_distance')
#             ).filter(
#                 and_(
#                     model.TargetNode.id != model.Match.end,
#                     model.TargetNode.neighbours.any(model.TargetNode.id != model.Match.end),
#                     model.TargetNode.neighbours.any(model.TargetNode.neighbours.any(model.TargetNode.id == model.Match.end))
#                 )
#             )
#         ]

#         queries = queries[0].union(*queries[1:]).subquery()
#         targets = targets[0].union(*targets[1:]).subquery()
#         query = self.session.query(
#             queries.c.match_start,
#             queries.c.match_end,
#             queries.c.query_node_id, 
#             targets.c.target_node_id,
#             queries.c.query_distance,
#             targets.c.target_distance,
#             literal(0)
#         ).join(
#             targets, 
#             and_(
#                 queries.c.match_start==targets.c.match_start, 
#                 queries.c.match_end == targets.c.match_end
#             )            
#         )

#         columns = [
#             'match_start', 'match_end', 'query_node_id', 'target_node_id', 
#             'query_proximity', 'target_proximity', 'delta'
#         ]
#         dtypes = ['int32', 'int32', 'int32', 'int32', 'float32', 'float32', 'float32']
#         rows = np.array(query.all(), dtype=list(zip(columns, dtypes)))
#         query_idx = rows[['match_start', 'match_end', 'query_node_id']]
#         rows['query_proximity'] = opt.proximity(2, .3, rows['query_proximity'])
#         rows['target_proximity'] = opt.proximity(2, .3, rows['target_proximity'])
#         mask = np.concatenate((np.array([True]), np.logical_not(query_idx[1:] == query_idx[:-1])))

#         match_idx = rows[mask][['match_start', 'match_end']]
#         match_mask = np.concatenate((np.array([True]), np.logical_not(match_idx[1:] == match_idx[:-1])))
#         where = np.where(match_mask)[0][1:]
#         ranked = rows

#         for i in range(2):
#             ranked['delta'] += opt.delta_plus(rows['query_proximity'], rows['target_proximity'])
#             ranked = np.sort(ranked, order=['match_start', 'match_end', 'query_node_id', 'delta'], axis=0)
#             optimised = ranked[mask]
#             matches = np.split(optimised, where)
#             scores = {
#                 tuple(match[['match_start', 'match_end']][0]):np.sum(match['delta']) 
#                 for match in matches
#             }
#             ranked['delta'] = np.vectorize(lambda x: scores.get(tuple(x), 1))(ranked[['query_node_id', 'target_node_id']])
            
#         self.assertListEqual(sorted([]), sorted([4, 2]))
        

# if __name__ == '__main__':
#     unittest.main()
