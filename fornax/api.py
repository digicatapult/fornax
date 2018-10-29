import fornax.select
import fornax.opt
import sqlalchemy
import contextlib
import itertools
import os
from typing import Iterable
from fornax.model import Node, Edge, Match

from sqlalchemy import event
from sqlalchemy.engine import Engine


DB_URL = os.environ.get('FORNAX_DB_URL')
if DB_URL is None:
    DB_URL = 'sqlite://'

ECHO = False
ENGINE = sqlalchemy.create_engine(DB_URL, echo=ECHO)
CONNECTION = ENGINE.connect() 
Session = sqlalchemy.orm.sessionmaker(bind=ENGINE)
fornax.model.Base.metadata.create_all(CONNECTION)

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

@contextlib.contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def check_nodes(nodes):

    for node in nodes:
        try:
            node = int(node)
        except ValueError:
            raise ValueError('<Node(node_id={})>, node_id must be an integer'.format(node))
        yield node


def check_edges(edges):

    for start, end in edges:
        try:
            start, end = int(start), int(end)
        except ValueError:
            raise ValueError('<Edge(start={}, end={})>, edge start and end must be integers'.format(start, end))
        if start == end:
            raise ValueError('<Edge(start={}, end={})>, edges must start and end on different nodes'.format(start, end))
        yield start, end


def check_matches(matches):
    for start, end, weight in matches:
        try:
            start = int(start) 
        except ValueError:
            raise ValueError('<Match(start={}, end={}, weight={})>, match start must be an integer')
        try:
            end = int(end) 
        except ValueError:
            raise ValueError('<Match(start={}, end={}, weight={})>, match end must be an integer')
        try:
            weight = float(weight)
        except ValueError:
            raise ValueError('<Match(start={}, end={}, weight={})>, match weight must be a number')
        if not 0 < weight <= 1:
            raise ValueError('<Match(start={}, end={}, weight={})>, bounds error: 0 < weight <= 1')
        yield start, end, weight


class Graph:


    def __init__(self, graph_id):
        self._graph_id = graph_id
        self.check_exists()
    
    def __len__(self):
        with session_scope() as session:
            count = session.query(Node).filter(Node.graph_id==self._graph_id).count()
        return count

    def __repr__(self):
        return '<GraphHandle(graph_id={})>'.format(self._graph_id)
        
    def nodes(self):
        self.check_exists()
        with session_scope() as session:
            query = session.query(Node.node_id).filter(Node.graph_id==self._graph_id)
            chained = itertools.chain.from_iterable(query)
            for node_id in chained:
                yield node_id
    
    def edges(self):
        with session_scope() as session:
            self.check_exists()
            query = session.query(Edge).filter(Edge.graph_id==self._graph_id)
            # only yield each edge once
            query = query.filter(Edge.start < Edge.end)
            for edge in query:
                yield (edge.start, edge.end)

    @property
    def graph_id(self):
        return self._graph_id

    @classmethod
    def create(cls, nodes:Iterable, edges:Iterable): 
        
        with session_scope() as session:
        
            query = session.query(sqlalchemy.func.max(Node.graph_id)).first()
            graph_id = query[0]
            
            if graph_id is None:
                graph_id = 0
            else:
                graph_id += 1
            
            session.add_all(Node(node_id=node_id, graph_id=graph_id) for node_id in check_nodes(nodes))
            session.commit()
            session.add_all(
                itertools.chain.from_iterable(
                    (
                        Edge(start=start, end=end, graph_id=graph_id), 
                        Edge(start=end, end=start, graph_id=graph_id)
                    )
                    for start, end in check_edges(edges)
                )
            )
            session.commit()
        
        return Graph(graph_id)

    @classmethod
    def read(cls, graph_id):
        return Graph(graph_id)

    def delete(self):
        self.check_exists()
        with session_scope() as session:
            session.query(Edge).filter(Edge.graph_id==self._graph_id).delete()
            session.query(Node).filter(Node.graph_id==self._graph_id).delete()

    def check_exists(self):
        with session_scope() as session:
            exists = session.query(sqlalchemy.exists().where(Node.graph_id==self._graph_id)).scalar()
        if not exists:
            raise ValueError('cannot read graph with graph id: {}'.format(self._graph_id))

class Query:

    def __init__(self, query_id):
        self._query_id = query_id
        self.check_exists()

    @property
    def query_id(self):
        return self._query_id

    @classmethod
    def create(cls, start_graph: Graph, end_graph: Graph, matches):

        with session_scope() as session:
            query = session.query(sqlalchemy.func.max(Match.query_id)).first()
            query_id = query[0]
            
            if query_id is None:
                query_id = 0
            else:
                query_id += 1 

            session.add_all(
                [
                    Match(
                        start=start, 
                        end=end, 
                        weight=weight, 
                        start_graph_id=start_graph.graph_id, 
                        end_graph_id=end_graph.graph_id, 
                        query_id=query_id
                    )
                    for start, end, weight in check_matches(matches)
                ]
            )
            session.commit()
        
        return Query(query_id)

    @classmethod
    def read(cls, query_id):
        return Graph(query_id)

    def delete(self):
        self.check_exists()
        with session_scope() as session:
            session.query(Match).filter(Match.query_id==self._query_id).delete()

    def check_exists(self):
        with session_scope() as session:
            exists = session.query(sqlalchemy.exists().where(Match.query_id==self._query_id)).scalar()
        if not exists:
            raise ValueError('cannot read query with graph id: {}'.format(self._query_id))

    def execute(self, hopping_distance=2, max_iters=10, n=5, edges=False):

        #TODO: support offsets
        offsets = None
        query = fornax.select.join(self._query_id, h=hopping_distance, offsets=offsets)

        with session_scope() as session:
            records = query.with_session(session).all()

        inference_costs, subgraphs, iters, sz = fornax.opt.solve(
            records, 
            hopping_distance=hopping_distance, 
            max_iters=max_iters
        )

        scores = []
        for subgraph in subgraphs:
            score = sum(inference_costs[k] for k in subgraph)
            score += sz - len(subgraph)
            scores.append(score)
        
        idx = sorted(enumerate(scores), key=lambda x: x[1])
        payload = []
        for i, score in idx[:min(n, len(idx))]:

            subgraph = subgraphs[i]
            query_edges = []
            target_edges = []

            if edges:
                #TODO: populate edges here, awating schema update
                pass
            
            payload.append(
                {
                    'graph': subgraph, 
                    'score': score,
                    'matches': {k:inference_costs[k] for k in subgraph},
                    'query_nodes': [k[0] for k in subgraph],
                    'query_edges': query_edges,
                    'target_nodes': [k[1] for k in subgraph],
                    'target_edges': target_edges
                } 
            )
        

        return payload
