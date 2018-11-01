import fornax.select
import fornax.opt
import sqlalchemy
import contextlib
import itertools
import json
import os

from typing import Iterable
from sqlalchemy import event
from sqlalchemy.engine import Engine
import fornax.model as model

# Set this environment variable to point towards another database
DB_URL = os.environ.get('FORNAX_DB_URL')
if DB_URL is None:
    DB_URL = 'sqlite://'

ECHO = False
ENGINE = sqlalchemy.create_engine(DB_URL, echo=ECHO)
CONNECTION = ENGINE.connect() 
Session = sqlalchemy.orm.sessionmaker(bind=ENGINE)
fornax.model.Base.metadata.create_all(CONNECTION)


# enforce foreign key constrains in SQLite
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
    """ guard for inserting nodes """
    for node in nodes:
        try:
            node = int(node)
        except ValueError:
            raise ValueError('<Node(node_id={})>, node_id must be an integer'.format(node))
        if node > 2147483647 and DB_URL == 'sqlite://':
            raise ValueError('node id {} is too large'.format(node))
        yield node


def check_edges(edges):
    """ guard for inserting nodes edges """
    for start, end in edges:
        try:
            start, end = int(start), int(end)
        except ValueError:
            raise ValueError('<Edge(start={}, end={})>, edge start and end must be integers'.format(start, end))
        if start == end:
            raise ValueError('<Edge(start={}, end={})>, edges must start and end on different nodes'.format(start, end))
        yield start, end


def check_matches(matches):
    """ guard for inserted matches """
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


class GraphHandle:
    """ Represents a connection to a graph in the database backend """

    def __init__(self, graph_id: int):
        """Get a handle to the graph with id `graph_id` in the database backend
        
        Arguments:
            graph_id {int} -- unique id for an existing graph
        """

        self._graph_id = graph_id
        self.check_exists()
    
    def __len__(self):
        """Return the number of nodes in the graph
        
        Returns:
            int -- node count
        """

        with session_scope() as session:
            count = session.query(model.Node).filter(model.Node.graph_id==self._graph_id).count()
        return count

    def __repr__(self):
        return '<GraphHandle(graph_id={})>'.format(self._graph_id)
        
    def nodes(self):
        """ Yield each node id in the graph"""

        self.check_exists()
        with session_scope() as session:
            query = session.query(model.Node.node_id).filter(model.Node.graph_id==self._graph_id)
            chained = itertools.chain.from_iterable(query)
            for node_id in chained:
                yield node_id
    
    def edges(self):
        """ Yield each edge in the graph as a tuple (start, end) """
        with session_scope() as session:
            self.check_exists()
            query = session.query(model.Edge).filter(model.Edge.graph_id==self._graph_id)
            query = query.filter(model.Edge.start < model.Edge.end)
            for edge in query:
                yield (edge.start, edge.end)

    @property
    def graph_id(self):
        return self._graph_id

    @classmethod
    def create(cls, nodes:Iterable, edges:Iterable, metadata=None): 
        """Create a new graph and return a handle to it.
        
        Arguments:
            nodes {Iterable} -- Iterable of unique integer node identifiers
            edges {Iterable} -- Pairs of unique integer node identifiers being joined by an undirected edge
        
        Keyword Arguments:
            metadata {dict} -- A dictionary of metadata about each node (must be serialisable to json) (default: {None})
        
        Returns:
            GraphHandle -- Handle to a new graph
        """

        with session_scope() as session:
        
            query = session.query(sqlalchemy.func.max(model.Node.graph_id)).first()
            graph_id = query[0]
            
            if graph_id is None:
                graph_id = 0
            else:
                graph_id += 1
            
            if metadata is not None:
                session.add_all(
                    model.Node(node_id=node_id, graph_id=graph_id, meta=json.dumps(meta)) 
                    for node_id, meta in zip(check_nodes(nodes), metadata)
                )
            else:
                session.add_all(
                    model.Node(node_id=node_id, graph_id=graph_id) 
                    for node_id in check_nodes(nodes)
                )
            session.commit()

            session.add_all(
                itertools.chain.from_iterable(
                    (
                        model.Edge(start=start, end=end, graph_id=graph_id), 
                        model.Edge(start=end, end=start, graph_id=graph_id)
                    )
                    for start, end in check_edges(edges)
                )
            )
            session.commit()
        
        return GraphHandle(graph_id)

    @classmethod
    def read(cls, graph_id: int):
        """ return a handle to the graph with id graph_id"""
        return GraphHandle(graph_id)

    def delete(self):
        """ delete this graph from the database back end """
        self.check_exists()
        with session_scope() as session:
            session.query(model.Edge).filter(model.Edge.graph_id==self._graph_id).delete()
            session.query(model.Node).filter(model.Node.graph_id==self._graph_id).delete()

    def check_exists(self):
        """ raise an exception if the graph handle is pointing to a graph that no longer exists """
        with session_scope() as session:
            exists = session.query(sqlalchemy.exists().where(model.Node.graph_id==self._graph_id)).scalar()
        if not exists:
            raise ValueError('cannot read graph with graph id: {}'.format(self._graph_id))

class QueryHandle:
    """ Represents a handle to a query in the database back end.
    A query is a tuple of 3 objects:
        - a query graph
        - a target graph
        - a set of directed edges going from the query graph to the target graph
    """


    def __init__(self, query_id: int):
        """Create a handle to a query with uniuque integer id `query_id`
        
        Arguments:
            query_id {int} -- a uniuque integer id number for a query
        """
        self._query_id = query_id
        self.check_exists()

    @property
    def query_id(self):
        return self._query_id

    @classmethod
    def create(cls, start_graph: GraphHandle, end_graph: GraphHandle, matches: (int, int)):
        """Create a new query
        
        Arguments:
            start_graph {GraphHandle} -- A query graph
            end_graph {GraphHandle} -- A target graph
            matches {(int, int)} -- a set of directed edges going from the query graph to the target graph
        
        Returns:
            QueryHandle -- a new query handle
        """

        with session_scope() as session:
            query = session.query(sqlalchemy.func.max(model.Query.query_id)).first()
            query_id = query[0]
            
            if query_id is None:
                query_id = 0
            else:
                query_id += 1 

            new_query = model.Query(
                query_id=query_id, start_graph_id=start_graph.graph_id, end_graph_id=end_graph.graph_id
            )
            session.add(new_query)
            session.commit()

            session.add_all(
                [
                    model.Match(
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
        
        return QueryHandle(query_id)

    @classmethod
    def read(cls, query_id):
        """ create a new handle to the query with query_id `query_id`"""
        return GraphHandle(query_id)

    def delete(self):
        """ delete this query"""
        self.check_exists()
        with session_scope() as session:
            session.query(model.Match).filter(model.Match.query_id==self._query_id).delete()

    def check_exists(self):
        """ raise an exception if this query no longer exists """
        with session_scope() as session:
            exists = session.query(sqlalchemy.exists().where(model.Match.query_id==self._query_id)).scalar()
        if not exists:
            raise ValueError('cannot read query with graph id: {}'.format(self._query_id))

    def __repr__(self):
        return '<QueryHandle(query_id={})>'.format(self._query_id)

    def execute(self, hopping_distance=2, max_iters=10, n=5, edges=False):
        """Find the `n` best subgraphs in the target graph that match the query graph
        
        Keyword Arguments:
            hopping_distance {int} -- Hyperparameter representing a number of hops on a graph (default: {2})
            max_iters {int} -- maximum number of iterations before the optimiser quits (default: {10})
            n {int} -- number of matches to return (default: {5})
            edges {bool} -- include diagnostic information about edges (default: {False})
        
        Returns:
            dict -- a query result which can be serialised as JSON
        """

        self.check_exists()
        #TODO: support offsets
        offsets = None

        # optimisation
        with session_scope() as session:
            sql_query = fornax.select.join(self._query_id, h=hopping_distance, offsets=offsets)
            records = sql_query.with_session(session).all()

        inference_costs, subgraphs, iters, sz, target_edges_arr = fornax.opt.solve(
            records, 
            hopping_distance=hopping_distance, 
            max_iters=max_iters
        )

        # query node meta data
        with session_scope() as session:
            sql_query = session.query(model.Node)
            sql_query = sql_query.join(model.Query, model.Node.graph_id == model.Query.start_graph_id)
            sql_query = sql_query.filter(model.Query.query_id==self._query_id)
            query_nodes = [
                {'id': node.node_id, 'meta':json.loads(node.meta)} 
                if node.meta is not None
                else {'id': node.node_id}
                for node in sql_query.all()
            ]

        # query edge meta data
        with session_scope() as session:
            query_edges = None
            if edges:
                sql_query = session.query(model.Edge)
                sql_query = sql_query.join(model.Query, model.Edge.graph_id == model.Query.start_graph_id)
                sql_query = sql_query.filter(model.Query.query_id==self._query_id)
                sql_query = sql_query.filter(model.Edge.start < model.Edge.end)
                query_edges = [(edge.start, edge.end, 1) for edge in sql_query.all()]

        # generate json for top k-matches
        scores = []
        for subgraph in subgraphs:
            score = sum(inference_costs[k] for k in subgraph)
            score += sz - len(subgraph)
            score /= len(query_nodes)
            scores.append(score)
        
        idx = sorted(enumerate(scores), key=lambda x: x[1])
        payload = {
            'iterations': iters, 
            'subgraph_matches': [],
            'query_nodes': query_nodes,
        }

        for i, score in idx[:min(n, len(idx))]:

            subgraph = [
                {
                    'query_node_offset': int(a), 
                    'target_node_offset': int(b)
                } 
                for a, b in subgraphs[i]
            ]
            payload['subgraph_matches'].append(
                {
                    'subgraph_match': subgraph, 
                    'total_score': score,
                    'individual_scores': [
                        float(inference_costs[match['query_node_offset'], match['target_node_offset']])
                        for match in subgraph
                    ]
                } 
            )

        
        # only include data about target nodes that appear in one of the matches
        target_nodes = {match['target_node_offset'] for result in payload['subgraph_matches'] for match in result['subgraph_match']}
        
        # only include target edges that are between the target nodes above
        between_target_nodes = lambda x: x[0] in target_nodes and x[1] in target_nodes
        target_edges = [(int(start), int(end), int(d)) for start, end, d in target_edges_arr[['u', 'uu', 'dist_u']]]
        target_edges = list(filter(between_target_nodes, target_edges))
        target_nodes = list(target_nodes)

        # get the mata data for target nodes
        with session_scope() as session:
            sql_query = session.query(model.Node.meta)
            sql_query = sql_query.filter(model.Node.node_id.in_(target_nodes))
            sql_query = sql_query.order_by(model.Node.node_id.desc())
            meta_data = sql_query.all()
        target_nodes = sorted(target_nodes, reverse=True)
        payload['target_nodes'] = [{'id': item} for item in target_nodes]
        for target_node, meta in zip(payload['target_nodes'], meta_data):
            if meta[0] is not None:
                target_node['meta'] = json.loads(meta[0])

        # convert edge node ids into offsets
        query_node_lookup = {obj['id']:i for i, obj in enumerate(list(query_nodes))}
        target_node_lookup = {obj:i for i, obj in enumerate(list(target_nodes))}

        for result in payload['subgraph_matches']:
            result['subgraph_match'] = [
                {
                    'query_node_offset': query_node_lookup[item['query_node_offset']], 
                    'target_node_offset': target_node_lookup[item['target_node_offset']]
                } 
                for item in result['subgraph_match']
            ]

        if edges:

            payload['target_edges'] = [
                {
                    'start_offset': target_node_lookup[start],
                    'end_offset': target_node_lookup[end],
                }
                for start, end, d in target_edges
            ]

            payload['query_edges'] = [
                {
                    'start_offset': query_node_lookup[start],
                    'end_offset': query_node_lookup[end],
                }
                for start, end, d in query_edges
            ]

        return payload
