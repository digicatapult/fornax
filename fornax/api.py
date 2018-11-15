import fornax.select
import fornax.opt
import sqlalchemy
import contextlib
import itertools
import collections
import json
import os
import sys
import hashlib

from typing import Iterable
from sqlalchemy import event
from sqlalchemy.engine import Engine
import fornax.model as model

#TODO: sqlalchemy database integrity exceptions are not caught by the API

# Set this environment variable to point towards another database
DB_URL = os.environ.get('FORNAX_DB_URL')
if DB_URL is None:
    DB_URL = 'sqlite://'

MAX_SIZE = sys.maxsize
SQLITE_MAX_SIZE = 2147483647
if DB_URL == 'sqlite://':
    MAX_SIZE = min(MAX_SIZE, SQLITE_MAX_SIZE)

ECHO = False
ENGINE = sqlalchemy.create_engine(DB_URL, echo=ECHO)
CONNECTION = ENGINE.connect() 
Session = sqlalchemy.orm.sessionmaker(bind=ENGINE)
fornax.model.Base.metadata.create_all(CONNECTION)


def hash_id(item):
    """An unsalted hash function with a range between 0 and MAX_SIZE
    
    Arguments:
        item -- key that can be converted to a string using `str`
    
    Returns:
        {int} -- hash in the range 0 -> MAX_SIZE
    """
    if isinstance(item, int):
        return item % MAX_SIZE
    else:
        return int(hashlib.sha256(str(item).encode('utf-8')).hexdigest(), 16) % MAX_SIZE


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
            node_id = int(node.node_id)
        except ValueError:
            raise ValueError('<Node(node_id={})>, node_id must be an integer'.format(node))
        if node_id > SQLITE_MAX_SIZE and DB_URL == 'sqlite://':
            raise ValueError('node id {} is too large'.format(node))
        yield node


def check_edges(edges):
    """ guard for inserting nodes edges """
    for edge in edges:
        try:
            start, end = int(edge.start), int(edge.end)
        except ValueError:
            raise ValueError('<Edge(start={}, end={})>, edge start and end must be integers'.format(edge.start, edge.end))
        if start == end:
            raise ValueError('<Edge(start={}, end={})>, edges must start and end on different nodes'.format(start, end))
        yield edge


def check_matches(matches):
    """ guard for inserted matches """
    for match in matches:
        try:
            start = int(match.start) 
        except ValueError:
            raise ValueError('<Match(start={}, end={}, weight={})>, match start must be an integer')
        try:
            end = int(match.end) 
        except ValueError:
            raise ValueError('<Match(start={}, end={}, weight={})>, match end must be an integer')
        try:
            weight = float(match.weight)
        except ValueError:
            raise ValueError('<Match(start={}, end={}, weight={})>, match weight must be a number')
        if not 0 < weight <= 1:
            raise ValueError('<Match(start={}, end={}, weight={})>, bounds error: 0 < weight <= 1')
        yield match


class NullValue:
    """A dummy class to represent a missing value.
    This class intentionally cannot be json serialised
    """

    def __init__(self):
        pass


class Node:
    """A representation of a node internal to QueryHandle
    
    Raises:
        ValueError -- Raised if node.type is not {'query', 'target'}
    
    Returns:
        Node -- A tuple of id, type and a dictionary of metadata
    """


    __slots__ = ['id', 'type', 'meta']
    
    def __init__(self, node_id: int, node_type:str,  meta: dict):
        """Create a new Node
        
        Arguments:
            node_id {int} -- A unique indentifier for a node within a graph
            node_type {str} -- Either `query` or `target`
            meta {dict} -- A json serialisable dictionary of meta data
        
        Raises:
            ValueError -- Raised if Node.type is not either `query` or `target`
        
        Returns:
            Node -- new Node instance
        """

        if node_type not in ('query', 'target'):
            raise ValueError('Nodes must be of type "query", "target", "match"')
        self.id = node_id
        self.type = node_type
        self.meta = meta

    def __eq__(self, other):
        return (self.id, self.type, self.meta) == (other.id, other.type, other.meta)
    
    def __repr__(self):
        return '<Node(id={}, type={}, meta={})>'.format(self.id, self.type, self.meta)

    def __lt__(self, other):
        return (self.type, self.id) < (other.type, other.id)

    def to_dict(self):
        """Return self as a json serialisable dictionary
        
        Returns:
            dict -- Dictionary of node_id, type, and metadata
            id is the hash of the node id and type so that nodes are unique
            to either the query graph or the target graph
        """
        return {**{'id': hash_id((self.id, self.type)), 'type': self.type}, **self.meta}


class Edge:
    """Representation of an edge internal to QueryHandle
    
    Raises:
        ValueError -- Raised if type is not `query`, `target` or `match`
    
    Returns:
        Edge -- new edge object
    """

    __slots__ = ['start', 'end', 'type', 'meta', 'weight']

    def __init__(self, start:int, end:int, edge_type:str, meta:dict, weight=1.):
        """Create a new Edge
        
        Arguments:
            start {int} -- id of start node
            end {int} -- id of end node
            edge_type {str} -- either query, target or match
            meta {dict} -- dictionary of edge metadata
        
        Keyword Arguments:
            weight {float} -- An edge weight between 0 and 1 (default: {1.})
        
        Raises:
            ValueError -- Raised if type is not `query`, `target` or `match`
        """

        if edge_type not in ('query', 'target', 'match'):
            raise ValueError('Edges must be of type "query", "target", "match"')
        self.start = start
        self.end = end
        self.type = edge_type
        self.meta = meta
        self.weight = weight

    def __eq__(self, other):
        return (self.type, self.start, self.end, self.meta) == (other.type, other.start, other.end, other.meta)
    
    def __lt__(self, other):
        return (self.type, self.start, self.end) < (other.type, other.start, other.end) 
    
    def __repr__(self):
        return '<Edge(start={}, end={}, type={}, meta={})>'.format(
            self.start, self.end, self.type, self.meta
        )
    
    def to_dict(self):
        """Return self as a json serialisable dictionary
        
        Returns:
            dict -- Dictionary of start, end, type, metadata and weight
            start and end are the hash of the edge start/end and type so that nodes are unique
            to either the query graph or the target graph
        """
        if self.type == 'query' or self.type == 'target':
            start, end = hash_id((self.start, self.type)), hash_id((self.end, self.type))
        elif self.type == 'match':
            start, end = hash_id((self.start, 'query')), hash_id((self.end, 'target'))
        return {
            **{'source': start, 'target': end, 'type': self.type, 'weight': self.weight},
            **self.meta
        }

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
    
    def __eq__(self, other):
        return self.graph_id == other.graph_id

    @property
    def graph_id(self):
        return self._graph_id

    @classmethod
    def create(cls): 
        """Create a new graph
        
        Returns:
            GraphHandle -- a new graph handle with no nodes or edges
        """

        with session_scope() as session:
        
            query = session.query(sqlalchemy.func.max(model.Graph.graph_id)).first()
            graph_id = query[0]
            
            if graph_id is None:
                graph_id = 0
            else:
                graph_id += 1
            session.add(model.Graph(graph_id=graph_id))
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
            session.query(model.Graph).filter(model.Graph.graph_id==self._graph_id).delete()
            session.query(model.Edge).filter(model.Edge.graph_id==self._graph_id).delete()
            session.query(model.Node).filter(model.Node.graph_id==self._graph_id).delete()

    def check_exists(self):
        """ raise an exception if the graph handle is pointing to a graph that no longer exists """
        with session_scope() as session:
            exists = session.query(sqlalchemy.exists().where(model.Graph.graph_id==self._graph_id)).scalar()
        if not exists:
            raise ValueError('cannot read graph with graph id: {}'.format(self._graph_id))

    def add_nodes(self, **kwargs):
        """append nodes onto the graph
        
        Arguments:
            kwargs {Iterable} -- properties of the node are provided by iterables named using keyword args
            Use id_src to give a unique id to each node, if not provided a range index will be used by default
            id_src can be any type convertable to a string
            All iterables must be the same length.
            At least one keyword arg must be provided.
        """
        keys = kwargs.keys()
        if not len(keys):
            raise ValueError('add_nodes requires at least one keyword argument')
        if 'id' in keys:
            raise(ValueError('id is a reserved node attribute which cannot be assigned'))
        if 'id_src' in keys:
            id_src = kwargs['id_src']
            zipped = itertools.zip_longest(*kwargs.values(), fillvalue=NullValue())
            zipped = itertools.zip_longest(id_src, zipped, fillvalue=NullValue())
        else:
            zipped = enumerate(itertools.zip_longest(*kwargs.values(), fillvalue=NullValue()))

        nodes = (
            model.Node(
                node_id=hash_id(node_id),
                graph_id=self.graph_id, 
                meta=json.dumps({key: val for key, val in zip(keys, values)})
            )
            for node_id, values in zipped
        )
        nodes = check_nodes(nodes)
        with session_scope() as session:
            session.add_all(nodes)
            session.commit()

    def add_edges(self, sources, targets, **kwargs):
        """Add edges to the graph. 
        Edges are specified by using using the src_id of nodes.
        Use keyword args to attach json serialisable metadata to the edges.
        Edges may not start and end on the same node.
        Edges must be unique.
        Edges are undirected.
        
        Arguments:
            sources {Iterable} -- Iterable of src_ids
            targets {Iterable} -- Iterable of src_ids
            kwargs {Iterable} -- Iterable of json serialisable items
        """

        keys = kwargs.keys()
        if 'start' in keys:
            raise(ValueError('start is a reserved node attribute which cannot be assigned using kwargs'))
        if 'end' in keys:
            raise(ValueError('end is a reserved node attribute which cannot be assigned using kwargs'))
        if 'type' in keys:
            raise(ValueError('type is a reserved node attribute which cannot be assigned using kwargs'))
        if 'weight' in keys:
            raise(ValueError('weight is a reserved node attribute which cannot be assigned using kwargs'))
        hashed_sources, hashed_targets = map(hash_id, sources), map(hash_id, targets)
        zipped = itertools.zip_longest(hashed_sources, hashed_targets, *kwargs.values(), fillvalue=NullValue())
        edges = itertools.chain.from_iterable(
            (
                model.Edge(start=start, end=end, graph_id=self._graph_id,
                    meta=json.dumps({key: val for key, val in zip(keys, values)})
                ),
                model.Edge(start=end, end=start, graph_id=self._graph_id,
                    meta=json.dumps({key: val for key, val in zip(keys, values)})
                )
            )
            for start, end, *values in zipped
        )
        edges = check_edges(edges)
        with session_scope() as session:
            session.add_all(edges)
            session.commit()


class QueryHandle:
    """Represents a connection to a query in the database back end
    """

    def __init__(self, query_id: int):
        """Get a handle to a query in the database backend
        
        Arguments:
            query_id {int} -- An unique identifier for an existing query
        """

        self.query_id = query_id
        self._check_exists()
    
    def __eq__(self, other):
        return self.query_id == other.query_id

    def __len__(self):
        """Return the number of matches in the query
        
        Returns:
            {int} -- Count of matching edges
        """

        self._check_exists()
        with session_scope() as session:
            count = session.query(model.Match).filter(model.Match.query_id==self.query_id).count()
        return count
    
    def _check_exists(self):
        """Raise a value error is the query had been deleted
        
        Raises:
            ValueError -- Raised if the query had been deleted
        """

        with session_scope() as session:
            exists = session.query(model.Query).filter(model.Query.query_id==self.query_id).scalar()
        if not exists:
            raise ValueError('cannot read query with query id {}'.format(self.query_id))
    
    @classmethod
    def create(cls, query_graph:model.Query, target_graph:model.Query):
        """Create a new query
        
        Arguments:
            query_graph {model.Query} -- Query graph, this is the graph you're look for in the target graph
            target_graph {model.Query} -- Target graph, this is the graph being searched
        
        Returns:
            QueryHandle -- a new handle to a query
        """

        with session_scope() as session:
            query_id = session.query(sqlalchemy.func.max(model.Query.query_id)).first()[0]
            if query_id is None:
                query_id = 0
            else:
                query_id += 1
            new_query = model.Query(query_id=query_id, start_graph_id=query_graph.graph_id, end_graph_id=target_graph.graph_id)
            session.add(new_query)
        return QueryHandle(query_id)

    @classmethod
    def read(cls, query_id:int):
        """Get a handle to an existing query in the system
        
        Arguments:
            query_id {int} -- unique id of an existing query
        
        Returns:
            QueryHandle -- handle to an existing query
        """

        return QueryHandle(query_id)
    
    def delete(self):
        """Delete this query and any associated matches
        """

        self._check_exists()
        with session_scope() as session:
            session.query(model.Query).filter(model.Query.query_id==self.query_id).delete()
            session.query(model.Match).filter(model.Match.query_id==self.query_id).delete()
    
    def query_graph(self) -> GraphHandle:
        """Get a handle to the query graph used in this query
        
        Returns:
            GraphHandle -- query graph
        """

        self._check_exists()
        with session_scope() as session:
            start_graph = session.query(
                model.Graph
            ).join(
                model.Query, model.Graph.graph_id==model.Query.start_graph_id
            ).filter(model.Query.query_id==self.query_id).first()
            graph_id = start_graph.graph_id
        return GraphHandle(graph_id)
    
    def target_graph(self) -> GraphHandle:
        """Get a handle to the target graph used in this query
        
        Returns:
            GraphHandle -- target graph
        """

        self._check_exists()
        with session_scope() as session:
            end_graph = session.query(
                model.Graph
            ).join(
                model.Query, model.Graph.graph_id==model.Query.end_graph_id
            ).filter(model.Query.query_id==self.query_id).first()
            graph_id = end_graph.graph_id
        return GraphHandle(graph_id)

    def add_matches(self, sources:[int], targets:[int], weights:[int], **kwargs):
        """Add candidate correspondances between the query graph and the target graph
        
        Arguments:
            sources {[int]} -- integer offsets into the query graph
            targets {[int]} -- integer offsets into the target graph
            weights {[int]} -- corresondance scores between 0 and 1
        """

        self._check_exists()
        keys = kwargs.keys()
        if 'start' in keys:
            raise(ValueError('start is a reserved node attribute which cannot be assigned using kwargs'))
        if 'end' in keys:
            raise(ValueError('end is a reserved node attribute which cannot be assigned using kwargs'))
        if 'type' in keys:
            raise(ValueError('type is a reserved node attribute which cannot be assigned using kwargs'))
        if 'weight' in keys:
            raise(ValueError('weight is a reserved node attribute which cannot be assigned using kwargs'))
        hashed_sources, hashed_targetes = map(hash_id, sources), map(hash_id, targets)
        zipped = itertools.zip_longest(hashed_sources, hashed_targetes, weights, *kwargs.values(), fillvalue=NullValue())
        query_graph = self.query_graph()
        target_graph = self.target_graph()
        matches = (
            model.Match(
                start=start,
                end=end,
                start_graph_id=query_graph.graph_id,
                end_graph_id=target_graph.graph_id,
                query_id=self.query_id,
                weight=weight,
                meta=json.dumps({key: val for key, val in zip(keys, values)})
            )
            for start, end, weight, *values in zipped
        )
        matches = check_matches(matches)
        with session_scope() as session:
            session.add_all(matches)
            session.commit()

    def _query_nodes(self):
        with session_scope() as session:
            nodes = session.query(model.Node).join(
                model.Query, model.Node.graph_id==model.Query.start_graph_id
            ).filter(model.Query.query_id==self.query_id).all()
            nodes = [Node(n.node_id, 'query', json.loads(n.meta)) for n in nodes]
        return nodes

    def _query_edges(self):
        with session_scope() as session:
            edges = session.query(model.Edge).join(
                model.Query, model.Edge.graph_id == model.Query.start_graph_id
            ).filter(
                model.Query.query_id == self.query_id
            ).filter(
                model.Edge.start < model.Edge.end
            )
            edges = [Edge(e.start, e.end, 'query', json.loads(e.meta)) for e in edges]
        return edges
        
    def _target_nodes(self):
        with session_scope() as session:
            nodes = session.query(model.Node).join(
                model.Query, model.Node.graph_id==model.Query.end_graph_id
            ).filter(model.Query.query_id==self.query_id).all()
            nodes = [Node(n.node_id, 'target', json.loads(n.meta)) for n in nodes]
        return nodes
    
    def _target_edges(self, target_nodes, target_edges_arr):
        # only include target edges that are between the target nodes above
        target_ids = [n.id for n in target_nodes]
        is_between = lambda edge: edge.start in target_ids and edge.end in target_ids
        edges = (Edge(int(start), int(end), 'target', None) for start, end, d in target_edges_arr[['u', 'uu', 'dist_u']] if d < 2)
        edges = filter(is_between, edges)
        starts, ends = [], []
        for edge in edges:
            start, end = sorted((edge.start, edge.end))
            starts.append(start)
            ends.append(end)
        # starts, ends = zip(*((edge.start, edge.end) for edge in edges))
        with session_scope() as session:
            edges = session.query(model.Edge).join(
                model.Query, model.Query.end_graph_id == model.Edge.graph_id
            ).filter(
                model.Query.query_id == self.query_id
            ).filter(
                model.Edge.start.in_(starts)
            ).filter(
                model.Edge.end.in_(ends)
            ).filter(
                model.Edge.start < model.Edge.end
            ).distinct().all()
            edges = [Edge(e.start, e.end, 'target', json.loads(e.meta)) for e in edges]
        return edges

    def _optimise(self, hopping_distance, max_iters, offsets):
        with session_scope() as session:
            sql_query = fornax.select.join(self.query_id, h=hopping_distance, offsets=offsets)
            records = sql_query.with_session(session).all()

        inference_costs, subgraphs, iters, sz, target_edges_arr = fornax.opt.solve(
            records,
            hopping_distance=hopping_distance,
            max_iters=max_iters
        )  
        return inference_costs, subgraphs, iters, sz, target_edges_arr

    @classmethod
    def _get_scores(cls, inference_costs, query_nodes, subgraphs, sz):
        scores = []
        for subgraph in subgraphs:
            score = sum(inference_costs[k] for k in subgraph)
            score += sz - len(subgraph)
            score /= len(query_nodes)
            scores.append(score)
        return scores

    def execute(self, n=5, hopping_distance=2, max_iters=10):
        """Execute a query
        
        Keyword Arguments:
            n {int} -- number of subgraph matches to return (default: {5})
            hopping_distance {int} -- hyperparameter (default: {2})
            max_iters {int} -- maximum number of optimisation iterations before quitting (default: {10})
        
        Raises:
            ValueError -- raised if no matches are present in the query
        
        Returns:
            {dict} -- [description]
        """

        offsets = None # TODO: implement batching
        self._check_exists()
        if not len(self):
            raise ValueError('Cannot execute query with no matches')

        graphs = []
        query_nodes = sorted(self._query_nodes())
        target_nodes = sorted(self._target_nodes())
        # we will with get target edges from the optimiser since the optimiser knows this anyway
        target_edges = None
        query_edges = sorted(self._query_edges())
      
        inference_costs, subgraphs, iters, sz, target_edges_arr = self._optimise(hopping_distance, max_iters, offsets)
        target_edges = self._target_edges(target_nodes, target_edges_arr)
        target_edges = sorted(target_edges)

        scores = self._get_scores(inference_costs, query_nodes, subgraphs, sz)
        # sort graphs by score then deturministicly by hashing
        idxs = sorted(enumerate(scores), key=lambda x: (x[1], hash_id(tuple(subgraphs[x[0]]))))

        query_nodes_payload = [node.to_dict() for node in query_nodes]
        query_edges_payload = [edge.to_dict() for edge in query_edges]
        target_nodes_payload = [node.to_dict() for node in target_nodes]
        target_edges_payload = [edge.to_dict() for edge in target_edges]

        for i, score in idxs[:min(n, len(idxs))]:
            _, match_ends = zip(*subgraphs[i])
            matches =[
                Edge(s, e, 'match', {}, 1. - inference_costs[s,e]).to_dict() 
                for s, e in sorted(subgraphs[i])
            ]
            match_ends = set(hash_id((i, 'target')) for i in match_ends)
            nxt_graph = {
                'is_multigraph': False,
                'cost': score,
                'nodes': list(query_nodes_payload), # make a copy
                'links': matches + list(query_edges_payload)  # make a copy
            }
            nxt_graph['nodes'].extend([n for n in target_nodes_payload if n['id'] in match_ends])
            nxt_graph['links'].extend(
                [
                    e for e in target_edges_payload 
                    if e['source'] in match_ends and e['target'] in match_ends
                ]
            )
            graphs.append(nxt_graph)

        return {
            'graphs': graphs, 
            'iters': iters, 
            'hopping_distance':hopping_distance, 
            'max_iters': max_iters
        }
        

