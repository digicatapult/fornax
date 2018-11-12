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

#TODO: sqlalchemy database integrity exceptions are not caught by the API

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
            node_id = int(node.node_id)
        except ValueError:
            raise ValueError('<Node(node_id={})>, node_id must be an integer'.format(node))
        if node_id > 2147483647 and DB_URL == 'sqlite://':
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


class NullValue:
    """A dummy class to represent a missing value.
    This class intentionally cannot be json serialised
    """

    def __init__(self):
        pass


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
            All iterables must be the same length.
            At least one keyword arg must be provided.
        """
        keys = kwargs.keys()
        if not len(keys):
            raise ValueError('add_nodes requires at least one keyword argument')
        if 'id' in keys:
            raise(ValueError('id is a reserved node attribute which cannot be assigned'))
        zipped = enumerate(itertools.zip_longest(*kwargs.values(), fillvalue=NullValue()))
        nodes = (
            model.Node(
                node_id=node_id, 
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
        Edges are specified by using integer offsets into the graph nodes in insertion order.
        Use keyword args to attach json serialisable metadata to the edges.
        Edges may not start and end on the same node.
        Edges must be unique.
        Edges are undirected.
        
        Arguments:
            sources {Iterable} -- Iterable of integers
            targets {Iterable} -- Iterable of integers
            kwargs {Iterable} -- Iterable of json serialisable items
        """

        keys = kwargs.keys()
        zipped = itertools.zip_longest(sources, targets, *kwargs.values(), fillvalue=NullValue())
        edges = (
            model.Edge(
                start=start,
                end=end,
                graph_id=self._graph_id,
                meta=json.dumps({key: val for key, val in zip(keys, values)})
            )
            for start, end, *values in zipped
        )
        edges = check_edges(edges)
        with session_scope() as session:
            session.add_all(edges)
            session.commit()
        