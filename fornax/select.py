from typing import List
from sqlalchemy.orm import Query
from sqlalchemy.sql.expression import literal
from fornax.model import Node, Edge
import numpy as np
import itertools
import collections


class Table:
    """[summary]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self, field_names: List[str], items: list):
        """[summary]
        
        Arguments:
            field_names {List[str]} -- [description]
            items {list} -- [description]
        
        Returns:
            [type] -- [description]
        """
        self.Row = collections.namedtuple('Row', field_names)
        self.rows = [self.Row(*(field for field in item)) for item in items]

    def fields(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        return self.Row._fields

    def __getitem__(self, index: int) -> collections.namedtuple:
        """[summary]
        
        Arguments:
            index {int} -- [description]
        
        Returns:
            collections.namedtuple -- [description]
        """
        return self.rows[index]

    def __len__(self):
        return len(self.rows)

    def to_frame(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        return Frame(self.fields(), list(zip(*self.rows)))

    def join(predicate, other, suffixes=['left', 'right']):
    """[summary]
    
    Arguments:
        predicate {[type]} -- [description]
        tables {[type]} -- [description]
        suffixes {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    filtered_pairs = filter(predicate, itertools.product(self[:], other[:]))
    dictionaries =  map(to_dict, zip(*filtered_pairs))
    return dict(
        collections.ChainMap(
            *(
                # namespace the keys to provent collisions
                {'_'.join([k,key]):v for k,v in d.items()} 
                for key, d 
                in zip(suffixes, dictionaries)
            )
        )
    )


def get_candidate(distance: float, label: str) -> Query:
    """return a sqlalchemy query object to fuzzy 
    match a query node label to target node labels
    
    Arguments:
        distance {float} -- string matching distance
        label {str} -- a label to match
    
    Raises:
        ValueError -- if distance is not between zero and one
    
    Returns:
        Query -- query object to select a table of nodes
    """

    if not 0 <= distance < 1:
        raise ValueError("distances must be between zero and one")
    query = Query(Node)
    query = query.filter(Node.label.op('<->')(label) < distance)
    return query


def get_neighbours(query: Query) -> Query:
    """starting with a query that selects a table of nodes
    return a query that returns all of the neighbours
    of each node
    
    Arguments:
        query {Query} -- a query that selects a table a nodes
    
    Returns:
        Query -- a query selecting a table of nodes with their parent id
    """

    subquery = query.subquery()
    new_query = Query(
        [
            Node.id, 
            Node.label, 
            Node.type, 
            Edge.start.label('parent'),
            literal(1).label('distance')
        ]
    )
    new_query = new_query.join(Edge, Edge.end == Node.id)
    new_query = new_query.filter(Edge.start == subquery.c.id)
    return new_query


def to_dict(rows):
    """Creates a table like dictionary of values from a query result
    
    Arguments:
        rows {[type]} -- [description]
    
    Raises:
        ValueError -- if each row does not have the same set of labels
    
    Returns:
        dict -- a dict of numpy arrays where each key is a column label
    """

    if not len(rows):
        return dict()
    if not all(row.keys() == rows[0].keys() for row in rows):
        raise ValueError('inconsistent labels')
    labels = rows[0].keys()
    cols = tuple(np.array(col) for col in zip(*rows))
    return {label:col for label, col in zip(labels, cols)}
