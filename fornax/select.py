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


class Frame:
    """[summary]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self, field_names: List[str], columns: List[collections.namedtuple]):
        """[summary]
        
        Arguments:
            field_names {List[str]} -- [description]
            items {List[collections.namedtuple]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        if not all(len(a)==len(b) for a, b in itertools.product(columns, columns)):
            raise ValueError('all columns must have the same length')
    
        self._dict = {field:np.array(item) for field, item in zip(field_names, columns)}
        self._length = len(columns[0])
        self._fields = field_names

    def fields(self):
        return self._fields

    def __getattr__(self, attr: str):
        """[summary]
        
        Arguments:
            attr {str} -- [description]
        
        Returns:
            [type] -- [description]
        """
        return self._dict[attr]

    def __len__(self):
        return self._length

    def __eq__(self, other):
        if not set(self.fields()) == set(other.fields()):
            return False
        for key in other.fields():
            if not all(a == b for a, b in zip(getattr(self, key), getattr(other, key))):
                return False
        return True

    def to_table(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        attrs = self._dict.keys()
        return Table(attrs, zip(*(self[attr] for attr in attrs)))


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
    query = Query(
        [
            Node.id.label('id'), 
            Node.label.label('label'), 
            Node.type.label('type'),  
            literal(label).label('search_label')
        ]
    )
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
            literal(1).label('distance'),
            subquery.c.search_label
        ]
    )
    new_query = new_query.join(Edge, Edge.end == Node.id)
    new_query = new_query.join(subquery, Edge.start == subquery.c.id)
    return new_query


def exact_match(node_type, label):
    """[summary]
    
    Arguments:
        node_type {[type]} -- [description]
        label {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    query = Query([Node, literal(label).label('search_label')])
    query = query.filter(Node.label == label)
    query = get_neighbours(query)
    query = query.filter(Node.type == node_type)
    return query


def same_match(args):
    """[summary]
    
    Arguments:
        args {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    return args[0].search_label == args[1].search_label


    
    
    Returns:
    """

