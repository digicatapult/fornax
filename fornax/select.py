from typing import List
from sqlalchemy.orm import Query
from sqlalchemy.sql.expression import literal, alias
from fornax.model import Node, Edge
import numpy as np
import itertools
import collections


class Table:
    """An adaptor class abstrating a list of rows returned by a SQLAlchemy query"""

    def __init__(self, field_names: List[str], items: list):
        """Construct a Table from a list of column names and a set of rows (tuples)
        
        Arguments:
            field_names {List[str]} -- List of column names
            items {list} -- list of tuples, one for each row in the table
        
        Returns:
            Table -- a new table instance
        """
        self.Row = collections.namedtuple('Row', field_names)
        self.rows = [self.Row(*(field for field in item)) for item in items]

    def fields(self) -> List[str]:
        """get the column names of the table
        
        Returns:
            List[str] -- list of column names
        """
        return self.Row._fields

    def __getitem__(self, index: int) -> collections.namedtuple:
        """Get row i from a table
        
        Arguments:
            index {int} -- index of a row in this Table
        
        Returns:
            collections.namedtuple -- named tuple repreneting a row in a table with names self.fields()
        """
        return self.rows[index]

    def __len__(self) -> int:
        """number of rows in the table
        
        Returns:
            int -- number of rows in the table
        """

        return len(self.rows)

    def join(self, predicate, other: 'Table', suffixes=['_left', '_right']):
        """Equivalent to a SQL inner join between two tables
        
        Arguments:
            predicate {func} -- a joining condition for a pair of rows 
                                e.g. lambda pair: pair[0].id == ropairws[1].id
            other {Table} -- table to join to
        
        Keyword Arguments:
            suffixes {list} -- append these strings to the column names of the 
                               first and second table to avoid name conflicts 
                               (default: {['_left', '_right']})
        
        Raises:
            ValueError -- length of suffixes must be equal to 2
        
        Returns:
            Table -- a new table joined on predicate
        """

        if not len(suffixes) == 2:
            raise ValueError
        

        field_names = [s + suffixes[0] for s in self.fields()]
        field_names += [s + suffixes[1] for s in other.fields()]
        rows = (a + b for a, b in filter(predicate, itertools.product(self.rows, other.rows)))
        return Table(field_names, rows)
        

    def to_frame(self):
        """Convert the table to a Frame
        
        Returns:
            Frame -- a new Frame constructed from this table
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



