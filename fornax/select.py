from typing import List
import numpy as np
import itertools
import collections
import typing


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
    """Column first indexed table (the transpose of a Table, like a pandas DataFrame)"""

    def __init__(self, field_names: List[str], columns: List[List]):
        """Construct a new Frame from a list of column names and each column of data as a list
            
            Arguments:
                field_names {List[str]} -- list of column names
                items {List[List]} -- each column
            
            Raises:
                ValueError -- all columns must have the same length
            
            Returns:
                Frame -- new Frame instance
        """

        if not all(len(a)==len(b) for a, b in itertools.product(columns, columns)):
            raise ValueError('all columns must have the same length')
    
        self._dict = {field:np.array(item) for field, item in zip(field_names, columns)}
        self._length = len(columns[0])
        self._fields = field_names

    def fields(self) -> List[str]:
        """Get column names
        
        Returns:
            List[str] -- list of column names
        """
        return self._fields

    def __getattr__(self, attr: str) -> np.ndarray:
        """get column with name attr
        
        Arguments:
            attr {str} -- column name
        
        Returns:
            np.ndarray -- column as numpy array
        """
        return self._dict[attr]

    def __len__(self) -> int:
        """Length of Frame (number of rows)
        
        Returns:
            int -- number of rows in the frame (length of each column)
        """
        return self._length

    def __eq__(self, other: 'Frame') -> bool:
        """Compare equal if all column names compare equal and all columns compare equal
        
        Arguments:
            other {Frame} -- Frame to compare to
        
        Returns:
            bool -- True if equal else False
        """
        if not set(self.fields()) == set(other.fields()):
            return False
        for key in other.fields():
            if not all(a == b for a, b in zip(getattr(self, key), getattr(other, key))):
                return False
        return True

    def to_table(self) -> Table:
        """Convert Frame to Table
        
        Returns:
            Table -- this Frame as a Table
        """
        attrs = self._dict.keys()
        return Table(attrs, zip(*(self[attr] for attr in attrs)))






