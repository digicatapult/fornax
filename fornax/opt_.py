import numpy as np
from itertools import starmap
from typing import List


HOPPING_DISTANCE = 2
ALPHA = .3
LAMBDA = .3
MAX_ITERS = 10

def _proximity(h: float, alpha: float, distances: np.ndarray) -> np.ndarray:
    """Calculates the proximity factor P for an array of distances.
    Implements equation 1 in the paper
    
    Arguments:
        h {float} -- max hopping distance
        alpha {float} -- propagation factor
        distances {np.array} -- array of hopping distances
    
    Raises:
        ValueError -- if hopping distance is less than zero
        ValueError -- if propagation factor is not between zero and one
    
    Returns:
        np.array -- an array of proximiy values
    """

    if h < 0:
        raise ValueError('hopping distance h cannot be negative')
    if not 0 < alpha <= 1:
        raise ValueError('propagation factor alpha must be between 0 and 1')
    return np.multiply(
        np.less_equal(distances, h),
        np.power(alpha, distances)
    )


def _delta_plus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Comparator function. Equation 3 in the paper.
    
    Arguments:
        x {np.array} -- an array of floats
        y {np.array} -- an array of floats
    
    Returns
        np.array -- an array of floats
    """

    return np.multiply(
        np.greater(x, y),
        np.subtract(x, y)
    )


def group_by(columns, arr):
    """Split an array into n slices where 'columns'
    are all equal within each slice
    
    Arguments:
        columns {List[str]} -- a list of column names
        arr {np.array} -- a numpy structured array
    
    Returns
        keys: np.array -- the column values uniquly identifying each group
        groups: List[np.array] -- a list of numpy arrays
    """
    
    if not len(columns):
        raise ValueError("group_by requires a non empty list of column names")
    
    _, counts = np.unique(arr[columns], return_counts=True)
    indices = np.insert(np.cumsum(counts), 0, 0)
    split = np.split(arr, indices)
    filtered = list(filter(lambda x: len(x), split))
    keys = map(lambda x: x[columns][0].tolist(), filtered)
    stacked = np.vstack(v for v in keys)
    return stacked, filtered


def group_by_first(columns, arr):
    """
    Split an array into n slices where 'columns' all compare equal within each slide
    Take the first row of each slice
    Combine each of the rows into a single array through concatination
    
    Arguments:
        columns {[str]} -- a list of column names
        arr {[type]} -- a numpy structured array
    
    Returns:
        np.array - new concatinated array
    """
    _, counts = np.unique(arr[columns], return_counts=True)
    indices = np.cumsum(np.insert(counts, 0, 0))[:-1]
    return arr[indices]


class Base(np.recarray):

    """A Base class for subclassing numpy record arrays
    
    Returns:
        np.recarray -- A subclass of np.recarray
    """

    columns = []
    types = []

    def __new__(cls, *args, **kwargs):
        a = np.atleast_2d(np.array(*args, **kwargs))
        dtype = np.dtype(list(zip(cls.columns, cls.types)))
        r = np.recarray(shape=a.shape[0], dtype=dtype)
        for i, col in enumerate(cls.columns):
            setattr(r, col, a[:,i])
        return r.view(NeighbourHoodMatchingCosts)


class QueryResult(Base):

    """container for results returned by the database query"""

    columns = 'v u vv uu prox_v prox_u neighbours misses'.split()
    types = '<i8 <i8 <i8 <i8 <f8 <f8 <f8 <i8 <i8 <f8'.split()

    @property
    def v(self):
        """Get column v
        
        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'u')

    @property
    def u(self):
        """Get column u
        
        Returns:
            np.ndarray -- array of target node ids as integers
        """

        return getattr(super(), 'v')

    @property
    def vv(self):
        """Get column vv - written v prime (v') in the paper where v' is a query node within
        hopping distance h of query node v
        
        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'vv')

    @property
    def uu(self):
        """Get column uu - written u prime (u') in the paper where u' is a target node within
        hopping distance h of target node u
        
        Returns:
            np.ndarray -- array of target node ids as integers
        """

        return getattr(super(), 'uu')

    @property
    def prox_v(self):
        """The hopping distance between query node v and query node vv (v')
        
        Returns:
            np.ndarray -- array of hopping distances as integers
        """

        return getattr(super(), 'prox_v')
    
    @property
    def prox_u(self):
        """The hopping distance between target node u and target node uu (u')
        
        Returns:
            np.ndarray -- array of hopping distances as integers
        """

        return getattr(super(), 'prox_v')
    
    @property
    def neighbours(self):
        """The number of query nodes with hopping distance h of v

        Returns:
            np.ndarray -- array of integer counts
        """

        return getattr(super(), 'neighbours')

    @property
    def misses(self):
        """The number of query nodes v' within hopping distance h of v
        that do not match any nodes u' within hopping distance of u

        Returns:
            np.ndarray -- array of integer counts
        """

        return getattr(super(), 'misses')
    
    def __repr__(self):
        return 'QueryResult(records={}, dtypes={})'.format(
            [record for record in self], self.types
        )


class NeighbourHoodMatchingCosts(Base):

    """Represents a table of all valid neighbourhood matching costs"""

    columns = 'v u vv uu cost'.split()
    types = '<i8 <i8 <i8 <i8 <f8'.split()

    def __getitem__(self, indx):
        """Get the row representing the neighbourhood matching costs at index "indx"
        
        Arguments:
            indx {int, slice} -- index into the table
        
        Returns:
            NeighbourHoodMatchingCosts -- NeighbourHoodMatchingCosts sliced by indx
        """
        return super().__getitem__(indx)

    @property
    def v(self):
        """Get column v
        
        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'u')

    @property
    def u(self):
        """Get column u
        
        Returns:
            np.ndarray -- array of target node ids as integers
        """

        return getattr(super(), 'v')

    @property
    def vv(self):
        """Get column vv - written v prime (v') in the paper where v' is a query node within
        hopping distance h of query node v
        
        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'vv')

    @property
    def uu(self):
        """Get column uu - written u prime (u') in the paper where u' is a target node within
        hopping distance h of target node u
        
        Returns:
            np.ndarray -- array of target node ids as integers
        """

        return getattr(super(), 'uu')


    @property
    def cost(self):
        """Get column cost - all valid neighbourhood matching costs.

        Eq 2 in the paper - multiplied by 1 - lambda
        
        Returns:
            np.ndarray -- array of costs and floats
        """

        return getattr(super(), 'cost')

    def __repr__(self):
        return 'NeighbourHoodMatchingCosts(records={}, dtypes={})'.format(
            [record for record in self], self.types
        )


class PartialMatchingCosts(Base):

    """ A table representing all valid partial matching costs """

    columns = 'v u vv cost'.split()
    types = '<i8 <i8 <i8 <f8'.split()

    @property
    def v(self):
        """Get column v
        
        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'v')

    @property
    def u(self):
        """Get column u
        
        Returns:
            np.ndarray -- array of target node ids as integers
        """

        return getattr(super(), 'u')

    @property
    def vv(self):
        """Get column vv - written v prime (v') in the paper where v' is a query node within
        hopping distance h of query node v
        
        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'uu')

    @property
    def cost(self):
        """Get column cost - all valid partial matching costs.

        Eq 13 in the paper (W) - but with beta multiplied by a factor of 1 - lambda
        
        Returns:
            np.ndarray -- array of costs as floats
        """

        return getattr(super(), 'cost')

    def __repr__(self):
        return 'PartialMatchingCosts(records={}, dtypes={})'.format(
            [record for record in self], self.types
        )


class InferenceCost(Base):

    """ A table representing all valid inference costs between query node
    u and target node v"""

    columns = 'v u cost'.split()
    types = '<i8 <i8 <f8'.split()

    @property
    def v(self):
        """Get column v
        
        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'v')

    @property
    def u(self):
        """Get column u
        
        Returns:
            np.ndarray -- array of target node ids as integers
        """

        return getattr(super(), 'u')
    
    @property
    def cost(self):
        """Get column cost - all valid inference costs for query node v and target node u.

        Eq 14 in the paper (U)
        
        Returns:
            np.ndarray -- array of costs as floats
        """

        return getattr(super(), 'cost')

    def __repr__(self):
        return 'InferenceCost(records={}, dtypes={})'.format(
            [record for record in self], self.types
        )


class OptimalMatch(Base):

    """Table representing the cost of the optimal match for query node v going to u"""

    columns = 'v u cost'.split()
    types = '<i8 <i8 <f8'.split()

    @property
    def v(self):
        """Get column v
        
        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'v')

    @property
    def u(self):
        """Get column u
        
        Returns:
            np.ndarray -- array of target node ids as integers
        """

        return getattr(super(), 'u')

    @property
    def cost(self):
        """Get column cost - the optimal matching cost for u going to v.

        Eq 10 in the paper (O)
        
        Returns:
            np.ndarray -- array of costs as floats
        """

        return getattr(super(), 'cost')