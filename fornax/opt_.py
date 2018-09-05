import numpy as np


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