import numpy as np
from itertools import starmap
from typing import List


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
    Split an array into n slices where 'columns'
    all compare equal within each slide
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
            setattr(r, col, a[:, i])
        return r.view(cls)


class QueryResult(Base):

    """Represents a query from the database as a numpy rec array"""

    columns = 'v u vv uu dist_v dist_u weight'.split()
    types = [np.int32, np.int32, np.int32, np.int32,
             np.float32, np.float32, np.float32]

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
        """Get column vv - written v prime (v') in the paper
        where v' is a query node within
        hopping distance h of query node v

        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'vv')

    @property
    def uu(self):
        """Get column uu - written u prime (u') in the paper
        where u' is a target node within
        hopping distance h of target node u

        values less than zero indicate that uu (u') has
        no corresponding matches to any node v'

        Returns:
            np.ndarray -- array of target node ids as integers
        """

        return getattr(super(), 'uu')

    @property
    def dist_v(self):
        """The hopping distance between query node v and query node vv (v')

        Returns:
            np.ndarray -- array of hopping distances as integers
        """

        return getattr(super(), 'dist_v')

    @property
    def dist_u(self):
        """The hopping distance between target node u and target node uu (u')

        Returns:
            np.ndarray -- array of hopping distances as integers
        """

        return getattr(super(), 'dist_u')

    @property
    def weight(self):
        """String matching score between uu (u') and vv (v')

        Returns:
            np.ndarray -- array of floating point weights
        """

        return getattr(super(), 'weight')

    def __repr__(self):
        return 'QueryResult(records={}, dtypes={})'.format(
            [record for record in self], self.types
        )


class NeighbourHoodMatchingCosts(Base):

    """Represents a table of all valid neighbourhood matching costs"""

    columns = 'v u vv uu cost'.split()
    types = [np.int64, np.int64, np.int64, np.int64, np.float32]

    def __getitem__(self, indx):
        """Get the row representing the neighbourhood matching costs at index "indx"

        Arguments:
            indx {int, slice} -- index into the table

        Returns:
            NeighbourHoodMatchingCosts -- NeighbourHoodMatchingCosts
            sliced by indx
        """
        return super().__getitem__(indx)

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
        """Get column vv - written v prime (v') in the paper
        where v' is a query node within
        hopping distance h of query node v

        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'vv')

    @property
    def uu(self):
        """Get column uu - written u prime (u') in the paper
        where u' is a target node within
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
    types = [np.int64, np.int64, np.int64, np.float32]

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
        """Get column vv - written v prime (v') in the paper
        where v' is a query node within
        hopping distance h of query node v

        Returns:
            np.ndarray -- array of query node ids as integers
        """

        return getattr(super(), 'uu')

    @property
    def cost(self):
        """Get column cost - all valid partial matching costs.

        Eq 13 in the paper (W) - but with beta multiplied by
        a factor of 1 - lambda

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
    types = [np.int64, np.int64, np.float32]

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
        """Get column cost - all valid inference costs for
        query node v and target node u.

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

    """Table representing the cost of the optimal
    match for query node v going to u"""

    columns = 'v u cost'.split()
    types = [np.int64, np.int64, np.float32]

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


# TODO: This seems a big ugly as a class and not consistent with the rest
# of the code
class Refiner:
    """Take each of the matches and recursivly find
    all of their neighbours via a greedy algorithm"""

    def __init__(
            self, neighbourhood_matching_costs: NeighbourHoodMatchingCosts):
        """Initialise a refiner using a Frame instance

        For each pair (v, u) find the lowest cost adjacent pair (vv, uu)

        Arguments:
            frame {[Frame]} -- A frame constructed
            records returned by the database
        """

        # group all of the pairs (v, u)
        matches = {tuple(k): v for k, v in zip(
            *group_by(['v', 'u'], neighbourhood_matching_costs))}

        # get the costs cost neighbouring pair (vv, uu)
        neighbours = {
            k: group_by_first(['v', 'u', 'vv'], v)[['vv', 'uu']].tolist()
            for k, v in matches.items()
        }

        self.neighbours = {
            k: [item for item in v if self.valid_neighbours(k, item)]
            for k, v in neighbours.items()
        }

    def __call__(self, seed, result):
        """Given a pair v, u,
        greedily find the lowest cost neighbours
        recursivly covering the whole graph
        without cyclic paths

        Arguments:
            seed {[int, int]} -- u, v pair

        Keyword Arguments:
            result {[int, int]} -- stores the result between
            recursive calls (default: {None})

        Returns:
            [type] -- List of query_node, target_node
            id pairs constituting a result
        """

        # To not add a new result if we've already visited this query node
        if seed[0] in set(query_node for query_node, target_node in result):
            return

        # Now do the same for all the neighbours of the new result
        result += [seed]
        for neighbour in self.neighbours[seed]:
            self(neighbour, result)
        return

    @staticmethod
    def valid_neighbours(first: tuple, second: tuple):
        """Function that governs a valid hop between nodes

        Arguments:
            first {int, int} -- source query_node, target_node id pair
            second {int, int} -- target query_node, target_node id pair

        Returns:
            Bool -- True is a valid transition
        """

        # transitions to self are not valid
        if first == second:
            return False
        # transitions to missing target nodes are not valid
        if any(u < 0 for u in second):
            return False
        return True


def _get_matching_costs(
        records: List[tuple], hopping_distance, lmbda=.3, alpha=.3, ):
    """Create a table of matching costs from a
    table of query results using equation 2
    Equivalent to the first term of equation 13

    Returns:
        NeighbourHoodMatchingCosts -- table of all valid matching costs
        QueryResult -- query result as a numpy rec
        array rather than a list of tuples
    """

    # convert NaN records into negative numbers so they can be stored as ints
    # using numpy
    query_result = QueryResult(
        [
            tuple(item if item is not None else -1 for item in tup)
            for tup in records
        ]
    )
    # label costs are weights in the databse
    query_result.weight = 1. - query_result.weight
    query_result = np.sort(query_result, order=[
                           'v', 'u', 'vv', 'uu', 'weight'])

    nan_idx = query_result.dist_u < 0
    dist_u = query_result.dist_u
    dist_u[nan_idx] = hopping_distance + 1

    # convert hopping distances into proximities (eq. 1)
    prox_v = _proximity(hopping_distance, alpha, query_result.dist_v)
    prox_u = _proximity(hopping_distance, alpha, dist_u)
    cost = _delta_plus(prox_v, prox_u)
    cost *= (1. - lmbda)

    arr = np.unique(
        np.rec.fromarrays(
            (query_result.v, query_result.vv, prox_v),
            dtype=[('v', int), ('vv', int), ('prox_v', float)]
        ),
        axis=0
    )

    vs, groups = group_by('v', arr)
    beta_ = {v[0]: sum(group['prox_v']) for v, group in zip(vs, groups)}
    beta = np.vectorize(lambda x: beta_[x])
    neighbourhood_matching_costs = NeighbourHoodMatchingCosts(
        np.array([
            query_result.v,
            query_result.u,
            query_result.vv,
            query_result.uu,
            cost
        ]).transpose()
    )
    return neighbourhood_matching_costs, query_result, beta


def _get_partial_inference_costs(
        neighbourhood_matching_costs: NeighbourHoodMatchingCosts,
        beta
) -> PartialMatchingCosts:
    """get the lowest cost neighbourhood matching cost for each partial match

    Arguments:
        neighbourhood_matching_costs {NeighbourHoodMatchingCosts} -- sorted
        neighbourhood_matching_costs

    Returns:
        PartialMatchingCosts
    """

    grouped = group_by_first(['v', 'u', 'vv'], neighbourhood_matching_costs)
    partial_matching_costs = PartialMatchingCosts(
        np.array([grouped.v, grouped.u, grouped.vv, grouped.cost]).transpose()
    )
    partial_matching_costs.cost /= beta(partial_matching_costs.v)
    return partial_matching_costs


def _get_inference_costs(
        partial_matching_costs: PartialMatchingCosts) -> InferenceCost:
    """sum partial matching cost for each query node target node pair (v, u)

    Arguments:
        partial_matching_costs {PartialMatchingCosts} -- sorted
        partial_matching_costs

    Returns:
        InferenceCost
    """

    # TODO: inference_costs include the label weight
    keys, groups = group_by(['v', 'u'], partial_matching_costs)
    summed = starmap(
        lambda key, group: (key[0], key[1], np.sum(group.cost) / len(group)),
        zip(keys, groups)
    )
    columns = np.vstack(summed)
    return InferenceCost(columns)


def _get_optimal_match(inference_costs: InferenceCost) -> OptimalMatch:
    """Get the lowest cost match for each query node v

    Arguments:
        inference_costs {InferenceCost} -- sorted inference costs

    Returns:
        OptimalMatch
    """

    return group_by_first('v', inference_costs)


def solve(records: List[tuple], max_iters=10, hopping_distance=2):
    """Generate a set of subgraph matches and costs from a query result

    Arguments:
        records {List[tuple]}

    """

    # initialisation
    finished, iters = False, 0
    prv_optimum_match = None

    neighbourhood_matching_costs, query_result, beta = _get_matching_costs(
        records, hopping_distance)
    # keep a copy for successive iterations
    neighbourhood_matching_costs_cpy = neighbourhood_matching_costs.copy()

    label_costs = {
        (record.v, record.u): record.weight
        for record in query_result
    }
    label_costs_func = np.vectorize(lambda x: label_costs.get(tuple(x)))

    while True:

        # first optimisation
        neighbourhood_matching_costs = np.sort(
            neighbourhood_matching_costs,
            order=['v', 'u', 'vv', 'cost'],
            axis=0
        )
        partial_inference_costs = _get_partial_inference_costs(
            neighbourhood_matching_costs, beta)
        inference_costs = _get_inference_costs(partial_inference_costs)
        inference_costs.cost += label_costs_func(inference_costs[['v', 'u']])

        # second optimisation
        inference_costs = np.sort(inference_costs, order=['v', 'cost'])
        optimum_match = _get_optimal_match(inference_costs)
        inference_costs_dict = {
            (record.v, record.u): record.cost for record in inference_costs}
        apply = np.vectorize(
            lambda x: inference_costs_dict.get(tuple(x), iters))
        neighbourhood_matching_costs = neighbourhood_matching_costs_cpy.copy()
        neighbourhood_matching_costs.cost += apply(
            neighbourhood_matching_costs[['vv', 'uu']])
        iters += 1

        # test for convergance
        if prv_optimum_match is not None:
            diff = prv_optimum_match[['v', 'u']] == optimum_match[['v', 'u']]
            finished = (sum(diff) / len(optimum_match)) > .9
        if finished:
            break

        prv_optimum_match = optimum_match

        if iters >= max_iters:
            break

    neighbourhood_matching_costs = np.sort(
        neighbourhood_matching_costs,
        order=['v', 'u', 'vv', 'cost'],
        axis=0
    )
    refine = Refiner(neighbourhood_matching_costs)

    # normalise inference costs by the number of iterations
    inference_costs.cost /= iters
    inference_costs_dict = {
        (record.v, record.u): record.cost for record in inference_costs}

    inference_costs = np.sort(inference_costs, order=['cost'])
    subgraph_matches = []
    for seed in inference_costs[['v', 'u']]:
        subgraph_match = []
        refine(tuple(seed), subgraph_match)
        subgraph_match = sorted(subgraph_match)
        if subgraph_match not in subgraph_matches:
            subgraph_matches.append(subgraph_match)
    target_edges = group_by_first(['u', 'uu', 'dist_u'], query_result)[
        ['u', 'uu', 'dist_u']]
    mask = target_edges['dist_u'] > 0
    mask *= target_edges['dist_u'] <= 1
    mask *= target_edges['u'] < target_edges['uu']
    mask > 0
    target_edges = np.sort(target_edges[mask])
    return inference_costs_dict, subgraph_matches, iters, len(
        optimum_match), target_edges
