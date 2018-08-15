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


class Frame:

    """A class to represent a table of values returned from fornex.search
    The class represents a table with names self.columns.
    """


    columns = 'match_start match_end query_node_id target_node_id \
    query_proximity target_proximity delta totals misses weight'.split()
    types = 'i i i f f f f i i f'.split()

    def __init__(self, records, h=2, alpha=.3, lambda_=.3):
        """Create a new Frame instance
        
        Arguments:
            records {[type]} -- A list of tuples with dimensions Nx10.
        
        Keyword Arguments:
            h {int} -- max hopping distance (default: {2})
            alpha {float} -- proximity factor (default: {.3})
            lambda_ {float} -- label weight (default: {.3})
        """

        # init constants
        self.lambda_, self.h, self.alpha = lambda_, h, alpha

        # create a numpy structured array
        self.records = np.array(records, dtype=list(zip(self.columns, self.types)))
        # group the different node ids together
        self.sort()
        # initialise query and target proximity columns
        self._init_proximity()
        # initialise totals and misses columns
        self._totals_and_misses()

        # calculate proximity and label costs for each row
        label_score = self.records['weight']
        proximity_score = _delta_plus(self.records['query_proximity'], self.records['target_proximity'])
        proximity_score += self.records['misses']
        proximity_score /= self.records['totals']
        self.records['weight'] += self.lambda_*label_score + (1.-self.lambda_)*proximity_score

    def __getitem__(self, key):
        """Get the column with name key
        
        Arguments:
            key {[str]} -- string or list of strings representing column names
        
        Returns:
            [np.array] -- a numpy array 
        """

        return self.records[key]

    def __setitem__(self, key, item):
        """Set the column with name key to item
        
        Arguments:
            key {[str]} -- a column names
            item {[np.array]} -- a numpy array with the same dimensions as the existing column
        """

        self.records[key] = item

    def sort(self, order = ['match_start', 'match_end', 'query_node_id', 'delta']):
        """Sort the Frame inplace in order of columns specified in 'order'
        """
        self.records = np.sort(self.records, order=order, axis=0)
    
    def _init_proximity(self):
        """Apply the proximity function to the hopping distances in columns target_proximity and query_proximity
        """

        nan_idx = np.isnan(self.records['target_node_id'])
        self.records['target_proximity'][nan_idx] = self.h + 1
        self.records['query_proximity'] = _proximity(self.h, self.alpha, self.records['query_proximity'])
        self.records['target_proximity'] = _proximity(self.h, self.alpha, self.records['target_proximity'])
        self.records['weight'] = 1. - self.records['weight']

    def _totals_and_misses(self):
        """Populates columns totals and misses

        Totals represents the number of neighbouring query nodes within h hops of each query node
        Misses represents totals minus the number of neighbouring target nodes within h hops of each target query matching pair
        """

        # for each (match_start, match_end) pair how many query nodes are there?
        first = group_by_first(['match_start', 'match_end', 'query_node_id'], self.records)
        # each group will have a row per query node for each match pair
        # if a query node has no target then the target_node_id field will be Nan
        keys, groups = group_by(['match_start', 'match_end'], first)

        # how many of those query nodes have no corresponding target nodes?
        misses = {tuple(key): np.sum(np.isnan(group['target_node_id'])) for key, group in zip(keys, groups)}

        # how many do have corresponding target nodes?
        totals = {tuple(key): len(group) - misses[tuple(key)] for key, group in zip(keys, groups)}

        # apply therse counts to the table
        apply = np.vectorize(lambda x: totals.get(tuple(x)))
        self.records['totals'] = apply(self.records[['match_start', 'match_end']])

        apply = np.vectorize(lambda x: misses.get(tuple(x)))
        self.records['misses'] = apply(self.records[['match_start', 'match_end']])


class Optimiser:
    """Optimiser take a Frame and sucessivly reorders it by calling optimise(frame)."""

    def __init__(self, max_iter=10, convergence_threshold=.95):
        """Create a new optimiser instance
        
        Keyword Arguments:
            max_iter {int} -- the maximum number of iterations before the optimiser gives up (default: {10})
            convergence_threshold {float} -- the fraction of results that do not change before the optimiser gives up (default: {.95})
        """

        # constants
        self.max_iter, self.convergence_threshold = 10, .95
        # state
        self.prv_result, self.result, self.sums, self.iters = None, None, None, 1

    def optimise(self, frame):
        """Reorder a frame using the recursive optimisation proceedure described in by NeMa
        
        Returns None when no more optimisation is permitted.
        
        Arguments:
            frame {[type]} -- A frame that has been optimised 0 or more times
        
        Returns:
            [type] -- A reordered frame with updated cost columns 'delta'
        """

        finished = False

        if self.iters > self.max_iter:
            return None
        
        frame.sort()

        stacked = group_by_first(['match_start', 'match_end', 'query_node_id'], frame)
        
        # sum the costs from the previous iteration
        keys, groups = group_by(['match_start', 'match_end'], stacked)
        summed = starmap(lambda key, group: (*key, np.sum(group['delta'])/len(group)), zip(keys, groups))
        stacked = np.vstack(summed)
        self.sums = np.core.records.fromarrays(
            stacked.transpose(), 
            names='match_start, match_end, delta', 
            formats='i, i, f'
        )
        self.sums = np.sort(self.sums, order=['match_start', 'delta'])
        # create a lookup table for the sums
        sums_lookup = {(r[0], r[1]):r[2] for r in self.sums}
        
        self.prv_result = self.result
        self.result = group_by_first('match_start', self.sums)
        # record the costs from this iteration
        apply = np.vectorize(lambda x: sums_lookup.get(tuple(x), self.iters))

        frame['delta'] = frame['weight'] + apply(frame[['query_node_id', 'target_node_id']])

        if self.prv_result is not None:
            diff = self.result[['match_start', 'match_end']] == self.prv_result[['match_start', 'match_end']]
            finished = (sum(diff) / len(self.result)) > .9

        if finished:
            return None

        self.iters += 1

        return frame

def greedy_grab(idx, neighbours, path=None):
    if path is None:
        path = set([idx])
    else:
        if idx in path:
            return path
        else:
            path = path.union(set([idx]))
    for item in neighbours[idx].items():
        path = path.union(greedy_grab(item, neighbours, path))
    return path

def get_neighbours(ranked, sums):

    # if there's a dead heat then a node will always be ignored

    best = []
    _, groups = group_by('match_start', sums)
    for group in groups:
        first = group['delta'][0]
        sliced = group[[delta==first for delta in group['delta']]]
        best.append(sliced)

    result = np.hstack(best)
    best = set(result[['match_start', 'match_end']].tolist())
    keys, groups = group_by(['match_start', 'match_end'], ranked)

    neighbours = {}
    for key, group in zip(keys, groups):
        key = tuple(key)
        if key not in best:
            continue
        neighbours[key] = {}
        for value in reversed(group[['query_node_id', 'target_node_id']]):
            value = tuple(value)
            if key[0] == value[0]:
                continue
            neighbours[key][value[0]] = value[1]

    return neighbours

def optimise(h: int, alpha: float, records: List[tuple]) -> dict:

    prv_frame = Frame(records)
    optimiser = Optimiser()

    # optimise until optimiser.optimise returns None
    for frame in iter(lambda: optimiser.optimise(prv_frame), None):
        prv_frame = frame

    optimiser.sums['delta'] /= optimiser.iters + 1
    optimiser.result['delta'] /= optimiser.iters + 1
    sums_lookup = {(r[0], r[1]):r[2] for r in optimiser.sums}
    return sums_lookup, optimiser.result

