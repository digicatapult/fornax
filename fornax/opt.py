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

    Effectivly wraps a numpy structured array
    """


    columns = 'match_start match_end query_node_id target_node_id \
    query_proximity target_proximity delta totals misses weight'.split()
    types = 'i i i i f f f i i f'.split()

    def __init__(self, records, h=2, alpha=.3, lambda_=.3):
        """Create a new Frame instance
        
        Arguments:
            records {[(int, int, int, int, int, int, int, int, int, int)]} -- A list of tuples with dimensions Nx10.
        
        Keyword Arguments:
            h {int} -- max hopping distance (default: {2})
            alpha {float} -- proximity factor (default: {.3})
            lambda_ {float} -- label weight (default: {.3})
        """

        # init constants
        self.lambda_, self.h, self.alpha = lambda_, h, alpha

        # create a numpy structured array
        cleaned = [tuple(item if item is not None else -1 for item in tup) for tup in records]
        self.records = np.array(cleaned, dtype=list(zip(self.columns, self.types)))
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

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        return repr(self.records)

    def sort(self, order = ['match_start', 'match_end', 'query_node_id', 'delta']):
        """Sort the Frame inplace in order of columns specified in 'order'"""
        self.records = np.sort(self.records, order=order, axis=0)
    
    def _init_proximity(self):
        """Apply the proximity function to the hopping distances in columns target_proximity and query_proximity"""
        nan_idx = self.records['target_node_id'] < 0
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
        misses = {tuple(key): np.sum(np.less(group['target_node_id'], 0)) for key, group in zip(keys, groups)}

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
        self.max_iter, self.convergence_threshold = max_iter, convergence_threshold
        # state
        self.prv_result, self.result, self.sums, self.iters = None, None, None, 1

    def optimise(self, frame):
        """Reorder a frame using the recursive optimisation proceedure described in by NeMa
        
        Returns None when no more optimisation is permitted.
        
        Arguments:
            frame {[Frame]} -- A frame that has been optimised 0 or more times
        
        Returns:
            [Frame] -- A reordered frame with updated cost columns 'delta'
        """

        finished = False

        if self.iters > self.max_iter:
            return None

        frame.sort()
        best_target_nodes = group_by_first(
            ['match_start', 'match_end', 'query_node_id'], frame
        )[['match_start', 'match_end', 'query_node_id', 'target_node_id', 'delta']]
        
        # sum the costs from the previous iteration
        keys, groups = group_by(['match_start', 'match_end'], best_target_nodes)
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


class Refiner:
    """Take each of the matches and recursivly find all of their neighbours via a greedy algorithm"""

    def __init__(self, frame):
        """Initialise a refiner using a Frame instance
        
        Arguments:
            frame {[Frame]} -- A frame constructed records returned by the database
        """

        self.frame = frame

        matches =  {tuple(k): v for k, v in zip(*group_by(['match_start', 'match_end'], frame))}

        neighbours = {
            k: group_by_first(['match_start', 'match_end', 'query_node_id'], v)[['query_node_id', 'target_node_id']].tolist()
            for k,v in matches.items()
        }

        self.neighbours = {
            k: [item for item in v if self.valid_neighbours(k,item)] 
            for k,v in neighbours.items()
        }


    def __call__(self, seed, result=None):
        """Given a match_start, match end pair,
        greedily find the lowest cost neighbours
        recursivly covering the whole graph
        without cyclic paths
        
        Arguments:
            seed {[int, int]} -- match_start, match_end pair
        
        Keyword Arguments:
            result {[int, int]} -- stores the result between recursive calls (default: {None})
        
        Returns:
            [type] -- List of query_node, target_node id pairs constituting a result
        """

        if result is None:
            result = [seed]
        elif seed[0] in [first for first, second in result]:
            return result
        else:
            result.append(seed)
        
        for neighbour in self.neighbours[seed]:
            result = self(neighbour, result)
        
        return result

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
        if any(v < 0 for v in second):
            return False
        return True


def solve(n: int, h: int, alpha: float, records: List[tuple]) -> dict:
    """Find the best n matches from a set of records
    
    Arguments:
        n {int} -- number of results
        h {int} -- max hopping distance
        alpha {float} -- propergation factor
        records {List[tuple]} -- records returned from the database
    
    Returns:
        [[int,int], int] -- each result and corresponding score
    """

    prv_frame = Frame(records)
    optimiser = Optimiser()

    # optimise until optimiser.optimise returns None
    for frame in iter(lambda: optimiser.optimise(prv_frame), None):
        prv_frame = frame

    # normalise the costs by the number of iterations
    optimiser.sums['delta'] /= optimiser.iters + 1
    optimiser.result['delta'] /= optimiser.iters + 1

    refine = Refiner(frame)
    # order the matches by cost
    ordered = np.sort(optimiser.sums, order=['delta'])

    # greedily calcualte the set of all matching graphs using each match as a seed
    # starting with the lowest cost matches
    graphs = []
    for seed in ordered[['match_start', 'match_end']]:
        graph = sorted(refine(tuple(seed)))
        if graph not in graphs:
            graphs.append(graph)

    # record the scores of each graph
    sums_lookup = {(r[0], r[1]):r[2] for r in optimiser.sums}
    scores = [sum(sums_lookup[item] for item in graph) + (len(optimiser.result) - len(graph)) for graph in graphs]
    scores = [score/len(optimiser.result) for score in scores]

    # the the n best graphs
    ordered = sorted(zip(graphs, scores), key=lambda item: item[1])
    sliced = ordered[:min(n, len(graphs))]
    graphs, scores = list(zip(*sliced))

    return graphs, scores

