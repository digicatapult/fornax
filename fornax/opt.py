import numpy as np
from itertools import starmap
from typing import List

MAX_ITER = 10
CONVERGENCE_THRESHOLD = .95
LAMBDA = .3


h, alpha, LAMBDA = 2, .3, .3

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
    
    uniq, counts = np.unique(arr[columns], return_counts=True)
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


def optimise(h: int, alpha: float, records: List[tuple]) -> dict:
    """[summary]
    
    Arguments:
        h {int} -- [description]
        alpha {float} -- [description]
        recrods {List[tuple]} -- [description]
    
    Returns:
        dict -- [description]
    """

    # create a structured array from the database records
    ranked = np.array(
        records,
        dtype=[
            ('match_start', 'i'), ('match_end', 'i'), ('query_node_id', 'i'),
            ('target_node_id', 'f'), ('query_proximity', 'f'), ('target_proximity', 'f'),
            ('delta', 'f'), ('totals', 'i'), ('misses', 'i'), ('weight', 'f')
        ],
    )

    ranked = np.sort(ranked, order=['match_start', 'match_end', 'query_node_id'], axis=0)
    nan_idx = np.isnan(ranked['target_node_id'])
    ranked['target_proximity'][nan_idx] = h+1
    ranked['query_proximity'] = _proximity(h, alpha, ranked['query_proximity'])
    ranked['target_proximity'] = _proximity(h, alpha, ranked['target_proximity'])
    ranked['weight'] = 1. - ranked['weight']

    # for each (match_start, match_end) pair how many query nodes are there?
    first = group_by_first(['match_start', 'match_end', 'query_node_id'], ranked)
    # each group will have a row per query node for each match pair
    # if a query node has no target then the target_node_id field will be Nan
    keys, groups = group_by(['match_start', 'match_end'], first)

    # how many of those query nodes have no corresponding target nodes?
    misses = {
        tuple(key): np.sum(np.isnan(group['target_node_id'])) 
        for key, group in zip(keys, groups)
    }

    # how many do have corresponding target nodes?
    totals = {
        tuple(key): len(group) - misses[tuple(key)] 
        for key, group in zip(keys, groups)
    }

    # apply therse counts to the table
    apply = np.vectorize(lambda x: totals.get(tuple(x)))
    ranked['totals'] = apply(ranked[['match_start', 'match_end']])

    apply = np.vectorize(lambda x: misses.get(tuple(x)))
    ranked['misses'] = apply(ranked[['match_start', 'match_end']])

    names, formats = 'match_start, match_end, delta', 'i, i, f'

    # calculate proximity and label costs for each row
    label_score = ranked['weight']
    proximity_score = _delta_plus(ranked['query_proximity'], ranked['target_proximity'])
    proximity_score += ranked['misses']
    proximity_score /= ranked['totals']
    ranked['weight'] += LAMBDA*label_score + (1.-LAMBDA)*proximity_score

    prv_result, result, finished = None, None, False
    for iters in range(10):

        # order the relationships by their cost
        ranked = np.sort(ranked, order=['match_start', 'match_end', 'query_node_id', 'delta'], axis=0)
        
        # find the lowest cost target_node for each query node
        stacked = group_by_first(['match_start', 'match_end', 'query_node_id'], ranked)
        
        # sum the costs from the previous iteration
        keys, groups = group_by(['match_start', 'match_end'], stacked)
        summed = starmap(lambda key, group: (*key, np.sum(group['delta'])/len(group)), zip(keys, groups))
        stacked =  np.vstack(summed)
        sums = np.core.records.fromarrays(stacked.transpose(), names=names, formats=formats)
        sums = np.sort(sums, order=['match_start', 'delta'])
        prv_result = result
        result = group_by_first('match_start', sums)

        # create a lookup table for the sums
        sums_lookup = {(r[0], r[1]):r[2] for r in sums}
        # record the costs from this iteration
        apply = np.vectorize(lambda x: sums_lookup.get(tuple(x), iters))
        ranked['delta'] = ranked['weight'] + apply(ranked[['query_node_id', 'target_node_id']])

        if prv_result is not None:
            diff = result[['match_start', 'match_end']] == prv_result[['match_start', 'match_end']]
            finished = (sum(diff) / len(result)) > .9
        if finished:
            break

    sums['delta'] /= iters + 1
    result['delta'] /= iters + 1
    sums_lookup = {(r[0], r[1]):r[2] for r in sums}
    return sums_lookup, result


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