import numpy as np
from typing import List

MAX_ITER = 10
CONVERGENCE_THRESHOLD = .95
LAMBDA = .3


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


def _unique_index(arr: np.array) -> (np.array, np.array, np.array):
    """Given a sorted array, find all of the unique values, count them and
    find the index of the first each of the unique values in the array.

    Assumes that the array has been sorted.

    Arguments:
        arr {np.array} -- a 1d numpy array
    
    Returns:
        arr {np.array} -- frequency of each unique value
        arr {np.array} -- value of each unique element in arr
        arr {np.array} -- index of the first of each unique value in arr
    """

    uniq, counts = np.unique(arr, return_counts=True)
    idx = np.insert(np.cumsum(counts), 0, 0)
    return counts, uniq, idx


def _get_or_1(d: dict):
    """
    return a function that vectorizes dictionary lookup 
    returning a default value of 1 if the key is not found
    """

    return np.vectorize(lambda x: d.get(tuple(x), 1))


def optimise(h: int, alpha: float, records: List[tuple]) -> dict:
    """[summary]
    
    Arguments:
        h {int} -- [description]
        alpha {float} -- [description]
        recrods {List[tuple]} -- [description]
    
    Returns:
        dict -- [description]
    """

    # create a structured array from the records returned from the database
    ranked = np.array(
        records,
        dtype=[
            ('match_start', 'i'), ('match_end', 'i'), ('query_node_id', 'i'),
            ('target_node_id', 'f'), ('query_proximity',
                                      'f'), ('target_proximity', 'f'),
            ('delta', 'f'), ('totals', 'i'), ('misses', 'i'), ('weight', 'f')
        ],
    )

    ranked = np.sort(
        ranked, order=['match_start', 'match_end', 'query_node_id', 'delta'], axis=0)


    # weight of 1 is a cost of zero
    ranked['weight'] = 1. - ranked['weight'] 

    # give not found nodes have a distance larger than h
    nan_idx = np.isnan(ranked['target_node_id'])
    ranked['target_proximity'][nan_idx] = h+1
    ranked['query_proximity'] = _proximity(h, alpha, ranked['query_proximity'])
    ranked['target_proximity'] = _proximity(h, alpha, ranked['target_proximity'])

    # get the first index of each of the unique target node values in the sorted table
    _, uniq_matches, first_target_node_idx = _unique_index(ranked[['match_start', 'match_end', 'query_node_id']])
    first_target_node_idx = first_target_node_idx[:-1]
    keys, totals = np.unique(uniq_matches[['match_start', 'match_end']], return_counts=True)
    totals = {(k[0], k[1]):t for k, t in zip(keys, totals)}

    _, uniq_matches, _ = _unique_index(ranked[nan_idx][['match_start', 'match_end', 'query_node_id']])
    keys, missed = np.unique(uniq_matches[['match_start', 'match_end']], return_counts=True)
    misses = {(k[0], k[1]):t for k,t in zip(keys, missed)}
    # totals = {tuple(k): v for k, v in zip(list(keys), list(totals))}


    ranked['totals'] = np.vectorize(lambda x: totals.get(
        (x[0], x[1])))(ranked[['match_start', 'match_end']])
    ranked['misses'] = np.vectorize(lambda x: misses.get(
        (x[0], x[1]), 0))(ranked[['match_start', 'match_end']])

    # get the first index of each of the unique query node values in the sorted table
    # get a get of match start,end tuples and their frequency
    _, uniq_matches, first_query_node_idx = _unique_index(
        ranked[first_target_node_idx][['match_start', 'match_end']])
    first_query_node_idx = first_query_node_idx[:-1]

    # get the first index of each of the unique target node values in the sorted table
    _, _, first_target_match_index = _unique_index(
        ranked[first_target_node_idx][first_query_node_idx[:-1]]['match_start'])

    # a structured array for storing the sum of costs for all query-target 
    # node pairs in the communities around a matching pair
    first_query_nodes = np.array(
        list(zip(uniq_matches['match_start'], uniq_matches['match_end'], np.zeros(
            uniq_matches.shape[0]))),
        dtype=[('match_start', 'int'), ('match_end', 'int'), ('sum', 'float')]
    )

    first_target_matches, finished, iters = None, None, 0

    while not finished and iters < MAX_ITER:
        # add the score in this iteration
        ranked['delta'] += LAMBDA*ranked['weight']
        ranked['delta'] += (1-LAMBDA)*(
            _delta_plus(ranked['query_proximity'], ranked['target_proximity']) + 
            ranked['misses']
        )/ranked['totals']
        # sort the results
        ranked = np.sort(
            ranked, order=['match_start', 'match_end', 'query_node_id', 'delta'], axis=0)
        # filter lowest cost target node for each query node in each community
        first_target_nodes = ranked[first_target_node_idx]
        # sum the cost for each target-query node pair to get a cost for each match-start, match-end pair
        first_query_nodes['match_start'] = uniq_matches['match_start']
        first_query_nodes['match_end'] = uniq_matches['match_end']
        # sum the costs between the intervals described by first_query_node_idx
        first_query_nodes['sum'] = np.add.reduceat(
            first_target_nodes['delta'], first_query_node_idx)
        # normalise by the number of terms in the sum (not in the paper)
        first_query_nodes['sum'] /= np.diff(
            np.append(first_query_node_idx, first_target_nodes.shape[0]))
        # sort for the best match end for each match start
        first_query_nodes.sort(axis=0, order=('match_start', 'sum'))
        # stopping critera
        if first_target_matches is not None:
            finished = sum(a == b for a, b in zip(
                first_target_matches, first_query_nodes[first_target_match_index]))
            finished = finished / \
                len(first_target_matches) > CONVERGENCE_THRESHOLD
        iters += 1
        # filter lowest cost match end for each match start
        first_target_matches = first_query_nodes[first_target_match_index]
        # place the best score from the previous match in each row (U[i-1])
        lookup = {(a, b): c for (a, b, c) in first_query_nodes}
        ranked['delta'] = _get_or_1(lookup)(
            ranked[['query_node_id', 'target_node_id']])

    # normalise by the number of iterations to give a cost between 0 and 1
    lookup = {(a, b): c/iters for (a, b, c) in first_query_nodes}
    return lookup, first_target_matches
