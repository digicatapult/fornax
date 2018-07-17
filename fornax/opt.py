import numpy as np
from typing import List
import pandas as pd

MAX_ITER = 10
CONVERGENCE_THRESHOLD = .95

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


def _to_dataFrame(records: List[tuple]) -> pd.DataFrame:
    """A a list of records and convert them into a dataFrame
    with standard columns
    
    Arguments:
        records {List[tuple]} -- List of 4 tuples with columns "['match_start', 'match_end', 'query_node_id', 'query_proximity']"
    
    Returns:
        pd.DataFrame -- A pandas dataFrame with columns "['match_start', 'match_end', 'query_node_id', 'query_proximity']"
    """

    return pd.DataFrame.from_records(
        records, 
        columns=[
            'match_start',
            'match_end', 
            'query_node_id', 
            'query_proximity'
        ]
    )


def _join(query_table: List[tuple], target_table: List[tuple]) -> np.array:
    """Perform an inner join using pandas between the query and target records.
    Joining on equal match start and match end.
    
    Arguments:
        query_table {List[tuple]} -- List of 4 tuples with columns "['match_start', 'match_end', 'query_node_id', 'query_proximity']"
        target_table {List[tuple]} -- List of 4 tuples with columns "['match_start', 'match_end', 'query_node_id', 'query_proximity']"
    
    Returns:
        np.array -- a numpy structured array with columns:
        "['match_start', 'match_end', 'query_node_id', 'query_proximity', 'target_node_id', 'target_proximity', 'delta']"
    """

    columns = ['match_start', 'match_end', 'query_node_id', 'query_proximity', 'target_node_id', 'target_proximity', 'delta']
    dtypes = list('iiiifff')
    ranked = query_table.merge(target_table, on=['match_start', 'match_end'], how='inner')
    ranked['delta'] = np.zeros(len(ranked))
    return np.core.records.fromarrays(ranked.values.transpose(), names=columns, formats=dtypes)


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


def _totals_and_misses(query_table: pd.DataFrame, uniq_matches: np.array, counts):
    """
    Count the number of query nodes in the query table around each match start. (totals)
    Count the number that don't have a match in the corresponding target community (misses)
    
    Arguments:
        query_table {pd.DataFrame} -- List of 4 tuples with columns "['match_start', 'match_end', 'query_node_id', 'query_proximity']"
        uniq_matches {np.array} -- a 2xN array of all of the match start-end pairs 
        counts {[type]} -- the number of query nodes in each match-start community
    
    Returns:
        np.array, np.array -- totals and misses in the same order as unique matches
    """

    count_groups = query_table.groupby(['match_start', 'match_end']).size().reset_index(name='counts')
    sizes = {k:v for k,v in zip(count_groups['match_start'], count_groups['counts'])}
    totals = np.array([sizes[i[0]] for i in uniq_matches])
    misses = totals - counts
    return totals, misses


def _get_or_1(d: dict):
    """
    return a function that vectorizes dictionary lookup 
    returning a default value of 1 if the key is not found
    """

    return np.vectorize(lambda x: d.get(tuple(x), 1))


def optimise(h: int, alpha: float, query_table: List[tuple], target_table: List[tuple]) -> dict:
    """
    Join together query and target tables.
    Perform the main iteration and optimisation step in the paper.
    
    Arguments:
        h {int} -- max hopping distance
        alpha {float} -- propagation factor
        query_table {List[tuple]} -- query communities of neighbours found using match_nearest_neighbours
        target_table {List[tuple]} -- target communities of neighbours found using match_nearest_neighbours
    
    Returns:
        dict, dict -- optimum matches and scores
    """

    query_table, target_table = _to_dataFrame(query_table), _to_dataFrame(target_table)
    ranked = _join(query_table, target_table)
    ranked['query_proximity'] = _proximity(h, alpha, ranked['query_proximity'])
    ranked['target_proximity'] = _proximity(h, alpha, ranked['target_proximity'])

    # get the first index of each of the unique target node values in the sorted table
    _, _, first_target_node_idx = _unique_index(ranked[['match_start', 'match_end', 'query_node_id']])
    first_target_node_idx = first_target_node_idx[:-1]

    # get the first index of each of the unique query node values in the sorted table
    # get a get of match start,end tuples and their frequency
    counts, uniq_matches, first_query_node_idx = _unique_index(ranked[first_target_node_idx][['match_start', 'match_end']])
    first_query_node_idx = first_query_node_idx[:-1]

    # get the first index of each of the unique target node values in the sorted table
    totals, misses = _totals_and_misses(query_table, uniq_matches, counts)
    _, _, first_target_match_index = _unique_index(ranked[first_target_node_idx][first_query_node_idx[:-1]]['match_start'])

    # a structured array for storing the sum of costs for all query-target node pairs in the communities around a matching pair
    first_query_nodes = np.array(
        list(zip(uniq_matches['match_start'], uniq_matches['match_end'], np.zeros(uniq_matches.shape[0]))), 
        dtype=[('match_start', 'int'), ('match_end', 'int'), ('sum', 'float')]
    )

    first_target_matches, finished, iters = None, None, 0

    while not finished and iters < MAX_ITER:
        # add the score in this iteration
        ranked['delta'] += _delta_plus(ranked['query_proximity'], ranked['target_proximity'])
        # sort the results
        ranked = np.sort(ranked, order=['match_start', 'match_end', 'query_node_id', 'delta'], axis=0)
        # filter lowest cost target node for each query node in each community
        first_target_nodes = ranked[first_target_node_idx]
        # sum the cost for each target-query node pair to get a cost for each match-start, match-end pair
        first_query_nodes['match_start'] = uniq_matches['match_start']
        first_query_nodes['match_end'] = uniq_matches['match_end']
        first_query_nodes['sum'] = np.add.reduceat(first_target_nodes['delta'], first_query_node_idx)
        # normalise
        first_query_nodes['sum'] += misses
        first_query_nodes['sum'] /= totals
        # sort for the best match end for each match start
        first_query_nodes.sort(axis=0, order=('match_start', 'sum'))
        # stopping critera
        if first_target_matches is not None:
            finished = sum(a==b for a,b in zip(first_target_matches, first_query_nodes[first_target_match_index]))
            finished = finished/len(first_target_matches) > CONVERGENCE_THRESHOLD
        iters += 1
        # filter lowest cost match end for each match start
        first_target_matches = first_query_nodes[first_target_match_index]
        # place the best score from the previous match in each row (U[i-1])
        lookup  = {(a,b): c for (a,b,c) in first_query_nodes}
        ranked['delta'] = _get_or_1(lookup)(ranked[['query_node_id', 'target_node_id']])
    
    return lookup, first_target_matches