import numpy as np
from typing import List
import pandas as pd

MAX_ITER = 10
CONVERGENCE_THRESHOLD = .95

def proximity(h: float, alpha: float, distances: np.ndarray) -> np.ndarray:
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

def delta_plus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

def optimise(h: int, alpha: float, rows_a: List[tuple], rows_b: List[tuple]) -> dict:
        """[summary]
        
        Arguments:
            h {int} -- [description]
            alpha {float} -- [description]
            rows_a {List[tuple]} -- [description]
            rows_b {List[tuple]} -- [description]
        
        Returns:
            dict -- [description]
        """


        columns = ['match_start', 'match_end', 'query_node_id', 'query_proximity', 'target_node_id', 'target_proximity', 'delta']
        dtypes = list('iiiifff')

        rows_a = pd.DataFrame.from_records(rows_a, columns=['match_start', 'match_end', 'query_node_id', 'query_proximity'])
        rows_b = pd.DataFrame.from_records(rows_b, columns=['match_start', 'match_end', 'target_node_id', 'target_proximity'])
        ranked = rows_a.merge(rows_b, on=['match_start', 'match_end'], how='inner')
        ranked['delta'] = np.zeros(len(ranked))

        # create a numpy array - we will sort this list to find the best matches
        count_groups = rows_a.groupby(['match_start', 'match_end']).size().reset_index(name='counts')
        sizes = {k:v for k,v in zip(count_groups['match_start'], count_groups['counts'])}
        get_or_1 = np.vectorize(lambda x: d.get(tuple(x), 1))

        # calculate the proximity of the query and target nodes
        ranked['query_proximity'] = proximity(h, alpha, ranked['query_proximity'])
        ranked['target_proximity'] = proximity(h, alpha, ranked['target_proximity'])
        ranked = np.core.records.fromarrays(ranked.values.transpose(), names=columns, formats=dtypes)

        _, counts = np.unique(ranked[['match_start', 'match_end', 'query_node_id']], return_counts=True)
        best_matching_function_idx = np.insert(np.cumsum(counts), 0, 0)[:-1]

        matches, counts = np.unique(ranked[best_matching_function_idx][['match_start', 'match_end']], return_counts=True)
        match_idx = np.insert(np.cumsum(counts), 0, 0)[:-1]
        totals = np.array([sizes[i[0]] for i in matches])
        misses = totals - counts

        _, counts = np.unique(ranked[best_matching_function_idx][match_idx[:-1]]['match_start'], return_counts=True)
        optimum_idx = np.insert(np.cumsum(counts), 0, 0)

        optimum_match = None
        finished = None
        iters = 0

        while not finished and iters < MAX_ITER:
            # add the score in this iteration
            ranked['delta'] += delta_plus(ranked['query_proximity'], ranked['target_proximity'])
            # rank the results
            ranked = np.sort(ranked, order=['match_start', 'match_end', 'query_node_id', 'delta'], axis=0)
            # take the best scores         
            optimised = ranked[best_matching_function_idx]
            #  group by each match
            zipped = zip(matches['match_start'], matches['match_end'], np.add.reduceat(optimised['delta'], match_idx))
            score = np.array(list(zipped), dtype=[('match_start', 'int'), ('match_end', 'int'), ('sum', 'float')])
            score['sum'] += misses
            score['sum'] /= totals
            score.sort(axis=0, order=('match_start', 'sum'))
            if optimum_match is not None:
                finished = sum(a==b for a,b in zip(optimum_match, score[optimum_idx]))
                finished = finished/len(optimum_match) > CONVERGENCE_THRESHOLD
            iters += 1
            optimum_match = score[optimum_idx]
            # place the best score from the previous match in each row (U[i-1])
            d  = {(a,b): c for (a,b,c) in score}
            ranked['delta'] = get_or_1(ranked[['query_node_id', 'target_node_id']])
        
        return d, optimum_match