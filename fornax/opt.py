import numpy as np
import collections
from typing import List

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

def optimise(h: int, alpha: float, rows: List[tuple]) -> dict:
        """[summary]
        
        Arguments:
            arr {List[tuple]} -- [description]
        
        Returns:
            dict -- [description]
        """

        columns = ['match_start', 'match_end', 'query_node_id', 'target_node_id', 'query_proximity', 'target_proximity', 'delta']
        dtypes = ['int', 'int', 'int', 'int', 'float', 'float', 'float']

        # create a numpy array - we will sort this list to find the best matches
        ranked = np.array(rows, dtype=list(zip(columns, dtypes)))

        # calculate the proximity of the query and target nodes
        ranked['query_proximity'] = proximity(h, alpha, ranked['query_proximity'])
        ranked['target_proximity'] = proximity(h, alpha, ranked['target_proximity'])

        _, counts = np.unique(ranked[['match_start', 'match_end', 'query_node_id']], return_counts=True)
        best_matching_function_idx = np.insert(np.cumsum(counts), 0, 0)[:-1]

        matches, counts = np.unique(ranked[best_matching_function_idx][['match_start', 'match_end']], return_counts=True)
        match_idx = np.insert(np.cumsum(counts), 0, 0)[:-1]

        _, counts = np.unique(ranked[best_matching_function_idx][match_idx[:-1]]['match_start'], return_counts=True)
        optimum_idx = np.insert(np.cumsum(counts), 0, 0)

        optimum_match = None
        finished = None

        while not finished:
            # add the score in this iteration
            ranked['delta'] += delta_plus(ranked['query_proximity'], ranked['target_proximity'])
            # rank the results
            ranked = np.sort(ranked, order=['match_start', 'match_end', 'query_node_id', 'delta'], axis=0)
            # take the best scores         
            optimised = ranked[best_matching_function_idx]
            #  group by each match
            gen = zip(matches['match_start'], matches['match_end'], np.add.reduceat(optimised['delta'], match_idx))
            score = np.array(list(gen), dtype=[('match_start', 'int'), ('match_end', 'int'), ('sum', 'float')])
            score.sort(axis=0, order=('match_start', 'sum'))
            if optimum_match is not None:
                finished = sum(a==b for a,b in zip(optimum_match, score[optimum_idx]))/len(optimum_match) > .9
            optimum_match = score[optimum_idx]
            # place the best score from the previous match in each row (U[i-1])
            d = {(a,b): c for (a,b,c) in score}
            func = np.vectorize(lambda x: d.get(tuple(x), 1))
            ranked['delta'] = func(ranked[['query_node_id', 'target_node_id']])
        
        return d, {(a,b): c for (a,b,c) in optimum_match}