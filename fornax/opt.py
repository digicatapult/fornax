import numpy as np
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
        dtypes = ['int32', 'int32', 'int32', 'int32', 'float32', 'float32', 'float32']

        # create a numpy array - we will sort this list to find the best matches
        ranked = np.array(rows, dtype=list(zip(columns, dtypes)))

        # calculate the proximity of the query and target nodes
        ranked['query_proximity'] = proximity(h, alpha, ranked['query_proximity'])
        ranked['target_proximity'] = proximity(h, alpha, ranked['target_proximity'])

        # create an index of where to find the best node for each match
        mask = np.logical_not(ranked[['match_start', 'match_end', 'query_node_id']][1:] == ranked[['match_start', 'match_end', 'query_node_id']][:-1])
        mask = np.concatenate((np.array([True]), mask))
        match_idx = ranked[mask][['match_start', 'match_end']]
        match_mask = np.logical_not(match_idx[1:] == match_idx[:-1])
        match_mask = np.where(np.concatenate((np.array([True]), match_mask)))[0][1:]

        for i in range(10):
            # add the score in this iteration
            ranked['delta'] += delta_plus(ranked['query_proximity'], ranked['target_proximity'])
            # rank the results
            ranked = np.sort(ranked, order=['match_start', 'match_end', 'query_node_id', 'delta'], axis=0)
            # take the best scores         
            optimised = ranked[mask]
            #  group by each match
            matches = np.split(optimised, match_mask)
            # sum the cost for each match
            scores = {tuple(match[['match_start', 'match_end']][0]):np.sum(match['delta']) for match in matches}
            # place the best score from the previous match in each row (U[i-1])
            ranked['delta'] = np.vectorize(lambda x: scores.get(tuple(x), 1))(ranked[['query_node_id', 'target_node_id']])
        
        return scores