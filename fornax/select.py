from fornax.model import Base, Match, QueryNode, TargetNode
from sqlalchemy.orm import Query
from sqlalchemy import literal, and_

def select(node_type: Base, h:int) -> Query:
    """SELECT statement as a function. Equivalent to:

        SELECT match.start, match.end, node_type.id, h 
        FROM match 
        OUTER JOIN node_type
    
    Arguments:
        node_type {Base} -- Either QueryNode or TargetNode
        h {int} -- hopping distance
    
    Returns:
        Query -- returns a SQLAlchemy query
    """

    return Query([
        Match.start.label('match_start'),
        Match.end.label('match_end'),
        node_type.id.label('node_id'), 
        literal(h).label('query_distance')
    ])


def match_nearest_neighbours(node_type: Base, h=2) -> Query:
    """
    
    Filter the select statement above to return all nodes within
    hopping distance h of a matching edge.

    Usage:
        # Get the query node for each match and all of its neighbours
        # within 1 hop
        query = select.match_nearest_neighbours(QueryNode, h=1)
    
        # Get the target node for each match and all of its neighbours
        # within 2 hops
        query = select.match_nearest_neighbours(TargetNode, h=1)

    Arguments:
        node_type {Base} -- Either QueryNode or TargetNode
    
    Keyword Arguments:
        h {int} -- h {int} -- hopping distance
    
    Raises:
        ValueError -- if h < 0
        NotImplementedError -- if h > 2
    
    Returns:
        Query -- returns a SQLAlchemy query
    """

    if h < 0:
        raise ValueError("max hopping distance 'h' must be greater than or equal to 0")

    elif h == 0:
        query = select(node_type, h).filter(node_type.id == Match.start)

    elif h == 1:
        query = match_nearest_neighbours(node_type, h=0).union(
            select(
                node_type, 1
            ).filter(
                node_type.neighbours.any(node_type.id == Match.start)
            )
        )
    
    elif h == 2:
        query = match_nearest_neighbours(node_type, h=1).union(
            select(
                node_type, 2
            ).filter(
                node_type.neighbours.any(node_type.neighbours.any(node_type.id == Match.start))
            ).filter(
                node_type.id != Match.start
            )
        )

    else:
        raise NotImplementedError("not implemented for h = '{}'".format(h))

    return query

