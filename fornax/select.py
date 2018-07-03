from fornax.model import Base, Match, QueryNode, TargetNode, Integer
from sqlalchemy.dialects.postgresql import ARRAY, array
from sqlalchemy.orm import Query, aliased
from sqlalchemy import literal, and_, cast, not_
import numpy as np

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
        literal(h).label('distance')
    ])


def match_nearest_neighbours(Node: Base, h: int) -> Query:
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
        ValueError -- if node_type is not QueryNode or TargetNode
        NotImplementedError -- if h > 2
    
    Returns:
        Query -- returns a SQLAlchemy query

    """


    if h < 0:
        raise ValueError("max hopping distance 'h' must be greater than or equal to 0")

    parent_match = aliased(Match, name="parent_match")
    parent_node = aliased(Node, name="parent_node")
    child_match = aliased(Match, name="child_match")
    child_node = aliased(Node, name="child_node")
    search_graph = Query([
        parent_match.start.label('match_start'), 
        parent_match.end.label('match_end'), 
        parent_node.id.label('node_id'), 
        literal(0).label('distance'),
        cast(array([parent_node.id]), ARRAY(Integer)).label("path"),
    ]).join(
        parent_node
    ).cte(recursive=True)
    
    query = search_graph.union(
        Query([
            child_match.start.label('match_start'), 
            child_match.end.label('match_end'), 
            child_node.id.label('node_id'), 
            search_graph.c.distance + 1,
            search_graph.c.path + cast(array([child_node.id]), ARRAY(Integer)).label("path"),
        ]).filter(
            child_node.neighbours.any(Node.id == search_graph.c.node_id)
        ).filter(
            search_graph.c.distance < h
        ).filter(
            not_(search_graph.c.path.contains(array([child_node.id])))
        ).filter(
            child_match.start.label('match_start') == search_graph.c.match_start
        ).filter(
            child_match.end.label('match_end') == search_graph.c.match_end
        )
    ) 

    return Query([
        query.c.match_start,
        query.c.match_end,
        query.c.node_id,
        query.c.distance,
    ]).distinct()


def generate_query(h: int):
    """

    Returns a query to generate a table of the form

    | match.start | match.end | query_node.id | target_node.id | query_node_distance | target_node_distance | local_cost |
    |-------------|:---------:|--------------:|---------------:|--------------------:|---------------------:|-----------:|
    |     0       |     0     |       0       |       0        |          0          |          0           |      0     |
    |     0       |     0     |       0       |       1        |          0          |          1           |      0     |
    |     0       |     0     |       1       |       1        |          1          |          0           |      0     |

    The will be a row for

        each match start:
            each match end:
                each query node where query_node_distance < h:
                    each target node where target_node_distance < h
    
    query_node_distance is the distance between the query node and the query node at match.start

    target_node_distance is the distance between the target node and the target node at match.end

    local_cost is a column initialised to zero which will be used to compute the matching costs at
    each iteration.

    Arguments:
        h {int} -- max hopping distnace
    
    Returns:
        Query -- a sqlalchemy query object
    """

    query_node_subquery = match_nearest_neighbours(QueryNode, h).subquery()
    target_node_subquery = match_nearest_neighbours(TargetNode, h).subquery()
    
    query = Query([
        query_node_subquery.c.match_start,
        query_node_subquery.c.match_end,
        query_node_subquery.c.node_id, 
        target_node_subquery.c.node_id,
        query_node_subquery.c.distance,
        target_node_subquery.c.distance,
        literal(0)
    ]).join(
        target_node_subquery,
        and_(
            query_node_subquery.c.match_start == target_node_subquery.c.match_start, 
            query_node_subquery.c.match_end == target_node_subquery.c.match_end           
        )
    )

    return query

