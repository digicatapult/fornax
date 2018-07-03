from fornax.model import Base, Match, QueryNode, TargetNode, Integer
from sqlalchemy.dialects.postgresql import ARRAY, array
from sqlalchemy.orm import Query, aliased
from sqlalchemy import literal, and_, cast, not_


def match_nearest_neighbours(Node: Base, h: int) -> Query:
    """
    
    Return a query to select all nodes within hopping distance h of a matching edge
    where the paths contain no cycles.

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


    See example in the postgres documentation:
    https://www.postgresql.org/docs/current/static/queries-with.html

    WITH RECURSIVE search_graph(id, link, data, depth, path, cycle) AS (
        SELECT g.id, g.link, g.data, 1,
            ARRAY[g.id],
            false
        FROM graph g
    UNION ALL
        SELECT g.id, g.link, g.data, sg.depth + 1,
            path || g.id,
            g.id = ANY(path)
        FROM graph g, search_graph sg
        WHERE g.id = sg.link AND NOT cycle
    )
    SELECT * FROM search_graph;

    """


    if h < 0:
        raise ValueError("max hopping distance 'h' must be greater than or equal to 0")

    # keep track of with nodes are parents and children in each recursion
    parent_match = aliased(Match, name="parent_match")
    parent_node = aliased(Node, name="parent_node")
    child_match = aliased(Match, name="child_match")
    child_node = aliased(Node, name="child_node")

    # Get all of the nodes that have a match
    seed_query = Query([
        parent_match.start.label('match_start'), 
        parent_match.end.label('match_end'), 
        parent_node.id.label('node_id'), 
        literal(0).label('distance'),
        cast(array([parent_node.id]), ARRAY(Integer)).label("path"),
    ])
    seed_query = seed_query.join(parent_node)
    seed_query = seed_query.cte(recursive=True)
    
    # recursivly get neighbouring nodes
    neighbour_query = Query([
            child_match.start.label('match_start'), 
            child_match.end.label('match_end'), 
            child_node.id.label('node_id'), 
            seed_query.c.distance + 1,
            seed_query.c.path + cast(array([child_node.id]), ARRAY(Integer)).label("path"),
    ])
    # new node is a neighbour of a previous node
    neighbour_query = neighbour_query.filter(child_node.neighbours.any(Node.id == seed_query.c.node_id))
    # node is within distance h of a match
    neighbour_query = neighbour_query.filter(seed_query.c.distance < h)
    # node has not been reached using a cyclical path
    neighbour_query = neighbour_query.filter(not_(seed_query.c.path.contains(array([child_node.id]))))
    # track the match that started the path
    neighbour_query = neighbour_query.filter(child_match.start.label('match_start') == seed_query.c.match_start)
    neighbour_query = neighbour_query.filter(child_match.end.label('match_end') == seed_query.c.match_end)

    query = seed_query.union(neighbour_query)
    return Query([
        query.c.match_start,
        query.c.match_end,
        query.c.node_id,
        query.c.distance,
    ])


def generate_query(h: int):
    """

    Returns a query to generate a table of the form

    | match.start | match.end | query_node.id | target_node.id | query_node_distance | target_node_distance | local_cost |
    |-------------|:---------:|--------------:|---------------:|--------------------:|---------------------:|-----------:|
    |     0       |     0     |       0       |       0        |          0          |          0           |      0     |
    |     0       |     0     |       0       |       1        |          0          |          1           |      0     |
    |     0       |     0     |       1       |       1        |          1          |          0           |      0     |

        for each match start:
            for each match end:
                for each query node where query_node_distance < h:
                    for each target node where target_node_distance < h
                        compute row
    
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
    ).order_by(
        query_node_subquery.c.match_start,
        query_node_subquery.c.match_end,
        query_node_subquery.c.node_id,
    )

    return query

