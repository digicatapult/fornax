from fornax.model import Base, Match, QueryNode, TargetNode, Integer
from sqlalchemy.dialects.postgresql import ARRAY, array
from sqlalchemy.orm import Query, aliased
from sqlalchemy import literal, and_, cast, not_, func, or_


def match_nearest_neighbours(matches: Query, Node: Base, h: int) -> Query:
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
    matches_sub = matches.subquery()
    parent_node = aliased(Node, name="parent_node")
    child_match = aliased(Match, name="child_match")
    child_node = aliased(Node, name="child_node")

    # Get all of the nodes that have a match
    seed_query = Query([
        matches_sub.c.start.label('match_start'), 
        matches_sub.c.end.label('match_end'),
        matches_sub.c.weight.label('weight'),  
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
        query.c.weight,
        query.c.node_id,
        query.c.distance,
    ])


def join_neighbourhoods(matches: Query, h: int) -> Query:
    """

    Returns a query to generate a table of the form

    | match.start | match.end | query_node.id | target_node.id | query_node_distance | target_node_distance | delta | misses | totals | weight |
    |-------------|:---------:|--------------:|---------------:|--------------------:|---------------------:|:-----:|:------:|:------:|:------:|
    |     0       |     0     |       0       |       0        |          0          |          0           |   0   |    0   |    0   |    0   |
    |     0       |     0     |       0       |       1        |          0          |          1           |   0   |    0   |    0   |    0   |
    |     0       |     0     |       1       |       1        |          1          |          0           |   0   |    0   |    0   |    0   |

        for each match start:
            for each match end:
                for each query node where query_node_distance < h:
                    for each target node where target_node_distance < h
                        compute row
    
    query_node_distance is the distance between the query node and the query node at match.start

    target_node_distance is the distance between the target node and the target node at match.end

    Query nodes with no correspondances in the target graph will have Null values for target_node.id and target_node_distance

    delta, misses and totals are place holders to be populated by opt.py

    Arguments:
        matches {query} -- the set of matches to solve for
        h {int} -- max hopping distance
    
    Returns:
        Query -- a sqlalchemy query object

    """

    query = match_nearest_neighbours(matches, QueryNode, h).subquery()
    target = match_nearest_neighbours(matches, TargetNode, h).subquery()
    right = target.join(Match, target.c.node_id == Match.end)

    left = Query([
        query.c.match_start,
        query.c.match_end,
        query.c.node_id.label('query_id'),
        target.c.node_id.label('target_id'),
        query.c.distance.label('query_distance'),
        target.c.distance.label('target_distance'),
        literal(0),
        literal(0),
        literal(0),
        query.c.weight
    ])
    left = left.outerjoin(right,
        and_(
            query.c.node_id == Match.start,
            query.c.match_start == target.c.match_start,
            query.c.match_end == target.c.match_end,
        )
    )

    return left