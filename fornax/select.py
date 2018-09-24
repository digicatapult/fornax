from fornax.model import Base, Match, QueryNode, QueryEdge, TargetNode, TargetEdge, Integer
from sqlalchemy.dialects.postgresql import ARRAY, array
from sqlalchemy.orm import Query, aliased
from sqlalchemy import literal, and_, cast, not_, func, or_, alias
from typing import List


def query_neighbours(h) -> Query:
    seed = Query([
            Match.start.label('match'),
            QueryNode.id.label('neighbour'),
            literal(0).label('distance')
    ]).join(QueryNode)
    n = seed.union(_neighbours(QueryNode, seed, h)).subquery()
    return Query([
        n.c.match,
        n.c.neighbour,
        func.min(n.c.distance).label('distance')
    ]).group_by(n.c.match, n.c.neighbour)


def target_neighbours(h) -> Query:
    seed = Query([
            Match.end.label('match'),
            TargetNode.id.label('neighbour'),
            literal(0).label('distance')
    ]).join(TargetNode)
    n = seed.union(_neighbours(TargetNode, seed, h)).subquery()
    return Query([
        n.c.match,
        n.c.neighbour,
        func.min(n.c.distance).label('distance')
    ]).group_by(n.c.match, n.c.neighbour)


def _neighbours(Node: Base, seed: Query, h, max_=None) -> Query:

    if max_ is None:
        max_, h = h, 1

    if Node.__tablename__ == QueryNode.__tablename__:
        Edge = QueryEdge
    elif Node.__tablename__ == TargetNode.__tablename__:
        Edge = TargetEdge
    else:
        raise ValueError("Unrecognised node type")

    seed = seed.subquery()
    neighbours = Query([
        seed.c.match.label('match'), 
        Edge.end.label('neighbour'), 
        literal(h).label('distance'),
    ])
    neighbours = neighbours.distinct()
    neighbours = neighbours.join(Edge, Edge.start == seed.c.neighbour)

    if h == max_:
        return neighbours
    else:
        return neighbours.union(_neighbours(Node, neighbours, h+1, max_=max_))


def _join(h: int) -> Query:

    left = query_neighbours(h).subquery()
    right = target_neighbours(h).subquery()
    NeighbourMatch = alias(Match, "neighbour_match")

    left_joined = Query([
        Match.start,
        Match.end,
        left.c.neighbour.label("neighbour_start"),
        left.c.distance,
        Match.weight,
    ])

    left_joined = left_joined.join(left, Match.start == left.c.match)
    left_joined = left_joined.subquery()

    right_joined = Query([
        Match.start,
        Match.end,
        NeighbourMatch.c.start.label("neighbour_start"),
        right.c.neighbour.label("neighbour_end"),
        right.c.distance,
    ])

    right_joined = right_joined.join(right, Match.end == right.c.match)
    right_joined = right_joined.join(NeighbourMatch, NeighbourMatch.c.end == right.c.neighbour)
    right_joined = right_joined.subquery()

    joined = Query([
        left_joined.c.start,
        left_joined.c.end,
        left_joined.c.neighbour_start,
        right_joined.c.neighbour_end,
        left_joined.c.distance,
        right_joined.c.distance,
        left_joined.c.weight
    ]).outerjoin(
        right_joined,
        and_(
            left_joined.c.start == right_joined.c.start,
            left_joined.c.end == right_joined.c.end,
            left_joined.c.neighbour_start == right_joined.c.neighbour_start,
        )
    )

    return joined


def join(h:int, batch_size:int = None):
    if batch_size is None:
        return _join(h)