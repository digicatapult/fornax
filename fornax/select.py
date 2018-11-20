from fornax.model import Base, Match, Node, Edge
from sqlalchemy.orm import Query
from sqlalchemy import literal, and_, func, alias
from typing import Tuple
from collections import Iterable


def neighbours(h: int, start) -> Query:

    if start:
        seed = Query([
            Match.start.label('match'),
            Match.start_graph_id.label('graph_id'),
            Node.node_id.label('neighbour'),
            literal(0).label('distance')
        ]).join(
            Node,
            and_(
                Node.node_id == Match.start,
                Node.graph_id == Match.start_graph_id
            )
        )
    else:
        seed = Query([
            Match.end.label('match'),
            Match.end_graph_id.label('graph_id'),
            Node.node_id.label('neighbour'),
            literal(0).label('distance')
        ]).join(
            Node,
            and_(
                Node.node_id == Match.end,
                Node.graph_id == Match.end_graph_id
            )
        )

    n = seed.union(_neighbours(seed, 1, h)).subquery()
    return Query([
        n.c.match,
        n.c.neighbour,
        func.min(n.c.distance).label('distance')
    ]).group_by(n.c.match, n.c.neighbour)


def _neighbours(seed: Query, h, max_=None) -> Query:

    seed = seed.subquery()
    neighbours_query = Query([
        seed.c.match.label('match'),
        seed.c.graph_id.label('graph_id'),
        Edge.end.label('neighbour'),
        literal(h).label('distance'),
    ])
    neighbours_query = neighbours_query.distinct()
    neighbours_query = neighbours_query.join(
        Edge,
        and_(
            Edge.start == seed.c.neighbour,
            Edge.graph_id == seed.c.graph_id
        )
    )

    if h == max_:
        return neighbours_query
    else:
        return neighbours_query.union(
            _neighbours(neighbours_query, h + 1, max_=max_)
        )


def join(query_id: int, h: int, offsets: Tuple[int, int]=None) -> Query:

    left = neighbours(h, True).subquery()
    right = neighbours(h, False).subquery()
    NeighbourMatch = alias(Match, "neighbour_match")

    left_joined = Query([
        Match.start,
        Match.end,
        left.c.neighbour.label("neighbour_start"),
        left.c.distance,
        Match.weight,
    ]).filter(Match.query_id == query_id)

    left_joined = left_joined.join(left, Match.start == left.c.match)

    # batching of data is implemented here
    if offsets is not None:
        if not isinstance(offsets, Iterable) or not len(offsets) == 2:
            raise ValueError('offsets must be of length 2')
        # limit the query between offset "offsets[0]" and limit "offsets[1]"
        left_joined = left_joined.slice(int(offsets[0]), int(offsets[1]))

    left_joined = left_joined.subquery()

    right_joined = Query([
        Match.start,
        Match.end,
        NeighbourMatch.c.start.label("neighbour_start"),
        right.c.neighbour.label("neighbour_end"),
        right.c.distance,
    ])

    right_joined = right_joined.join(right, Match.end == right.c.match)
    right_joined = right_joined.join(
        NeighbourMatch, NeighbourMatch.c.end == right.c.neighbour
    )
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
