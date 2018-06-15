from sqlalchemy.orm import Query
from sqlalchemy.sql.expression import literal
from fornax.model import Node, Edge


def get_candidate(distance: float, label: str) -> Query:
    """ 
        return a sqlalchemy query object to fuzzy 
        match a query node label to target node labels
    """

    if not 0 <= distance < 1:
        raise ValueError("distances must be between zero and one")
    query = Query(Node)
    query = query.filter(Node.label.op('<->')(label) < distance)
    return query


def get_neighbours(query: Query) -> Query:
    """ 
        starting with a query that selects a table of nodes
        return a query that returns all of the neighbours
        of each node

        query: a query that selects a table a nodes
        returns: a query selecting a table of nodes with their parent id
    """
    
    subquery = query.subquery()
    new_query = Query(
        [
            Node.id, 
            Node.label, 
            Node.type, 
            Edge.start.label('parent'),
            literal(1).label('distance')
        ]
    )
    new_query = new_query.join(Edge, Edge.end == Node.id)
    new_query = new_query.filter(Edge.start == subquery.c.id)
    return new_query