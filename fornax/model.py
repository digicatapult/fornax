from sqlalchemy import Column, Integer, Float, CheckConstraint, String
from sqlalchemy import PrimaryKeyConstraint, Index, UniqueConstraint
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Graph(Base):
    """ A graph containing nodes and edges """
    __tablename__ = 'graph'
    graph_id = Column(Integer, primary_key=True)


class Query(Base):

    __tablename__ = 'query'
    __table_args__ = (
        UniqueConstraint('query_id', 'start_graph_id', 'end_graph_id'),
    )

    query_id = Column(Integer, primary_key=True)
    start_graph_id = Column(
        Integer, ForeignKey("graph.graph_id"), nullable=False, index=True)
    end_graph_id = Column(
        Integer, ForeignKey("graph.graph_id"), nullable=False, index=True)
    Index(
        'query_idx', 'query_id', 'start_graph_id', 'end_graph_id', unique=True)


class Match(Base):
    """Joins Query Nodes to Candidate Target Nodes"""

    __tablename__ = 'match'
    __table_args__ = (
        PrimaryKeyConstraint(
            'query_id', 'start_graph_id', 'end_graph_id', 'start', 'end'),
        ForeignKeyConstraint(
            ['start_graph_id', 'start'], ['node.graph_id', 'node.node_id'],
            name="fk_match_start"),
        ForeignKeyConstraint(
            ['end_graph_id', 'end'], ['node.graph_id', 'node.node_id'],
            name="fk_match_end"),
        ForeignKeyConstraint(
            ['query_id', 'start_graph_id', 'end_graph_id'],
            ['query.query_id', 'query.start_graph_id', 'query.end_graph_id'],
            name="fk_query"
        )
    )

    start = Column(Integer)
    end = Column(Integer)
    start_graph_id = Column(Integer)
    end_graph_id = Column(Integer)
    query_id = Column(Integer)
    meta = Column(String, nullable=True)

    weight = Column(
        Float,
        CheckConstraint("weight>0", name="min_check"),
        CheckConstraint("weight<=1", name="max_check"),
        nullable=False
    )

    start_node = relationship(
        'Node',
        primaryjoin="and_(Match.start == Node.node_id, \
        Match.start_graph_id == Node.node_id)",
        backref="start_matches"
    )

    end_node = relationship(
        'Node',
        primaryjoin="and_(Match.end == Node.node_id, \
        Match.end_graph_id == Node.graph_id)",
        backref="end_matches"
    )

    def __repr__(self):
        return "<Match(start={}, end={}, weight={}, start_graph_id={}, \
        end_graph_id={}, query={})>".format(
            self.start, self.end, self.weight, self.start_graph_id,
            self.end_graph_id, self.query_id
        )


class Node(Base):
    """Node in a Graph"""

    __tablename__ = 'node'
    __table_args__ = (
        PrimaryKeyConstraint('graph_id', 'node_id'),
    )
    node_id = Column(
        Integer,
        CheckConstraint("node_id>=0", name="q_min_id_check")
    )
    graph_id = Column(Integer, ForeignKey("graph.graph_id"))
    meta = Column(String, nullable=True)

    def neighbours(self):
        return [x.end_node for x in self.start_edges]

    def __repr__(self):
        return "<Node(node_id={}, graph_id={})>".format(
            self.node_id, self.graph_id)


class Edge(Base):
    """Joins Nodes in a Graph"""

    __tablename__ = 'edge'
    __table_args__ = (
        PrimaryKeyConstraint('graph_id', 'start', 'end'),
        ForeignKeyConstraint(
            ['graph_id', 'start'],
            ['node.graph_id', 'node.node_id']
        ),
        ForeignKeyConstraint(
            ['graph_id', 'end'],
            ['node.graph_id', 'node.node_id']
        )
    )

    start = Column(Integer)
    end = Column(Integer)
    graph_id = Column(Integer)
    meta = Column(String, nullable=True)

    start_node = relationship(
        'Node',
        primaryjoin='and_(Node.node_id == Edge.start, \
        Node.graph_id == Edge.graph_id)',
        backref='start_edges'
    )

    end_node = relationship(
        'Node',
        primaryjoin='and_(Node.node_id == Edge.end, \
        Node.graph_id == Edge.graph_id)',
        backref='end_edges'
    )

    def __repr__(self):
        return "<Edge(start={}, end={}, graph_id={})>".format(
            self.start, self.end, self.graph_id)
