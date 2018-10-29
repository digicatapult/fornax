from sqlalchemy import Column, ForeignKey, Integer, Float, CheckConstraint, String, ForeignKeyConstraint, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

"""Eagerly load Nodes within a hopping distance of JOIN_DEPTH"""
JOIN_DEPTH = 2


class Match(Base):
    """Joins Query Nodes to Candidate Target Nodes"""

    __tablename__ = 'match'
    __table_args__ = (
        PrimaryKeyConstraint('start', 'end', 'start_graph_id', 'end_graph_id', 'query_id'),
        ForeignKeyConstraint(['start', 'start_graph_id'], ['node.node_id', 'node.graph_id'], name="fk_match_start"),
        ForeignKeyConstraint(['end', 'end_graph_id'], ['node.node_id', 'node.graph_id'], name="fk_match_end")
    )

    start = Column(Integer)
    end = Column(Integer)
    start_graph_id = Column(Integer)
    end_graph_id = Column(Integer)
    query_id = Column(Integer)

    weight = Column(Float, 
        CheckConstraint("weight>0", name="min_check"),
        CheckConstraint("weight<=1", name="max_check"),
        nullable=False
    )
    
    start_node = relationship(
        'Node', 
        primaryjoin="and_(Match.start == Node.node_id, Match.start_graph_id == Node.node_id)", 
        backref="start_matches"
    )
    
    end_node = relationship(
        'Node',
        primaryjoin="and_(Match.end == Node.node_id, Match.end_graph_id == Node.graph_id)",
        backref="end_matches"
    )

    def __repr__(self):
        return "<Match(start={}, end={}, weight={}, start_graph_id={}, end_graph_id={}, query={})>".format(
            self.start, self.end, self.weight, self.start_graph_id, self.end_graph_id, self.query_id
        )


class Node(Base):
    """Node in the Query Graph"""

    __tablename__ = 'node'
    __table_args__ = (
        PrimaryKeyConstraint('node_id', 'graph_id'),
    )
    node_id = Column(Integer, CheckConstraint("node_id>=0", name="q_min_id_check"))
    graph_id = Column(Integer)
    
    def neighbours(self):
        return [x.end_node for x in self.start_edges]

    def __repr__(self):
        return "<Node(node_id={}, graph_id={})>".format(self.node_id, self.graph_id)


class Edge(Base):
    """Joins Nodes it the Query Graph"""

    __tablename__ = 'edge'
    __table_args__ = (
        PrimaryKeyConstraint('start', 'end', 'graph_id'),
        ForeignKeyConstraint(['start', 'graph_id'], ['node.node_id', 'node.graph_id']),
        ForeignKeyConstraint(['end', 'graph_id'], ['node.node_id', 'node.graph_id'])
    )

    start = Column(Integer)
    end = Column(Integer)
    graph_id = Column(Integer)

    start_node = relationship(
        'Node', 
        primaryjoin='and_(Node.node_id == Edge.start, Node.graph_id == Edge.graph_id)', 
        backref='start_edges'
    )

    end_node = relationship(
        'Node', 
        primaryjoin='and_(Node.node_id == Edge.end, Node.graph_id == Edge.graph_id)',
        backref='end_edges'
    )
    
    def __repr__(self):
        return "<Edge(start={}, end={}, graph_id={})>".format(self.start, self.end, self.graph_id)
