from sqlalchemy import Column, ForeignKey, Integer, Float, CheckConstraint, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

"""Eagerly load Nodes within a hopping distance of JOIN_DEPTH"""
JOIN_DEPTH = 2


class Match(Base):
    """Joins Query Nodes to Candidate Target Nodes"""

    __tablename__ = 'match'
    start = Column(Integer, ForeignKey("query_node.id"), primary_key=True)
    start_gid = Column(Integer, ForeignKey("query_node.gid"), primary_key=True)
    end = Column(Integer, ForeignKey("target_node.id"), primary_key=True)
    end_gid = Column(Integer, ForeignKey("target_node.gid"), primary_key=True)

    weight = Column(
        Float, 
        CheckConstraint("weight>0", name="min_check"),
        CheckConstraint("weight<=1", name="max_check"),
        nullable=False
    )
    
    query_node = relationship(
        'QueryNode', 
        primaryjoin="and_(Match.start == QueryNode.id, Match.start_gid == QueryNode.gid)", 
        back_populates="matches"
    )
    
    target_node = relationship(
        'TargetNode',
        primaryjoin="and_(Match.end == TargetNode.id, Match.end_gid == TargetNode.gid)",
        back_populates="matches"
    )

    def __repr__(self):
        return "<Match(start={}, end={}, weight={}, start_graph={}, end_graph={})>".format(
            self.start, self.end, self.weight, self.start_gid, self.end_gid
        )


class QueryNode(Base):
    """Node in the Query Graph"""

    __tablename__ = 'query_node'
    # node id
    id = Column(Integer, CheckConstraint("id>=0", name="q_min_id_check"), primary_key=True)
    # graph id
    gid = Column(Integer, primary_key=True)
    # node type
    type = Column(Integer)
    
    def neighbours(self):
        return [x.end_node for x in self.start_edges]

    matches = relationship("Match", primaryjoin="and_(Match.start == QueryNode.id, Match.start_gid == QueryNode.gid)")

    def __repr__(self):
        return "<QueryNode(id={}, graph={}, type={})>".format(self.id, self.gid, self.type)


class QueryEdge(Base):
    """Joins Nodes it the Query Graph"""

    __tablename__ = 'query_edge'
    start = Column(Integer, ForeignKey("query_node.id"), primary_key=True)
    end = Column(Integer, ForeignKey("query_node.id"), primary_key=True)
    gid = Column(Integer, ForeignKey("query_node.gid"), primary_key=True)

    start_node = relationship(
        QueryNode, 
        primaryjoin="and_(QueryEdge.start == QueryNode.id, QueryEdge.gid == QueryNode.gid)", 
        backref="start_edges"
    )

    end_node = relationship(
        QueryNode, 
        primaryjoin="and_(QueryEdge.end == QueryNode.id, QueryEdge.gid == QueryNode.gid)", 
        backref="end_edges"
    )

    def __repr__(self):
        return "<QueryEdge(start={}, end={}, graph={})>".format(self.start, self.end, self.graph)


class TargetNode(Base):
    """Node in the Target Graph"""

    __tablename__ = 'target_node'
    # node id
    id = Column(Integer, CheckConstraint("id>=0", name="t_min_id_check"), primary_key=True)
    # graph id
    gid = Column(Integer, primary_key=True)
    # node type
    type = Column(Integer)

    def neighbours(self):
        return [x.end_node for x in self.start_edges]

    matches = relationship("Match", primaryjoin="and_(Match.start == TargetNode.id, Match.end_gid == TargetNode.gid)")

    def __repr__(self):
        return "<TargetNode(id={}, graph={}, type={})>".format(self.id, self.graph, self.type)


class TargetEdge(Base):
    """Joins Nodes in the Target Graph"""

    __tablename__ = 'target_edge'
    start = Column(Integer, ForeignKey("target_node.id"), primary_key=True)
    end = Column(Integer, ForeignKey("target_node.id"), primary_key=True)
    gid = Column(Integer, ForeignKey("query_node.gid"), primary_key=True)

    start_node = relationship(
        TargetNode, 
        primaryjoin="and_(TargetEdge.start == TargetNode.id, TargetEdge.gid == TargetNode.gid)", 
        backref="start_edges"
    )
    
    end_node = relationship(
        TargetNode, 
        primaryjoin="and_(TargetEdge.end == TargetNode.id, TargetEdge.gid == TargetNode.gid)", 
        backref="end_edges"
    )

    def __repr__(self):
        return "<TargetEdge(start={}, end={}, graph={})>".format(self.start, self.end, self.graph)