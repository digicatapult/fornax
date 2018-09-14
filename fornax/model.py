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
    end = Column(Integer, ForeignKey("target_node.id"), primary_key=True)
    weight = Column(
        Float, 
        CheckConstraint("weight>0", name="min_check"),
        CheckConstraint("weight<=1", name="max_check"),
        nullable=False
    )
    
    query_node = relationship('QueryNode', back_populates="matches")
    target_node = relationship('TargetNode', back_populates="matches")

    def __repr__(self):
        return "<Match(start={}, end={}, weight={})>".format(
            self.start, self.end, self.weight
        )


class QueryNode(Base):
    """Node in the Query Graph"""

    __tablename__ = 'query_node'
    id = Column(Integer, CheckConstraint("id>=0", name="q_min_id_check"), primary_key=True)
    type = Column(Integer)
    
    def neighbours(self):
        return [x.end_node for x in self.start_edges]

    matches = relationship("Match")

    def __repr__(self):
        return "<QueryNode(id={}, type={})>".format(self.id, self.type)


class QueryEdge(Base):
    """Joins Nodes it the Query Graph"""

    __tablename__ = 'query_edge'
    start = Column(Integer, ForeignKey("query_node.id"), primary_key=True)
    end = Column(Integer, ForeignKey("query_node.id"), primary_key=True)
    start_node = relationship(QueryNode, primaryjoin=start == QueryNode.id, backref="start_edges")
    end_node = relationship(QueryNode, primaryjoin=end == QueryNode.id, backref="end_edges")

    def __repr__(self):
        return "<QueryEdge(start={}, end={})>".format(self.start, self.end)


class TargetNode(Base):
    """Node in the Target Graph"""

    __tablename__ = 'target_node'
    id = Column(Integer, CheckConstraint("id>=0", name="t_min_id_check"), primary_key=True)
    type = Column(Integer)

    def neighbours(self):
        return [x.end_node for x in self.start_edges]

    matches = relationship("Match")

    def __repr__(self):
        return "<TargetNode(id={}, type={})>".format(self.id, self.type)


class TargetEdge(Base):
    """Joins Nodes in the Target Graph"""

    __tablename__ = 'target_edge'
    start = Column(Integer, ForeignKey("target_node.id"), primary_key=True)
    end = Column(Integer, ForeignKey("target_node.id"), primary_key=True)

    start_node = relationship(TargetNode, primaryjoin=start == TargetNode.id, backref="start_edges")
    end_node = relationship(TargetNode, primaryjoin=end == TargetNode.id, backref="end_edges")

    def __repr__(self):
        return "<TargetEdge(start={}, end={})>".format(self.start, self.end)