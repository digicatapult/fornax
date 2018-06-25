from sqlalchemy import Column, ForeignKey, Integer, Float, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

"""Eagerly load Nodes within a hopping distance of JOIN_DEPTH"""
JOIN_DEPTH = 2


class QueryEdge(Base):
    """Joins Nodes it the Query Graph"""

    __tablename__ = 'query_edge'
    start = Column(Integer, ForeignKey("query_node.id"), primary_key=True)
    end = Column(Integer, ForeignKey("query_node.id"), primary_key=True)

    def __repr__(self):
        return "<QueryEdge(id={}, start={}, end={})>".format(
            self.id, self.start, self.end
        )


class TargetEdge(Base):
    """Joins Nodes in the Target Graph"""

    __tablename__ = 'target_edge'
    start = Column(Integer, ForeignKey("target_node.id"), primary_key=True)
    end = Column(Integer, ForeignKey("target_node.id"), primary_key=True)

    def __repr__(self):
        return "<TargetEdge(id={}, start={}, end={})>".format(
            self.id, self.start, self.end
        )


class Match(Base):
    """Joins Query Nodes to Candidate Target Nodes"""

    __tablename__ = 'matching_edge'
    start = Column(Integer, ForeignKey("query_node.id"), primary_key=True)
    end = Column(Integer, ForeignKey("target_node.id"), primary_key=True)
    weight = Column(
        Float, 
        CheckConstraint("weight>0", name="max_check"),
        CheckConstraint("weight<=1", name="min_check"),
        nullable=False
    )


    def __repr__(self):
        return "<TargetEdge(id={}, start={}, end={}, weight={})>".format(
            self.id, self.start, self.end, self.weight
        )


class QueryNode(Base):
    """Node in the Query Graph"""

    __tablename__ = 'query_node'
    id = Column(Integer, primary_key=True)
    
    neighbours = relationship(
        "QueryNode",
        secondary=QueryEdge.__table__,
        primaryjoin=id==QueryEdge.__table__.c.start,
        secondaryjoin=id==QueryEdge.__table__.c.end,
        join_depth=JOIN_DEPTH,
        lazy="joined"
    )

    matches = relationship(
        "QueryNode",
        secondary=Match.__table__,
        primaryjoin=id==Match.__table__.c.start,
        secondaryjoin=id==Match.__table__.c.end,
    )

    def __repr__(self):
        return "<QueryNode(id={})>".format(self.id)


class TargetNode(Base):
    """Node in the Target Graph"""

    __tablename__ = 'target_node'
    id = Column(Integer, primary_key=True)

    neighbours = relationship(
        "TargetNode",
        secondary=TargetEdge.__table__,
        primaryjoin=id==TargetEdge.__table__.c.start,
        secondaryjoin=id==TargetEdge.__table__.c.end,
        join_depth=JOIN_DEPTH,
        lazy="joined"
    )

    matches = relationship(
        "TargetNode",
        secondary=Match.__table__,
        primaryjoin=id==Match.__table__.c.end,
        secondaryjoin=id==Match.__table__.c.start,
    )

    def __repr__(self):
        return "<TargetNode(id={})>".format(self.id)

