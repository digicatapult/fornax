from sqlalchemy import Column, ForeignKey, Integer, Float, CheckConstraint, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

"""Eagerly load Nodes within a hopping distance of JOIN_DEPTH"""
JOIN_DEPTH = 2


class Match(Base):
    """Joins Query Nodes to Candidate Target Nodes"""

    __tablename__ = 'match'
    start = Column(Integer, ForeignKey("node.id"), primary_key=True)
    start_gid = Column(Integer, ForeignKey("node.gid"), primary_key=True)
    end = Column(Integer, ForeignKey("node.id"), primary_key=True)
    end_gid = Column(Integer, ForeignKey("node.gid"), primary_key=True)

    weight = Column(
        Float, 
        CheckConstraint("weight>0", name="min_check"),
        CheckConstraint("weight<=1", name="max_check"),
        nullable=False
    )
    
    query_node = relationship(
        'Node', 
        primaryjoin="and_(Match.start == Node.id, Match.start_gid == Node.gid)", 
        back_populates="matches"
    )
    
    target_node = relationship(
        'Node',
        primaryjoin="and_(Match.end == Node.id, Match.end_gid == Node.gid)",
        back_populates="matches"
    )

    def __repr__(self):
        return "<Match(start={}, end={}, weight={}, start_graph={}, end_graph={})>".format(
            self.start, self.end, self.weight, self.start_gid, self.end_gid
        )


class Node(Base):
    """Node in the Query Graph"""

    __tablename__ = 'node'
    # node id
    id = Column(Integer, CheckConstraint("id>=0", name="q_min_id_check"), primary_key=True)
    # graph id
    gid = Column(Integer, primary_key=True)
    # node type
    type = Column(Integer)
    
    def neighbours(self):
        return [x.end_node for x in self.start_edges]

    matches = relationship("Match", primaryjoin="and_(Match.start == Node.id, Match.start_gid == Node.gid)")

    def __repr__(self):
        return "<Node(id={}, graph={}, type={})>".format(self.id, self.gid, self.type)


class Edge(Base):
    """Joins Nodes it the Query Graph"""

    __tablename__ = 'edge'
    start = Column(Integer, ForeignKey("node.id"), primary_key=True)
    end = Column(Integer, ForeignKey("node.id"), primary_key=True)
    gid = Column(Integer, ForeignKey("node.gid"), primary_key=True)

    start_node = relationship(
        Node, 
        primaryjoin="and_(Edge.start == Node.id, Edge.gid == Node.gid)", 
        backref="start_edges"
    )

    end_node = relationship(
        Node, 
        primaryjoin="and_(Edge.end == Node.id, Edge.gid == Node.gid)", 
        backref="end_edges"
    )

    def __repr__(self):
        return "<Edge(start={}, end={}, graph={})>".format(self.start, self.end, self.graph)
