from sqlalchemy import Column, ForeignKey, Integer, Float, CheckConstraint, String, ForeignKeyConstraint, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

"""Eagerly load Nodes within a hopping distance of JOIN_DEPTH"""
JOIN_DEPTH = 2


class Match(Base):
    """Joins Query Nodes to Candidate Target Nodes"""

    __tablename__ = 'match'
    start = Column(Integer, ForeignKey("node.node_id"))
    end = Column(Integer, ForeignKey("node.node_id"))
    start_graph_id = Column(Integer, ForeignKey("node.graph_id"))
    end_graph_id = Column(Integer, ForeignKey("node.graph_id"))
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

    __table_args__ = (
        PrimaryKeyConstraint('start', 'end', 'start_graph_id', 'end_graph_id', 'query_id'),
        ForeignKeyConstraint(
            ['start', 'end', 'start_graph_id', 'end_graph_id'],
            ['node.node_id', 'node.node_id', 'node.graph_id', 'node.graph_id']
        )
    )

    def __repr__(self):
        return "<Match(start={}, end={}, weight={}, start_graph={}, end_graph={}, query={})>".format(
            self.start, self.end, self.weight, self.start_gid, self.end_gid, self.query_id
        )


class Node(Base):
    """Node in the Query Graph"""

    __tablename__ = 'node'
    node_id = Column(Integer, CheckConstraint("node_id>=0", name="q_min_id_check"), primary_key=True)
    graph_id = Column(Integer, primary_key=True)
    
    def neighbours(self):
        return [x.end_node for x in self.start_edges]

    def __repr__(self):
        return "<Node(id={}, graph={}, type={})>".format(self.id, self.gid, self.type)


class Edge(Base):
    """Joins Nodes it the Query Graph"""

    __tablename__ = 'edge'
    start = Column(Integer, ForeignKey("node.node_id"))
    end = Column(Integer, ForeignKey("node.node_id"))
    graph_id = Column(Integer, ForeignKey("node.graph_id"))

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

    __table_args__ = (
        PrimaryKeyConstraint('start', 'end', 'graph_id'),
        ForeignKeyConstraint(
            ['start', 'end', 'graph_id'],
            ['node.node_id', 'node.node_id', 'node.graph_id']
        )
    )
    
    def __repr__(self):
        return "<Edge(start={}, end={}, graph={})>".format(self.start, self.end, self.graph)
