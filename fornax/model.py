from sqlalchemy import Column, ForeignKey, Integer, String, Float, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Node(Base):

    __tablename__ = 'node'
    __table_args__ = (
        Index('trgm_idx', text("label gin_trgm_ops"), postgresql_using='gin'),
    )

    id = Column(Integer, primary_key=True)
    label = Column(String(255), nullable=False)
    type = Column(Integer, ForeignKey("node_type.id"), nullable=False)

    def __repr__(self):
        return "<Node(id={}, label={}, format={})>".format(
            self.id,
            self.label,
            self.type
        )


class Edge(Base):

    __tablename__ = 'edge'

    id = Column(Integer, primary_key=True)
    start = Column(Integer, ForeignKey("node.id"), nullable=False)
    end = Column(Integer, ForeignKey("node.id"), nullable=False)
    type = Column(Integer, ForeignKey("edge_type.id"), nullable=False)
    weight = Column(Float, nullable=False)

    def __repr__(self):
        return "<Edge(id={}, start={}, end={}, type={}, weight={})>".format(
            self.id,
            self.start,
            self.end,
            self.type,
            self.weight
        )


class NodeType(Base):

    __tablename__ = 'node_type'

    id = Column(Integer, primary_key=True)
    description = String()


class EdgeType(Base):

    __tablename__ = 'edge_type'

    id = Column(Integer, primary_key=True)
    description = String()