from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    Float,
    String,
    DateTime,
    Boolean,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    provider_type = Column(String(128), nullable=False)
    description = Column(String(1024), nullable=True)
    frequency = Column(String(32), nullable=True)
    extra = Column(JSON, nullable=True)


class TimeseriesPoint(Base):
    __tablename__ = 'timeseries_points'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    # Flexible numeric columns; store as JSON for wide coverage without schema churn
    values = Column(JSON, nullable=False)
    # Optional dedup/versioning
    is_final = Column(Boolean, nullable=False, default=True)

    __table_args__ = (
        UniqueConstraint('dataset_id', 'timestamp', name='uq_dataset_timestamp'),
    )


