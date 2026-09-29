"""SQLAlchemy ORM models for the existing HMNC_PRO SQL Server database."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class User(Base):
    __tablename__ = "Users"

    UserID = Column(Integer, primary_key=True, autoincrement=True)
    Username = Column(String(100), nullable=False)
    PasswordHash = Column(String(255), nullable=False)
    Role = Column(String(20), nullable=False)
    Status = Column(String(20), nullable=False)
    CreatedAt = Column(DateTime, nullable=False)
    LastLogin = Column(DateTime, nullable=True)

    predictions = relationship(
        "Prediction",
        back_populates="user",
    )

    kernel_usage = relationship(
        "KernelUsageHistory",
        back_populates="user",
    )


class Prediction(Base):
    __tablename__ = "Predictions"

    PredictionID = Column(Integer, primary_key=True, autoincrement=True)

    UserID = Column(
        Integer,
        ForeignKey("Users.UserID"),
        nullable=False,
    )

    CreatedAt = Column(DateTime, nullable=False)

    SepalLength = Column(Float, nullable=False)
    SepalWidth = Column(Float, nullable=False)
    PetalLength = Column(Float, nullable=False)
    PetalWidth = Column(Float, nullable=False)

    PredictedSpecies = Column(String(50), nullable=False)

    ProbabilitySetosa = Column(Float, nullable=True)
    ProbabilityVersicolor = Column(Float, nullable=True)
    ProbabilityVirginica = Column(Float, nullable=True)

    ModelName = Column(String(100), nullable=True)
    Kernel = Column(String(20), nullable=True)

    user = relationship(
        "User",
        back_populates="predictions",
    )


class ModelHistory(Base):
    __tablename__ = "ModelHistory"

    ModelID = Column(Integer, primary_key=True, autoincrement=True)

    ModelName = Column(String(100), nullable=False)
    Kernel = Column(String(20), nullable=False)

    Accuracy = Column(Float, nullable=True)
    PrecisionScore = Column(Float, nullable=True)
    RecallScore = Column(Float, nullable=True)
    F1Score = Column(Float, nullable=True)

    CreatedAt = Column(DateTime, nullable=False)


class KernelUsageHistory(Base):
    __tablename__ = "KernelUsageHistory"

    UsageID = Column(Integer, primary_key=True, autoincrement=True)

    UserID = Column(
        Integer,
        ForeignKey("Users.UserID"),
        nullable=False,
    )

    Kernel = Column(String(20), nullable=False)
    UsedAt = Column(DateTime, nullable=False)

    user = relationship(
        "User",
        back_populates="kernel_usage",
    )