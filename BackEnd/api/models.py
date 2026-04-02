from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from api.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    # 建立与历史记录的关联
    histories = relationship("TrajectoryHistory", back_populates="owner")


class TrajectoryHistory(Base):
    __tablename__ = "trajectory_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    trajectory_id = Column(String, index=True)
    model_id = Column(String)  # 记录使用的是 exp1/exp2/exp3/exp4
    predicted_mode = Column(String)  # 记录预测结果 (Walk, Bike等)
    confidence = Column(Float)  # 记录置信度
    distance = Column(Float, default=0.0)  # 记录绿色减排里程
    created_at = Column(DateTime, default=datetime.utcnow)  # 预测时间

    owner = relationship("User", back_populates="histories")