from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TrajectoryPoint(BaseModel):
    lat: float = Field(..., description="纬度")
    lng: float = Field(..., description="经度")
    timestamp: str = Field(..., description="时间戳")
    speed: Optional[float] = Field(None, description="速度 (m/s)")


class TrajectoryStats(BaseModel):
    distance: float = Field(..., description="总距离 (m)")
    duration: float = Field(..., description="总时长 (s)")
    avg_speed: float = Field(..., description="平均速度 (m/s)")
    max_speed: float = Field(..., description="最高速度 (m/s)")


class TrajectoryPrediction(BaseModel):
    trajectory_id: str = Field(..., description="轨迹ID")
    predicted_mode: str = Field(..., description="预测的交通方式")
    confidence: float = Field(..., description="置信度 (0-1)")
    points: List[TrajectoryPoint] = Field(..., description="轨迹点列表")
    stats: TrajectoryStats = Field(..., description="轨迹统计信息")


class TransportMode(BaseModel):
    id: str = Field(..., description="交通方式ID")
    name: str = Field(..., description="交通方式名称")
    color: str = Field(..., description="显示颜色")
    icon: str = Field(..., description="图标")


class ExperimentInfo(BaseModel):
    id: str = Field(..., description="实验ID")
    name: str = Field(..., description="实验名称")
    description: str = Field(..., description="实验描述")
    features: List[str] = Field(..., description="使用的特征")
    status: str = Field(..., description="状态: completed, not_trained, training")


class EvaluationReport(BaseModel):
    accuracy: float = Field(..., description="总体准确率")
    precision: Dict[str, float] = Field(..., description="各类别精确率")
    recall: Dict[str, float] = Field(..., description="各类别召回率")
    f1_score: Dict[str, float] = Field(..., description="各类别F1分数")
    classification_report: Dict[str, Any] = Field(..., description="完整分类报告")


class PredictionSummary(BaseModel):
    total_predictions: int = Field(..., description="总预测数")
    mode_distribution: Dict[str, int] = Field(..., description="各交通方式预测数量")
    accuracy: float = Field(..., description="准确率")


class DatasetStats(BaseModel):
    total_trajectories: int = Field(..., description="总轨迹数")
    total_users: int = Field(..., description="用户数")
    mode_distribution: Dict[str, int] = Field(..., description="各交通方式样本数")
    avg_trajectory_length: float = Field(..., description="平均轨迹长度")
    date_range: Dict[str, str] = Field(..., description="时间范围")
    total_distance: str = Field(..., description="总里程（估算）")


class DataCleaningStep(BaseModel):
    name: str = Field(..., description="步骤名称")
    count: int = Field(..., description="数量")


class DataCleaningStats(BaseModel):
    steps: List[DataCleaningStep] = Field(..., description="数据清洗各步骤数据")


class TrainingProgress(BaseModel):
    task_id: str = Field(..., description="任务ID")
    exp_name: str = Field(..., description="实验名称")
    epoch: int = Field(..., description="当前epoch")
    total_epochs: int = Field(..., description="总epoch数")
    loss: float = Field(..., description="当前loss")
    accuracy: float = Field(..., description="当前accuracy")
    status: str = Field(..., description="状态: training, completed, failed")


class TrainingRequest(BaseModel):
    exp_name: str = Field(..., description="实验名称 (exp1/exp2/exp3/exp4)")
    epochs: Optional[int] = Field(50, description="训练轮数")
    batch_size: Optional[int] = Field(32, description="批次大小")
    learning_rate: Optional[float] = Field(0.001, description="学习率")


class TrainingResponse(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")


class UserCreate(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")

class UserResponse(BaseModel):
    id: int
    username: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class HistoryRecord(BaseModel):
    id: int
    trajectory_id: str
    model_id: str
    predicted_mode: str
    confidence: float
    distance: float
    created_at: datetime

    class Config:
        from_attributes = True