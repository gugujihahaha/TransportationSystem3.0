from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from api.schemas import ExperimentInfo, EvaluationReport, PredictionSummary

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent.parent


def get_exp_dir(exp_id: str) -> Path:
    return BASE_DIR / exp_id


def get_evaluation_dir(exp_id: str) -> Path:
    return get_exp_dir(exp_id) / "evaluation_results"


EXPERIMENTS = [
    {
        "id": "exp1",
        "name": "实验1: 基线",
        "description": "traj_9 + stats_18，层次化Bi-LSTM，带类别权重的CrossEntropyLoss",
        "features": ["traj_9", "stats_18", "层次化Bi-LSTM", "带类别权重的CrossEntropyLoss"],
        "status": "completed"
    },
    {
        "id": "exp2",
        "name": "实验2: +OSM",
        "description": "在exp1基础上：traj_9→traj_21，引入OSM空间特征，模型结构不变",
        "features": ["traj_21", "stats_18", "OSM空间特征", "层次化Bi-LSTM", "带类别权重的CrossEntropyLoss"],
        "status": "completed"
    },
    {
        "id": "exp3",
        "name": "实验3: +天气",
        "description": "在exp2基础上：加入天气特征，双路编码器（轨迹hidden=128 + 天气hidden=32），AttentionPooling后拼接stats_18",
        "features": ["traj_21", "stats_18", "OSM空间特征", "天气特征", "双路编码器 + AttentionPooling", "带类别权重的CrossEntropyLoss"],
        "status": "completed"
    },
    {
        "id": "exp4",
        "name": "实验4: 方法对比",
        "description": "与exp3使用相同数据和模型结构，损失函数改为LabelSmoothingFocalLoss（γ=2.0，smoothing=0.1）",
        "features": ["traj_21", "stats_18", "OSM空间特征", "天气特征", "双路编码器 + AttentionPooling", "LabelSmoothingFocalLoss"],
        "status": "completed"
    },
]


@router.get("", response_model=List[ExperimentInfo])
async def get_experiments():
    """获取所有实验的基本信息和状态"""
    return [
        ExperimentInfo(**exp) for exp in EXPERIMENTS
    ]


@router.get("/{exp_id}/report", response_model=EvaluationReport)
async def get_exp_report(exp_id: str):
    """获取指定实验的评估报告"""
    report_path = get_evaluation_dir(exp_id) / "evaluation_report.json"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"{exp_id} 评估报告不存在")
    
    try:
        with open(report_path, 'r', encoding='utf-8-sig') as f:
            report_data = json.load(f)
        
        precision = {}
        recall = {}
        f1_score = {}
        
        for key, value in report_data.items():
            if key in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(value, dict):
                precision[key] = value.get('precision', 0)
                recall[key] = value.get('recall', 0)
                f1_score[key] = value.get('f1-score', 0)
        
        return EvaluationReport(
            accuracy=report_data.get('accuracy', 0),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            classification_report=report_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取评估报告失败: {str(e)}")


@router.get("/{exp_id}/confusion-matrix")
async def get_exp_confusion_matrix(exp_id: str):
    """获取指定实验的混淆矩阵图片"""
    matrix_path = get_evaluation_dir(exp_id) / "confusion_matrix.png"
    
    if not matrix_path.exists():
        raise HTTPException(status_code=404, detail=f"{exp_id} 混淆矩阵图片不存在")
    
    return FileResponse(
        path=str(matrix_path),
        media_type="image/png",
        filename=f"{exp_id}_confusion_matrix.png"
    )


@router.get("/{exp_id}/predictions", response_model=PredictionSummary)
async def get_exp_predictions(exp_id: str):
    """获取指定实验的预测结果统计"""
    predictions_path = get_evaluation_dir(exp_id) / f"predictions_{exp_id}.csv"
    
    if not predictions_path.exists():
        raise HTTPException(status_code=404, detail=f"{exp_id} 预测结果文件不存在")
    
    try:
        df = pd.read_csv(predictions_path)
        
        total_predictions = len(df)
        
        if 'predicted_mode' in df.columns:
            mode_distribution = df['predicted_mode'].value_counts().to_dict()
        else:
            mode_distribution = {}
        
        if 'accuracy' in df.columns:
            accuracy = df['accuracy'].iloc[0] if len(df) > 0 else 0.0
        else:
            accuracy = 0.0
        
        return PredictionSummary(
            total_predictions=total_predictions,
            mode_distribution=mode_distribution,
            accuracy=float(accuracy)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取预测结果失败: {str(e)}")


@router.get("/{exp_id}/error-analysis")
async def get_exp_error_analysis(exp_id: str):
    """获取指定实验的错误分析"""
    error_path = get_evaluation_dir(exp_id) / "error_analysis.csv"
    
    if not error_path.exists():
        raise HTTPException(status_code=404, detail=f"{exp_id} 错误分析文件不存在")
    
    try:
        df = pd.read_csv(error_path)
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取错误分析失败: {str(e)}")
