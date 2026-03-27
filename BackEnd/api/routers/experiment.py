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
        "name": "实验1: 纯轨迹特征",
        "description": "仅使用GPS轨迹的9维特征进行交通方式识别",
        "features": ["轨迹特征 (9维)", "段级统计特征 (18维)"],
        "status": "completed"
    },
    {
        "id": "exp2",
        "name": "实验2: 轨迹 + OSM空间特征",
        "description": "结合GPS轨迹和OpenStreetMap地理空间特征",
        "features": ["轨迹特征 (9维)", "段级统计特征 (18维)", "OSM空间特征 (11维)"],
        "status": "completed"
    },
    {
        "id": "exp3",
        "name": "实验3: 轨迹 + 空间 + 天气特征",
        "description": "结合GPS轨迹、OSM空间特征和气象数据",
        "features": ["轨迹特征 (9维)", "段级统计特征 (18维)", "OSM空间特征 (15维)", "天气特征 (12维)"],
        "status": "completed"
    },
    {
        "id": "exp4",
        "name": "实验4: 对比学习",
        "description": "使用Focal Loss和对比学习改进模型",
        "features": ["轨迹特征 (9维)", "段级统计特征 (18维)", "OSM空间特征 (15维)", "天气特征 (12维)", "对比学习"],
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
