from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import uuid
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import TrainingRequest, TrainingResponse, TrainingProgress

router = APIRouter()

training_tasks: Dict[str, Dict] = {}
active_connections: Dict[str, WebSocket] = {}


async def simulate_training(task_id: str, exp_name: str, epochs: int):
    """模拟训练过程（实际应调用真实的训练脚本）"""
    try:
        training_tasks[task_id]["status"] = "training"
        
        for epoch in range(1, epochs + 1):
            if training_tasks[task_id]["status"] == "cancelled":
                break
            
            await asyncio.sleep(0.5)
            
            loss = max(0.1, 1.0 - (epoch / epochs) * 0.8 + np.random.uniform(-0.05, 0.05))
            accuracy = min(0.95, 0.5 + (epoch / epochs) * 0.4 + np.random.uniform(-0.02, 0.02))
            
            progress = TrainingProgress(
                task_id=task_id,
                exp_name=exp_name,
                epoch=epoch,
                total_epochs=epochs,
                loss=float(loss),
                accuracy=float(accuracy),
                status="training"
            )
            
            training_tasks[task_id]["progress"] = progress
            
            if task_id in active_connections:
                ws = active_connections[task_id]
                try:
                    await ws.send_json(progress.dict())
                except:
                    pass
        
        if training_tasks[task_id]["status"] != "cancelled":
            training_tasks[task_id]["status"] = "completed"
            
            final_progress = TrainingProgress(
                task_id=task_id,
                exp_name=exp_name,
                epoch=epochs,
                total_epochs=epochs,
                loss=0.15,
                accuracy=0.88,
                status="completed"
            )
            
            if task_id in active_connections:
                ws = active_connections[task_id]
                try:
                    await ws.send_json(final_progress.dict())
                except:
                    pass
    
    except Exception as e:
        training_tasks[task_id]["status"] = "failed"
        training_tasks[task_id]["error"] = str(e)


@router.post("/start/{exp_name}", response_model=TrainingResponse)
async def start_training(exp_name: str, request: TrainingRequest, background_tasks: BackgroundTasks):
    """启动指定实验的训练"""
    if exp_name not in ["exp1", "exp2", "exp3", "exp4"]:
        raise HTTPException(status_code=400, detail="无效的实验名称")
    
    if exp_name == "exp1":
        return TrainingResponse(
            task_id="",
            status="already_completed",
            message="exp1 已经完成训练"
        )
    
    task_id = str(uuid.uuid4())
    
    training_tasks[task_id] = {
        "exp_name": exp_name,
        "status": "pending",
        "epochs": request.epochs,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate,
        "progress": None,
        "error": None
    }
    
    background_tasks.add_task(simulate_training, task_id, exp_name, request.epochs)
    
    return TrainingResponse(
        task_id=task_id,
        status="started",
        message=f"训练任务已启动: {exp_name}"
    )


@router.get("/tasks")
async def get_training_tasks():
    """获取所有训练任务状态"""
    tasks_info = []
    for task_id, task_info in training_tasks.items():
        tasks_info.append({
            "task_id": task_id,
            "exp_name": task_info["exp_name"],
            "status": task_info["status"],
            "progress": task_info["progress"].dict() if task_info["progress"] else None,
            "error": task_info["error"]
        })
    
    return {"tasks": tasks_info}


@router.get("/tasks/{task_id}")
async def get_training_task(task_id: str):
    """获取指定训练任务的状态"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_info = training_tasks[task_id]
    
    return {
        "task_id": task_id,
        "exp_name": task_info["exp_name"],
        "status": task_info["status"],
        "progress": task_info["progress"].dict() if task_info["progress"] else None,
        "error": task_info["error"]
    }


@router.delete("/tasks/{task_id}")
async def cancel_training(task_id: str):
    """取消训练任务"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_info = training_tasks[task_id]
    
    if task_info["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="任务已结束，无法取消")
    
    training_tasks[task_id]["status"] = "cancelled"
    
    return {"message": "训练任务已取消"}


@router.websocket("/ws/training/{task_id}")
async def websocket_training(websocket: WebSocket, task_id: str):
    """WebSocket连接，实时推送训练进度"""
    await websocket.accept()
    
    active_connections[task_id] = websocket
    
    try:
        if task_id in training_tasks:
            task_info = training_tasks[task_id]
            if task_info["progress"]:
                await websocket.send_json(task_info["progress"].dict())
        
        while True:
            await asyncio.sleep(1)
            
            if task_id in training_tasks and training_tasks[task_id]["status"] in ["completed", "failed", "cancelled"]:
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        if task_id in active_connections:
            del active_connections[task_id]
