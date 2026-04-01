from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from api.database import engine, Base
from api.routers import trajectory, experiment, dataset, training
from api.routers.auth import router as auth_router
from api.models import User
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="交通方式识别 API",
    description="基于深度学习的城市出行方式识别系统",
    version="1.0.0"
)

app = FastAPI(
    title="交通方式识别 API",
    description="基于深度学习的城市出行方式识别系统",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trajectory.router, prefix="/api/trajectory", tags=["轨迹"])
app.include_router(experiment.router, prefix="/api/experiments", tags=["实验"])
app.include_router(dataset.router, prefix="/api/dataset", tags=["数据集"])
app.include_router(training.router, prefix="/api/training", tags=["训练"])
app.include_router(auth_router, prefix="/api/auth", tags=["认证"])

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型和天气数据"""
    print("=" * 60)
    print("正在启动服务...")
    print("=" * 60)
    
    from api.routers.trajectory import load_predictors, load_weather_data, load_osm_data
    
    print("\n📋 加载OSM数据...")
    load_osm_data()
    
    print("\n📋 加载天气数据...")
    load_weather_data()
    
    print("\n📋 加载预测模型...")
    load_predictors()
    
    print("\n" + "=" * 60)
    print("服务启动完成！")
    print("=" * 60)


@app.get("/")
async def root():
    return {
        "message": "交通方式识别 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
