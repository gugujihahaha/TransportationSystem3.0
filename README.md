基于您提供的代码和项目配置文件，我为您重新生成了一份更全面、更符合当前代码实际运行情况的 `README.md` 文档。这份文档整合了您原有的说明、前端真实的依赖库（如 DataV、Turf.js 等）以及后端主程序（如权限认证、模型训练路由、自动加载机制）的最新进展。

您可以直接复制以下 Markdown 内容作为项目的 `README.md`：

```markdown
# 城市出行智能感知平台 (TransportationSystem 3.0)

基于深度学习的城市出行方式识别系统，支持多种轨迹数据格式上传和实时预测。

## 🌟 核心特性

### 前端功能 (FrontEnd)
* **地图分析与可视化**：支持轨迹文件上传，实时预测并在基于 Leaflet 的地图上生成可视化轨迹与热力图。
* **多维度数据概览**：集成 ECharts 与 DataV 提供炫酷的数据集统计大屏和特征维度说明。
* **模型对比与分析**：可直观对比四个不同特征维度的实验模型（LSTM网络）的训练结果和性能指标。
* **报表导出**：支持通过 html2pdf.js 一键导出相关分析报告。

### 后端服务 (BackEnd)
* **RESTful API**：基于 FastAPI 构建，提供包含轨迹处理 (`trajectory`)、实验预测 (`experiments`)、数据集管理 (`dataset`)、在线训练 (`training`) 及用户认证 (`auth`) 在内的完整接口支持。
* **智能启动机制**：服务启动时会自动加载并初始化外部依赖数据，包括 OSM（OpenStreetMap）路网数据、天气数据及深度学习预测模型。
* **多格式轨迹解析**：原生支持 CSV、JSON、PLT (Geolife 标准) 格式的轨迹文件上传。

---

## 🛠 技术栈

### 前端 (Vue 3 生态)
* **核心框架**: Vue 3 (Composition API), TypeScript, Vite 构建工具。
* **状态与路由**: Pinia, Vue Router。
* **UI与可视化**: Element Plus, ECharts, @kjgl77/datav-vue3。
* **GIS与地图**: Leaflet, Leaflet.heat, @turf/turf (地理空间分析)。

### 后端 (Python 生态)
* **Web 框架**: FastAPI, Uvicorn (ASGI)。
* **深度学习**: PyTorch (构建 LSTM 等序列模型)。
* **数据处理**: Pandas, NumPy。
* **数据库持久化**: SQLAlchemy (通过 `api.database` 管理数据库引擎与模型生成)。

---

## 📂 项目结构

```text
城市出行智能感知平台/
├── BackEnd/                 # 后端服务
│   ├── api/                # FastAPI 后端接口
│   │   ├── routers/        # 包含 auth, dataset, experiment, training, trajectory 路由
│   │   ├── main.py         # FastAPI 主入口及生命周期管理
│   │   ├── database.py     # 数据库连接池与配置
│   │   ├── models.py       # 数据库 ORM 模型 (如 User)
│   │   └── schemas.py      # Pydantic 数据验证模型
│   ├── common/             # 通用数据清洗与工具类
│   ├── exp1/               # 实验1：基础9维轨迹特征模型
│   ├── exp2/               # 实验2：9维特征 + OSM 路网映射
│   ├── exp3/               # 实验3：9维特征 + 气象数据融合
│   ├── exp4/               # 实验4：特征全融合 (轨迹+OSM+天气)
│   └── scripts/            # 数据爬取与预处理脚本
├── FrontEnd/               # 前端应用
│   ├── src/
│   │   ├── views/          # 视图层 (主页、大屏、历史记录等)
│   │   ├── components/     # 复用组件
│   │   ├── stores/         # Pinia 状态管理
│   │   ├── api/            # Axios API 封装接口
│   │   └── types/          # TS 接口定义
│   └── package.json        # 依赖与脚本配置
└── data/                   # 本地持久化数据目录（被 .gitignore 忽略）
```
**

---

## 🚀 快速开始

### 1. 环境准备
* **Python**: >= 3.8 (推荐使用 conda 管理环境)
* **Node.js**: >= 20.19.0 (项目 `package.json` 引擎限制)

### 2. 后端服务启动

推荐使用 Conda 创建隔离环境：
```bash
conda create -n TraeAI-7 python=3.11
conda activate TraeAI-7
```

安装依赖并启动：
```bash
cd BackEnd
pip install -r requirements_api.txt
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
> **注意**：服务启动时控制台会输出加载日志，自动完成数据库表初始化、OSM和天气数据加载及模型权重加载。后端服务将在 `http://localhost:8000` 提供，API 文档访问 `http://localhost:8000/docs`。跨域已默认放行 `localhost:5173` 和 `localhost:3000`。

### 3. 前端界面启动

```bash
cd FrontEnd
npm install
npm run dev
```
前端应用将在 `http://localhost:5173` 启动。可通过 `npm run build` 打包生产环境代码。

---

## 📖 核心功能说明

### 轨迹文件上传支持
系统原生支持三种轨迹数据输入方式：
1. **CSV 格式** (包含 `latitude`, `longitude`, `timestamp` 字段)
2. **JSON 格式** (对象数组，包含上述三个键值)
3. **PLT 格式** (微软 Geolife 项目原始数据集格式)
**

### 模型实验与选择
平台在地图分析页面提供四个深度的实验模型用于交通方式推理：

| 实验代号 | 接入特征组合 | 核心模型结构 | 损失函数 (Loss) |
|------|----------|----------|----------|
| **exp1** | 9维轨迹基础特征 | LSTM | CrossEntropy |
| **exp2** | 9维特征 + **OSM 空间信息** | LSTM | CrossEntropy |
| **exp3** | 9维特征 + **天气时序信息** | LSTM + Weather | Focal Loss |
| **exp4** | 9维特征 + **OSM** + **天气** | LSTM + Weather | LabelSmoothingFocalLoss |

各实验预训练权重路径位于 `BackEnd/exp*/checkpoints/exp*_model.pth`，由 `main.py` 在启动阶段统一载入。

---

## 📄 许可证

本项目仅供学术学习和科学研究使用，请勿用于商业生产环境。
```