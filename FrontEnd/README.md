# 城市出行智能感知平台 - 前端 (FrontEnd)

本项目是基于 Vue 3 + TypeScript 构建的交通方式智能识别与分析平台前端。通过结合深度学习模型的后端 API，实现了轨迹数据可视化、模型性能对比、绿色出行分析及碳排放评估等功能。

## 🛠 技术栈

* **核心框架**: Vue 3 (Composition API)
* **开发语言**: TypeScript
* **构建工具**: Vite
* **状态管理**: Pinia
* **路由管理**: Vue Router
* **UI 组件库**: Element Plus
* **数据可视化**: ECharts (`vue-echarts`), DataV (`@kjgl77/datav-vue3`)
* **地图与GIS**: Leaflet, Leaflet.heat, Turf.js (`@turf/turf`)

## 📂 项目结构

结合前端实际业务，核心目录及 Vue 文件说明如下：

```text
FrontEnd/
├── public/                 # 静态资源与本地预设数据 (如 JSON 报表、Geo 数据)
├── src/
│   ├── api/                # 后端接口封装
│   │   ├── auth.ts         # 用户认证相关
│   │   ├── dataset.ts      # 数据集统计与获取
│   │   ├── experiment.ts   # 实验模型评估与报告
│   │   └── trajectory.ts   # 轨迹上传与解析
│   ├── assets/             # 样式文件与图片资源
│   ├── components/         # 全局复用组件
│   │   ├── AppLayout.vue   # 页面基础布局
│   │   ├── HomePage.vue    # 首页组件模块
│   │   ├── PageHeader.vue  # 顶部导航栏
│   │   └── icons/          # SVG 图标组件
│   ├── router/             # 路由配置中心 (index.ts)
│   ├── stores/             # Pinia 状态仓库 (auth, counter, experiment, trajectory)
│   ├── types/              # TypeScript 全局接口与类型声明
│   ├── utils/              # 通用工具函数 (如 colors.ts)
│   ├── views/              # 🌟 核心业务视图页面
│   │   ├── CarbonHeatmap.vue  # 碳排放热力图大屏
│   │   ├── DataInsight.vue    # 数据洞察与可视化统计
│   │   ├── GreenTravel.vue    # 绿色出行指标与分析
│   │   ├── HistoryView.vue    # 历史轨迹记录与回放
│   │   ├── HomeView.vue       # 主页 (主地图分析与预测)
│   │   ├── LoginView.vue      # 用户登录界面
│   │   ├── ModelCompare.vue   # 多维特征实验模型对比评估
│   │   ├── RegionDetail.vue   # 区域级出行特征详情
│   │   ├── TechSupport.vue    # 技术支持与架构说明
│   │   └── UserCenterView.vue # 用户中心管理
│   ├── App.vue             # 根组件
│   └── main.ts             # 挂载入口
├── package.json            # 项目依赖配置
└── vite.config.ts          # Vite 构建配置
**

🚀 快速开始
1. 环境要求
Node.js: ^20.19.0 或 >=22.12.0 (严格遵循 package.json 的引擎要求)

2. 安装依赖
进入 FrontEnd 目录，运行以下命令安装所需依赖：

Bash
npm install
3. 开发环境启动
Bash
npm run dev
项目默认将在 http://localhost:5173 启动，并支持热重载（HMR）。

注：确保后端服务（BackEnd）已在 localhost:8000 正常运行，否则前端的相关 API 请求将无法成功。

4. 生产环境构建
当开发完成需要部署时，可执行打包命令：

Bash
npm run build
这将会先执行 TypeScript 类型检查 (vue-tsc)，然后通过 Vite 将静态文件打包至 dist 目录中。

💡 核心功能模块提示
地图预测交互 (HomeView.vue): 支持用户上传 CSV/JSON/PLT 格式的轨迹，调用后端进行实时的交通方式判定。

模型对比评价 (ModelCompare.vue): 可直观查看不同特征融合（基础轨迹、附加路网OSM、附加气象数据等）下，深度学习预测模型的精度差异与评估报告。

报表下载 (TechSupport/ModelCompare 等): 集成了 html2pdf.js，支持将可视化图表导出为高质量 PDF 报告。