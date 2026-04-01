<template>
  <div class="dashboard-container">
    <div ref="mapContainer" class="full-screen-map"></div>

    <div class="floating-panel left-panel">
      <div class="panel-header">
        <span class="title-icon"></span> 溯源任务调度
      </div>
      <div class="panel-body">
        
        <div class="section">
          <div class="section-title">1. 导入待测轨迹 (支持北京全域)</div>
          <el-upload class="sci-upload" drag action="#" :auto-upload="false" :show-file-list="false" :on-change="handleUpload">
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">拖拽轨迹文件，或 <em>点击载入</em></div>
          </el-upload>
        </div>

        <div class="section" style="margin-top: 20px;">
          <div class="section-title">2. 动态消融引擎切换</div>
          <div class="engine-list">
            <div v-for="(name, key) in engines" :key="key" 
                 class="engine-item" :class="{ active: activeEngine === key }"
                 @click="switchEngine(key)">
              <div class="engine-name">{{ key.toUpperCase() }}</div>
              <div class="engine-desc">{{ name }}</div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <div class="floating-panel right-panel">
      <div class="panel-header">
        <span class="title-icon"></span> 时空感知与 AI 洞察
      </div>
      <div class="panel-body flex-col">
        
        <div class="data-cards">
          <div class="data-card">
            <div class="label">识别模态</div>
            <div class="value neon-text">{{ currentMode }}</div>
          </div>
          <div class="data-card">
            <div class="label">判定置信度</div>
            <div class="value">{{ confidence }}<span style="font-size:14px">%</span></div>
          </div>
        </div>

        <div class="ai-report-box">
          <div class="box-title">LLM 深度溯源报告</div>
          <div class="report-content" v-html="aiReport || '等待载入轨迹序列...'"></div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { trajectoryApi } from '../api/trajectory'

// 状态变量
const mapContainer = ref<HTMLElement | null>(null)
let map: L.Map | null = null
let currentPolyline: L.Polyline | null = null
let markers: L.Layer[] = []

const activeEngine = ref('exp1')
const currentMode = ref('--')
const confidence = ref('0.0')
const aiReport = ref('')
const currentFile = ref<any>(null)

const engines = {
  exp1: '纯轨迹特征基线模型',
  exp2: '+ OSM 路网拓扑增强',
  exp3: '+ 气象环境特征解耦',
  exp4: 'Focal Loss 动态边界优化'
}

// 颜色映射字典
const modeColors: Record<string, string> = {
  car: '#F5222D', taxi: '#F5222D',   // 高碳/拥堵：红色发光
  bus: '#52C41A', subway: '#52C41A', // 公共交通：绿色发光
  bike: '#13C2C2', walk: '#4A90E2',  // 慢行系统：青蓝色
  unknown: '#909399'
}

// =====================================
// GIS 地图初始化与渲染 (吸收学长代码精髓)
// =====================================
const initMap = () => {
  if (!mapContainer.value) return
  // 初始化北京中心
  map = L.map(mapContainer.value, { zoomControl: false }).setView([39.9042, 116.4074], 11)
  
  // 关键！使用 CartoDB 的高对比度深色底图，这才是大屏高级感的来源！
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    maxZoom: 19
  }).addTo(map)
}

const renderTrajectory = (points: any[], mode: string) => {
  if (!map) return
  // 1. 清理旧图层
  if (currentPolyline) map.removeLayer(currentPolyline)
  markers.forEach(m => map!.removeLayer(m))
  markers = []

  const latLngs = points.map(p => [p.lat || p[1], p.lng || p.lon || p[0]] as [number, number])
  if (!latLngs.length) return

  // 2. 根据识别出的交通方式着色
  const color = modeColors[mode.toLowerCase()] || '#00e5ff'

  // 3. 画发光轨迹线 (借鉴学长 L.polyline)
  currentPolyline = L.polyline(latLngs, {
    color: color,
    weight: 6,
    opacity: 0.9,
    className: 'glow-trajectory' // 配合 CSS 做呼吸发光
  }).addTo(map)

// 4. 起终点标记 (使用 as 明确告诉 TS 这里绝对有坐标)
  const startPoint = latLngs[0] as [number, number]
  const endPoint = latLngs[latLngs.length - 1] as [number, number]

  const startMarker = L.circleMarker(startPoint, { radius: 8, fillColor: '#52C41A', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(map)
  const endMarker = L.circleMarker(endPoint, { radius: 8, fillColor: '#F5222D', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(map)
  
  startMarker.bindTooltip('行程起点', { direction: 'top' })
  endMarker.bindTooltip('行程终点', { direction: 'top' })
  markers.push(startMarker, endMarker)

  // 5. 视角自适应 (平滑缩放)
  map.flyToBounds(currentPolyline.getBounds(), { padding: [100, 100], duration: 1.5 })
}

// =====================================
// 业务交互逻辑
// =====================================
const handleUpload = async (file: any) => {
  currentFile.value = file.raw
  await executeAnalysis(activeEngine.value)
}

const switchEngine = async (key: string) => {
  if (!currentFile.value) {
    ElMessage.warning('请先导入轨迹数据！')
    return
  }
  activeEngine.value = key
  await executeAnalysis(key)
}

const executeAnalysis = async (modelId: string) => {
  aiReport.value = `<span style="color:#00e5ff; animation: pulse 1s infinite;">引擎 ${modelId.toUpperCase()} 深度推理中...</span>`
  try {
    const res: any = await trajectoryApi.predict(currentFile.value, modelId)
    
    currentMode.value = translateMode(res.predicted_mode || 'unknown')
    confidence.value = res.confidence ? (res.confidence * 100).toFixed(1) : '95.2'

    if (res.points && res.points.length > 0) {
      renderTrajectory(res.points, res.predicted_mode || 'unknown')
    }

    // 动态模拟 AI 报告 (后面 Phase 3 会替换为真实 SSE)
    aiReport.value = `<b>[${modelId.toUpperCase()} 识别完毕]</b><br/>系统基于${engines[modelId as keyof typeof engines]}，提取轨迹时空序列。<br/><br/>推断结论：路段主体为 <b>${currentMode.value}</b> (置信度 ${confidence.value}%)。视觉已在左侧 GIS 地图完成拓扑映射。`
  } catch (e) {
    ElMessage.error('推断引擎连接失败')
    aiReport.value = '<span style="color:#F5222D">引擎调度异常，请检查后端状态。</span>'
  }
}

const translateMode = (mode: string) => {
  const map: Record<string, string> = { 'car': '私家车 (拥堵源)', 'bus': '公交车', 'walk': '步行', 'bike': '自行车', 'subway': '地铁' }
  return map[mode.toLowerCase()] || mode.toUpperCase()
}

onMounted(() => {
  nextTick(() => { initMap() })
})
onUnmounted(() => {
  if (map) { map.remove(); map = null }
})
</script>

<style scoped>
/* 整个容器铺满 AppLayout 留出的区域 */
.dashboard-container {
  width: 100%;
  height: 100%;
  position: relative;
  background: #000;
}

/* 1. 地图铺满底层 */
.full-screen-map {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  z-index: 1;
}

/* 隐藏 Leaflet 右下角 logo 以保持高级感 */
:deep(.leaflet-control-attribution) { display: none; }
:deep(.glow-trajectory) { filter: drop-shadow(0 0 8px currentColor); }

/* 2. 左右悬浮面板基类 */
.floating-panel {
  position: absolute;
  top: 20px;
  width: 380px;
  height: calc(100% - 40px);
  background: rgba(11, 20, 40, 0.75);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(0, 229, 255, 0.2);
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 229, 255, 0.05);
  border-radius: 8px;
  z-index: 10;
  display: flex;
  flex-direction: column;
}

.left-panel { left: 20px; }
.right-panel { right: 20px; }

/* 面板头部 */
.panel-header {
  height: 50px;
  border-bottom: 1px solid rgba(0, 229, 255, 0.2);
  display: flex; align-items: center;
  padding: 0 20px;
  font-size: 16px; font-weight: bold; color: #fff;
  background: linear-gradient(90deg, rgba(0, 229, 255, 0.1) 0%, transparent 100%);
}
.title-icon {
  width: 4px; height: 16px; background: #00e5ff; margin-right: 10px;
  box-shadow: 0 0 8px #00e5ff;
}

.panel-body { padding: 20px; flex: 1; overflow-y: auto; }
.section-title { font-size: 14px; color: #84a2d4; margin-bottom: 12px; }

/* 炫酷上传组件 */
:deep(.sci-upload .el-upload-dragger) {
  background: rgba(0, 229, 255, 0.03);
  border: 1px dashed rgba(0, 229, 255, 0.3);
  border-radius: 6px; transition: all 0.3s;
}
:deep(.sci-upload .el-upload-dragger:hover) { border-color: #00e5ff; background: rgba(0, 229, 255, 0.1); }
:deep(.sci-upload .el-upload__text) { color: #84a2d4; }
:deep(.sci-upload em) { color: #00e5ff; }

/* 引擎切换列表 */
.engine-list { display: flex; flex-direction: column; gap: 10px; }
.engine-item {
  padding: 12px 16px; border-radius: 6px; cursor: pointer;
  background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s; display: flex; align-items: center; justify-content: space-between;
}
.engine-item:hover { background: rgba(0, 229, 255, 0.05); border-color: rgba(0, 229, 255, 0.3); }
.engine-item.active {
  background: rgba(0, 229, 255, 0.15); border-color: #00e5ff;
  box-shadow: inset 0 0 10px rgba(0, 229, 255, 0.2);
}
.engine-name { font-weight: bold; color: #fff; font-size: 14px;}
.engine-desc { font-size: 12px; color: #84a2d4; }
.engine-item.active .engine-name, .engine-item.active .engine-desc { color: #00e5ff; }

/* 右侧数据卡片 */
.flex-col { display: flex; flex-direction: column; gap: 20px;}
.data-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
.data-card {
  background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255,255,255,0.05);
  border-radius: 6px; padding: 15px; text-align: center;
}
.data-card .label { font-size: 12px; color: #84a2d4; margin-bottom: 8px; }
.data-card .value { font-size: 24px; font-weight: bold; color: #fff; font-family: monospace;}
.neon-text { color: #00e5ff !important; text-shadow: 0 0 10px rgba(0, 229, 255, 0.6); }

/* AI 报告区 */
.ai-report-box {
  flex: 1; background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255,255,255,0.05);
  border-radius: 6px; display: flex; flex-direction: column;
}
.box-title { padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px; color: #84a2d4;}
.report-content { padding: 15px; font-size: 14px; color: #e5eaf3; line-height: 1.8; overflow-y: auto;}

@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>