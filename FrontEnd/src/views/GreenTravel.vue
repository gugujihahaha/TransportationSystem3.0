<template>
  <div class="cyber-battle-station">
    
    <div class="side-panel left-panel">
      
      <div class="cyber-card header-card">
        <div class="cyber-title-box">
          <h2 class="neon-text-green"> 个人低碳出行泛化验证控制台</h2>
        </div>
        <div class="sys-status blink-status-green"></div>
      </div>

      <div class="cyber-card content-card flex-col">
        <div class="module-box">
          <div class="module-header">
            <span class="bracket">[</span> <span class="neon-text-green-sub">01</span> <span class="bracket">]</span> 历史轨迹泛化验证
          </div>
          <el-upload 
            class="neon-upload-area green-border" 
            drag 
            action="#" 
            :auto-upload="false" 
            :show-file-list="false" 
            :on-change="handleFileUpload"
            :disabled="isAnalyzing"
          >
            <div class="upload-content">
              <el-icon class="upload-icon green-icon"><UploadFilled /></el-icon>
              <div class="upload-text">推入轨迹矩阵进行碳核算</div>
              <div class="upload-sub">CSV / PLT (自动调用预测)</div>
            </div>
          </el-upload>
        </div>

        <div class="module-box" :class="{ 'disabled-mask': !hasData }">
          <div class="module-header">
            <span class="bracket">[</span> <span class="neon-text-green-sub">02</span> <span class="bracket">]</span> 减排量化核算指标
          </div>
          <div class="carbon-grid">
            <div class="carbon-item">
              <div class="item-icon green-icon"><el-icon><Guide /></el-icon></div>
              <div class="item-data">
                <div class="val text-green">{{ stats.greenDistance }} <span class="unit">km</span></div>
                <div class="label">识别绿色里程</div>
              </div>
            </div>
            <div class="carbon-item">
              <div class="item-icon blue-icon"><el-icon><WindPower /></el-icon></div>
              <div class="item-data">
                <div class="val text-blue">{{ stats.co2Saved }} <span class="unit">kg</span></div>
                <div class="label">累计减少 CO₂</div>
              </div>
            </div>
          </div>
        </div>

        <div class="module-box flex-1" :class="{ 'disabled-mask': !hasData }">
          <div class="module-header">
            <span class="bracket">[</span> <span class="neon-text-green-sub">03</span> <span class="bracket">]</span> 模态锁定与环境效益
          </div>
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-val" :style="{ color: getModeColor(currentMode), textShadow: `0 0 15px ${getModeColor(currentMode)}` }">
                {{ currentMode }}
              </div>
              <div class="metric-label">出行模态判定</div>
            </div>
            <div class="metric-card tree-highlight">
              <div class="metric-val text-green">{{ stats.treesPlanted }}<span class="unit">棵</span></div>
              <div class="metric-label">相当于种树</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="center-panel">
      <div class="map-frame green-frame" v-loading="isAnalyzing" element-loading-text="时空特征提取与神经推理中..." element-loading-background="rgba(2, 11, 24, 0.85)">
        <div ref="mapContainer" class="leaflet-map"></div>
        
        <div class="frame-corner top-left green-line"></div>
        <div class="frame-corner top-right green-line"></div>
        <div class="frame-corner bottom-left green-line"></div>
        <div class="frame-corner bottom-right green-line"></div>

        <div class="map-legend">
          <div class="legend-item"><span class="line green-glow"></span> 绿色出行 (公交/地铁/骑行/步行)</div>
          <div class="legend-item"><span class="line red-glow"></span> 高碳出行 (私家车/出租车)</div>
        </div>
      </div>
    </div>

    <div class="side-panel right-panel">
      <div class="cyber-card ai-card green-border">
        <div class="ai-header green-bg-soft">
          <div class="ai-title">
            <el-icon class="pulse-icon-green"><Tickets /></el-icon> 星火 AI 专属表扬信
          </div>
          <el-button type="success" size="small" class="neon-btn-green" @click="exportToPDF" :disabled="!aiReport || isGeneratingReport">
            📸 分享海报
          </el-button>
        </div>
        
        <div class="ai-body" ref="scrollBox">
          <div v-if="!aiReport && !isGeneratingReport" class="ai-placeholder blink-text">
            [ 等待数据流注入计算环境效益 ]
          </div>
          <div v-if="isGeneratingReport && !aiReport" class="ai-processing green-text">
            <div class="spinner green-spinner"></div>
            正在定制你的环保文案...
          </div>
          <div v-else-if="aiReport" class="ai-stream-text">
            <span v-html="formattedReport"></span>
            <span v-if="isGeneratingReport" class="type-cursor green-cursor"></span>
          </div>
        </div>
      </div>
    </div>

    <div class="pdf-export-wrapper">
      <div id="green-pdf-poster" class="pdf-poster">
        <div class="poster-header">
          <div class="poster-title">🌱 个人低碳出行认证证书</div>
          <div class="poster-subtitle">
            <span>环保达人：<strong>{{ userName }}</strong></span>
            <span>认证日期：{{ currentDate }}</span>
          </div>
        </div>
        
        <div class="poster-stats">
          <div class="stat-box">
            <div class="stat-value">{{ currentMode }}</div>
            <div class="stat-label">出行模态判定</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">{{ stats.greenDistance }} <small>km</small></div>
            <div class="stat-label">绿色里程</div>
          </div>
          <div class="stat-box">
            <div class="stat-value" style="color: #67C23A;">{{ stats.co2Saved }} <small>kg</small></div>
            <div class="stat-label">碳排放减免</div>
          </div>
        </div>

        <div class="poster-ai-section">
          <div class="ai-badge">AI 专属寄语</div>
          <div class="pdf-ai-text" v-html="formattedReport"></div>
        </div>

        <div class="poster-footer">
          <p>由 TrafficRec 多模态深度学习系统 × 讯飞星火大模型 联合生成</p>
          <div class="watermark">Keep Green, Keep Going!</div>
        </div>
      </div>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, nextTick, computed } from 'vue'
import { useRoute } from 'vue-router'
import { Compass, Bicycle, UploadFilled, Guide, WindPower, Sunny, Tickets } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { trajectoryApi } from '../api/trajectory'
import { useAuthStore } from '@/stores/auth'
import html2pdf from 'html2pdf.js'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const route = useRoute()
const authStore = useAuthStore()
const userName = computed(() => authStore.username || '首席交通体验官')
const currentDate = new Date().toLocaleDateString()

const hasData = ref(false)
const isAnalyzing = ref(false)
const isGeneratingReport = ref(false)
const aiReport = ref('')
const currentMode = ref('--')
const scrollBox = ref<HTMLElement | null>(null)

let map: L.Map | null = null
let currentPolyline: L.Polyline | null = null
let markers: L.Layer[] = []
const mapContainer = ref<HTMLElement | null>(null)

const stats = reactive({
  greenDistance: '0.00',
  co2Saved: '0.00',
  treesPlanted: '0.0'
})

const formattedReport = computed(() => {
  if (!aiReport.value) return ''
  return aiReport.value
    .replace(/### (.*?)(?:\n|$)/g, '<h3 style="color: #39FF14; margin: 12px 0 8px; font-size: 16px;">▌ $1</h3>')
    .replace(/## (.*?)(?:\n|$)/g, '<h2 style="color: #E2E8F0; margin: 16px 0 10px; font-size: 18px; border-bottom: 1px solid rgba(57, 255, 20, 0.2); padding-bottom: 5px;">$1</h2>')
    .replace(/\*\*(.*?)\*\*/g, '<strong style="color: #39FF14;">$1</strong>')
    .replace(/\n/g, '<br/>')
})

const translateMode = (mode: string) => {
  const map: Record<string, string> = { 'car': '私家车', 'bus': '公交车', 'walk': '步行', 'bike': '自行车', 'subway': '地铁', 'train': '火车' };
  return map[mode.toLowerCase()] || mode;
}

// ================= 环境效益评估计算逻辑 =================
const evaluateEnvironmentalImpact = (mode: string, distanceKm: number) => {
  const lowerMode = mode.toLowerCase();
  const cnMode = translateMode(mode);

  const isGreen = ['walk', 'bike', 'bus', 'subway', '步行', '自行车', '公交车', '地铁'].includes(lowerMode) || 
                  ['步行', '自行车', '公交车', '地铁'].includes(cnMode);

  const calcGreenDist = isGreen ? distanceKm.toFixed(2) : '0.00';

  const carbonFactors: Record<string, number> = {
    'walk': 0.0, 'bike': 0.0, '步行': 0.0, '自行车': 0.0,
    'bus': 0.06, 'subway': 0.04, '公交车': 0.06, '地铁': 0.04,
    'car': 0.24, 'taxi': 0.24, '私家车': 0.24, '出租车': 0.24,
  };

  const baselineEmission = 0.24; 
  const currentFactor = carbonFactors[lowerMode] ?? carbonFactors[cnMode] ?? baselineEmission;

  let saved = (baselineEmission - currentFactor) * distanceKm;
  if (saved < 0 || !isGreen) saved = 0; 
  const calcCo2 = saved.toFixed(2);

  // 相当于种树棵数 = 减排量 * 0.05
  const calcTrees = (saved * 0.05).toFixed(1);

  return { isGreen, calcGreenDist, calcCo2, calcTrees };
}
// =================================================================

const handleFileUpload = async (uploadFile: any) => {
  const file = uploadFile.raw;
  if (!file) return;

  isAnalyzing.value = true;
  hasData.value = false;
  aiReport.value = '';
  
  try {
    const response = await trajectoryApi.predict(file, 'exp1', 'green'); 
    const result = response as any; 
    
    const mode = result.predicted_mode || 'unknown';
    currentMode.value = translateMode(mode);
    const distanceMeters = result.stats?.distance || 0;
    const distanceKm = distanceMeters / 1000; 

    const { isGreen, calcGreenDist, calcCo2, calcTrees } = evaluateEnvironmentalImpact(mode, distanceKm);

    stats.greenDistance = calcGreenDist;
    stats.co2Saved = calcCo2;
    stats.treesPlanted = calcTrees;

    hasData.value = true;
    
    if (result.points && result.points.length > 0) {
      renderLeafletTrajectory(result.points, isGreen);
    } else {
      ElMessage.warning('未能解析出坐标数据，无法在地图上绘制');
    }

    generateAIReport(calcGreenDist, calcCo2, calcTrees, currentMode.value);
    ElMessage.success(`绿色出行核算完毕！`);

  } catch (error) {
    ElMessage.error('模型推理失败，请检查服务状态');
  } finally {
    isAnalyzing.value = false;
  }
}

const restoreHistoryData = async (id: string) => {
  isAnalyzing.value = true;
  hasData.value = false;
  try {
    const record = await trajectoryApi.getHistoryById(id);
    const mode = record.predicted_mode || 'unknown';
    currentMode.value = translateMode(mode);
    
    const distanceKm = record.distance ? record.distance : 0;
    
    const { isGreen, calcGreenDist, calcCo2, calcTrees } = evaluateEnvironmentalImpact(mode, distanceKm);

    stats.greenDistance = calcGreenDist;
    stats.co2Saved = calcCo2;
    stats.treesPlanted = calcTrees;
    hasData.value = true;

    if (record.points && record.points !== '[]') {
      const pts = typeof record.points === 'string' ? JSON.parse(record.points) : record.points;
      renderLeafletTrajectory(pts, isGreen);
    }

    generateAIReport(calcGreenDist, calcCo2, calcTrees, currentMode.value);
    ElMessage.success(`时空档案还原成功！`);
  } catch (error) {
    ElMessage.error('无法读取历史档案数据');
  } finally {
    isAnalyzing.value = false;
  }
}

const generateAIReport = async (dist: string, co2: string, trees: string, modeName: string) => {
  isGeneratingReport.value = true
  aiReport.value = ''
  
  try {
    await trajectoryApi.streamReport(
      { model_id: 'exp1', mode: modeName, confidence: '95.0', scene: 'green', distance: dist, co2: co2 },
      (text) => {
        aiReport.value = text;
        nextTick(() => { if (scrollBox.value) scrollBox.value.scrollTop = scrollBox.value.scrollHeight })
      },
      () => { isGeneratingReport.value = false; },
      (error) => { console.error(error); isGeneratingReport.value = false; }
    );
  } catch (error) { isGeneratingReport.value = false; }
}

const exportToPDF = () => {
  const element = document.getElementById('green-pdf-poster')
  if (!element) return
  // 修复了 TS 报错
  const opt = {
    margin: 0,
    filename: `${userName.value}的绿色出行认证.pdf`,
    image: { type: 'jpeg' as const, quality: 1.0 },
    html2canvas: { scale: 3, useCORS: true, backgroundColor: '#f0f4f8' },
    jsPDF: { unit: 'px' as const, format: [800, 1000] as [number, number], orientation: 'portrait' as const }
  };
  ElMessage.success('正在为您生成专属纪念海报...')
  html2pdf().set(opt).from(element).save()
}

const initLeafletMap = () => {
  if (!mapContainer.value) return
  map = L.map(mapContainer.value, { zoomControl: false }).setView([39.9042, 116.4074], 12)
  L.tileLayer('http://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}', {
    attribution: '© AutoNavi', maxZoom: 19 
  }).addTo(map)
}

const renderLeafletTrajectory = (points: any[], isGreen: boolean) => {
  if (!map) return
  if (currentPolyline) map.removeLayer(currentPolyline)
  markers.forEach(m => map!.removeLayer(m))
  markers = []
  const latLngs = points.map(p => [p.lat || p.latitude, p.lng || p.longitude]).filter(p => p[0] && p[1]) as [number, number][];
  if (latLngs.length === 0) return;
  const lineColor = isGreen ? '#39FF14' : '#B86D6D';
  currentPolyline = L.polyline(latLngs, { color: lineColor, weight: 6, opacity: 0.85, lineCap: 'round', lineJoin: 'round'}).addTo(map);
  const startMarker = L.circleMarker(latLngs[0] as L.LatLngExpression, { radius: 6, fillColor: '#67C23A', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(map);
  const endMarker = L.marker(latLngs[latLngs.length - 1] as L.LatLngExpression, {
    icon: L.divIcon({ className: 'custom-end-marker', html: `<div style="width: 12px; height: 12px; background: ${lineColor}; border: 2px solid #fff; border-radius: 2px;"></div>`, iconSize: [12, 12], iconAnchor: [6, 6] })
  }).addTo(map);
  markers.push(startMarker, endMarker);
  map.fitBounds(currentPolyline.getBounds(), { padding: [50, 50], animate: true, duration: 1 });
}

const getModeColor = (mode: string) => {
  if (['私家车', '出租车', 'CAR', 'TAXI'].includes(mode)) return '#FF3366';
  return '#39FF14';
}

onMounted(() => { 
  nextTick(() => { initLeafletMap() }) 
  if (route.query.id) restoreHistoryData(route.query.id as string)
})
onUnmounted(() => { if (map) { map.remove(); map = null } })
</script>

<style scoped>
/* ================= 全局三栏严格物理布局 ================= */
.cyber-battle-station {
  display: flex;
  width: 100%;
  height: calc(100vh - 64px);
  background: #0F172A;
  padding: 16px;
  gap: 16px;
  box-sizing: border-box;
  font-family: 'Helvetica Neue', Helvetica, 'Microsoft YaHei', sans-serif;
}

.side-panel { display: flex; flex-direction: column; gap: 16px; flex-shrink: 0; }
.left-panel { width: 340px; }
.right-panel { width: 420px; }
.center-panel { flex: 1; display: flex; position: relative; }

/* 通用卡片 */
.cyber-card { 
  background: #1E293B; 
  border: 1px solid #334155; 
  border-radius: 8px; 
  display: flex; 
  flex-direction: column; 
  overflow: hidden; 
}

.header-card-dark { padding: 16px; background: #0F172A; border-bottom: 1px solid #334155; }
.neon-text-green { margin: 10px; font-size: 20px; font-weight: 800; color: #4fc76dc1; text-shadow: 0 0 10px rgba(57, 255, 20, 0.4); }
.neon-text-green-sub { color: #98dc96c1; font-weight: bold; }
.sub-title { font-size: 11px; color: #94a3b8; margin-top: 4px; }
.sys-status { width: 10px; height: 10px; border-radius: 50%; background: #a1dc96c1; box-shadow: 0 0 10px #39FF14; }
.blink-status-green { animation: breathe-green 2s infinite; }

.content-card { padding: 20px; flex: 1; overflow-y: auto; }
.flex-col { display: flex; flex-direction: column; gap: 20px; }
.flex-1 { flex: 1; }

.module-header { font-size: 13px; color: #cdd9e5; margin-bottom: 16px; }
.bracket { color: #888; font-family: monospace; }
.disabled-mask { opacity: 0.3; pointer-events: none; filter: grayscale(1); transition: 0.3s; }

/* 拖拽上传 */
:deep(.el-upload-dragger) {
  background: rgba(57, 255, 20, 0.02) !important;
  border: 1px dashed rgba(57, 255, 20, 0.3) !important;
  border-radius: 8px; padding: 24px 0;
}
:deep(.el-upload-dragger:hover) { border-color: #a1dc96c1 !important; background: rgba(57, 255, 20, 0.05) !important; }
.green-icon { color: #a1dc96c1; }
.upload-icon { font-size: 36px; margin-bottom: 12px; }
.upload-text { font-size: 14px; color: #fff; font-weight: 500; }
.upload-sub { font-size: 12px; color: #8892b0; margin-top: 4px; }

/* 减排指标 */
.carbon-grid { display: flex; flex-direction: column; gap: 12px; }
.carbon-item {
  display: flex; align-items: center; gap: 16px;
  background: rgba(0, 0, 0, 0.3); padding: 12px 16px; border-radius: 6px;
}
.item-icon { font-size: 24px; }
.blue-icon { color: #00F0FF; }
.item-data .val { font-size: 22px; font-weight: bold; font-family: monospace; }
.item-data .label { font-size: 11px; color: #94a3b8; margin-top: 2px; }
.unit { font-size: 12px; margin-left: 2px; }
.text-green { color: #54d55ad7; }
.text-blue { color: #00F0FF; }

/* 特征看板 */
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.metric-card {
  background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(57, 255, 20, 0.15);
  border-radius: 8px; padding: 20px 10px; text-align: center;
}
.tree-highlight { border-color: #a1dc96c1; background: rgba(57, 255, 20, 0.05); }
.metric-val { color: #54d55ad7;font-size: 24px; font-weight: bold; font-family: monospace; }
.metric-label { font-size: 11px; color: #8892b0; margin-top: 10px; }

/* ================= 中间：地图区域 ================= */
.map-frame {
  flex: 1; position: relative; border-radius: 12px;
  border: 2px solid rgba(57, 255, 20, 0.1);
  overflow: hidden;
}
.leaflet-map { width: 100%; height: 100%; background: #fff; z-index: 1; }

/* 地图图例 */
.map-legend {
  position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
  background: rgba(2, 11, 22, 0.85); padding: 12px 24px; border-radius: 30px;
  border: 1px solid rgba(57, 255, 20, 0.3); z-index: 1000; display: flex; gap: 24px;
}
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #fff; }
.line { width: 16px; height: 4px; border-radius: 2px; }
.green-glow { background: #39FF14; box-shadow: 0 0 8px #39FF14; }
.red-glow { background: #FF3366; box-shadow: 0 0 8px #FF3366; }

/* 边角装饰 */
.frame-corner { position: absolute; width: 20px; height: 20px; border: 2px solid transparent; z-index: 10; }
.green-line { border-color: #a1dc96c1; }
.top-left { top: 0; left: 0; border-right: none; border-bottom: none; }
.top-right { top: 0; right: 0; border-left: none; border-bottom: none; }
.bottom-left { bottom: 0; left: 0; border-right: none; border-top: none; }
.bottom-right { bottom: 0; right: 0; border-left: none; border-top: none; }

/* ================= 右侧：AI 报告 ================= */
.ai-header {
  padding: 16px 20px; border-bottom: 1px solid rgba(57, 255, 20, 0.2);
  display: flex; justify-content: space-between; align-items: center;
}
.green-bg-soft { background: linear-gradient(90deg, rgba(57, 255, 20, 0.1), transparent); }
.ai-title { font-size: 16px; font-weight: bold; color: #a1dc96c1; display: flex; align-items: center; gap: 8px; }
.neon-btn-green { background: transparent !important; border: 1px solid #37ff1484 !important; color: #37ff14ca !important; font-weight: bold;}
.neon-btn-green:hover:not(:disabled) { background: #a1dc96c1 !important; color: #020B16 !important; }

.ai-body { padding: 24px; flex: 1; overflow-y: auto; }
.ai-placeholder { text-align: center; color: #8892b0; margin-top: 40px; font-style: italic; }
.ai-processing { display: flex; flex-direction: column; align-items: center; gap: 16px; margin-top: 40px; }
.green-spinner { border-top-color: #a1dc96c1; }

.ai-stream-text { color: #e2e8f0; line-height: 1.8; font-size: 14.5px; }
.green-cursor { display: inline-block; width: 8px; height: 16px; background: #a1dc96c1; margin-left: 6px; animation: blink 1s infinite; }

@keyframes breathe-green { 0%, 100% { opacity: 1; box-shadow: 0 0 10px #a1dc96c1; } 50% { opacity: 0.6; box-shadow: 0 0 20px #a1dc96f4; } }
@keyframes blink { 50% { opacity: 0; } }

.pdf-export-wrapper { position: absolute; top: -9999px; left: -9999px; }
</style>