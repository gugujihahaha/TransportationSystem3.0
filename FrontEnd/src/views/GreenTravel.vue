<template>
  <div class="green-container">
    <el-row :gutter="20" class="full-height">
      
      <el-col :span="14" class="full-height">
        <div class="panel dark-panel map-panel">
          <div class="panel-header">
            <div class="header-left">
              <el-icon><Compass /></el-icon> 跨城轨迹泛化识别与真实路网映射
            </div>
            <div class="header-actions">
              <el-tag size="small" type="success" effect="dark">驱动引擎: Exp1 纯轨迹基线模型</el-tag>
            </div>
          </div>
          <div class="panel-content map-wrapper" v-loading="isAnalyzing" element-loading-text="历史数据追溯与渲染中..." element-loading-background="rgba(11, 13, 18, 0.8)">
            <div ref="mapContainer" class="leaflet-map"></div>
            
            <div class="map-legend">
              <div class="legend-item"><span class="line green"></span> 绿色出行 (公交/地铁/骑行/步行)</div>
              <div class="legend-item"><span class="line red"></span> 高碳出行 (私家车/出租车)</div>
            </div>
          </div>
        </div>
      </el-col>

      <el-col :span="10" class="full-height">
        <div class="panel dark-panel control-panel">
          <div class="panel-header">
            <el-icon><Bicycle /></el-icon> 个人低碳出行看板
          </div>
          <div class="panel-content flex-col">
            
            <div class="section-box">
              <div class="box-title">1. 上传历史轨迹进行泛化验证</div>
              <el-upload
                class="trajectory-upload"
                drag
                action="#"
                :auto-upload="false"
                :show-file-list="false"
                :on-change="handleFileUpload"
                :disabled="isAnalyzing"
              >
                <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                <div class="el-upload__text">拖拽轨迹文件到此处，或 <em>点击上传</em></div>
                <template #tip>
                  <div class="el-upload__tip text-center">支持 .csv, .plt 格式 (系统将自动调用后端预测)</div>
                </template>
              </el-upload>
            </div>

            <div class="section-box" :class="{ 'blur-mask': !hasData }">
              <div class="box-title">2. 减排量化核算 (基于真实识别结果)</div>
              <div class="carbon-dashboard">
                <div class="carbon-card">
                  <div class="icon-wrap text-green"><el-icon><Guide /></el-icon></div>
                  <div class="data-wrap">
                    <div class="val text-green">{{ stats.greenDistance }} <span class="unit">km</span></div>
                    <div class="label">识别绿色里程</div>
                  </div>
                </div>
                <div class="carbon-card">
                  <div class="icon-wrap text-blue"><el-icon><WindPower /></el-icon></div>
                  <div class="data-wrap">
                    <div class="val text-blue">{{ stats.co2Saved }} <span class="unit">kg</span></div>
                    <div class="label">累计减少 CO₂</div>
                  </div>
                </div>
                <div class="carbon-card tree-card">
                  <div class="icon-wrap text-green"><el-icon><Sunny /></el-icon></div>
                  <div class="data-wrap">
                    <div class="val text-green">{{ stats.treesPlanted }} <span class="unit">棵</span></div>
                    <div class="label">相当于种树</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="section-box report-box flex-1" :class="{ 'blur-mask': !hasData }">
              <div class="box-title ai-title" style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                <div><el-icon><Tickets /></el-icon> 星火 AI 专属表扬信</div>
                <el-button type="success" size="small" @click="exportToPDF" :disabled="!aiReport || isGeneratingReport" style="background: #10b981; border: none; font-weight: bold;">
                  📸 生成分享海报 (PDF)
                </el-button>
              </div>
              <div class="ai-content" ref="scrollBox">
                <div v-if="!aiReport && !isGeneratingReport" class="empty-text">等待模型分析完成...</div>
                <div v-if="isGeneratingReport && !aiReport" class="empty-text" style="color: #4A90E2; animation: pulse 1s infinite;">
                  星火大模型正在为你定制文案...
                </div>
                <div v-else-if="aiReport" class="typing-text">
                  <span v-html="formattedReport"></span>
                  <span v-if="isGeneratingReport" class="typing-cursor">|</span>
                </div>
              </div>
            </div>

          </div>
        </div>
      </el-col>
    </el-row>

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
            <div class="stat-label">本次出行判定</div>
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
    .replace(/### (.*?)(?:\n|$)/g, '<h3 style="color: #4A90E2; margin: 12px 0 8px; font-size: 16px;">$1</h3>')
    .replace(/## (.*?)(?:\n|$)/g, '<h2 style="color: #333; margin: 16px 0 10px; font-size: 18px; border-bottom: 1px solid #eee; padding-bottom: 5px;">$1</h2>')
    .replace(/\*\*(.*?)\*\*/g, '<strong style="color: #67C23A;">$1</strong>')
    .replace(/\n/g, '<br/>')
})

const handleFileUpload = async (uploadFile: any) => {
  const file = uploadFile.raw;
  if (!file) return;

  isAnalyzing.value = true;
  hasData.value = false;
  aiReport.value = '';
  
  try {
    // 重点修改：向后端发送 scene = 'green'
    const response = await trajectoryApi.predict(file, 'exp1', 'green'); 
    const result = response as any; 
    
    const mode = result.predicted_mode || 'unknown';
    currentMode.value = translateMode(mode);
    const distanceMeters = result.stats?.distance || 0;
    const distanceKm = (distanceMeters / 1000).toFixed(2); 

    const lowerMode = mode.toLowerCase();
    const isGreen = ['walk', 'bike', 'bus', 'subway', '步行', '自行车', '公交车', '地铁'].includes(lowerMode);

    const calcGreenDist = isGreen ? distanceKm : '0.00';
    const calcCo2 = (Number(calcGreenDist) * 0.15).toFixed(2); 
    const calcTrees = (Number(calcCo2) * 0.05).toFixed(1);     

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
    ElMessage.success(`模型解析完毕！真实判定为：${currentMode.value}`);

  } catch (error) {
    ElMessage.error('模型推理失败，请检查服务状态');
  } finally {
    isAnalyzing.value = false;
  }
}

// 核心时空穿梭功能：根据 ID 还原页面
const restoreHistoryData = async (id: string) => {
  isAnalyzing.value = true;
  hasData.value = false;
  try {
    const record = await trajectoryApi.getHistoryById(id);
    const mode = record.predicted_mode || 'unknown';
    currentMode.value = translateMode(mode);
    
    const distanceKm = record.distance ? record.distance.toFixed(2) : '0.00';
    const lowerMode = mode.toLowerCase();
    const isGreen = ['walk', 'bike', 'bus', 'subway', '步行', '自行车', '公交车', '地铁'].includes(lowerMode);

    const calcGreenDist = isGreen ? distanceKm : '0.00';
    const calcCo2 = (Number(calcGreenDist) * 0.15).toFixed(2);
    const calcTrees = (Number(calcCo2) * 0.05).toFixed(1);

    stats.greenDistance = calcGreenDist;
    stats.co2Saved = calcCo2;
    stats.treesPlanted = calcTrees;
    hasData.value = true;

    // 重新在地图上绘制坐标！
    if (record.points && record.points !== '[]') {
      const pts = typeof record.points === 'string' ? JSON.parse(record.points) : record.points;
      renderLeafletTrajectory(pts, isGreen);
    }

    // 再次触发大模型生成当年情境的报告
    generateAIReport(calcGreenDist, calcCo2, calcTrees, currentMode.value);
    ElMessage.success(`时空档案已还原！当时判定为：${currentMode.value}`);
  } catch (error) {
    ElMessage.error('无法读取历史档案数据，文件可能已损坏');
  } finally {
    isAnalyzing.value = false;
  }
}

const translateMode = (mode: string) => {
  const map: Record<string, string> = { 'car': '私家车', 'bus': '公交车', 'walk': '步行', 'bike': '自行车', 'subway': '地铁', 'train': '火车' };
  return map[mode.toLowerCase()] || mode;
}

const generateAIReport = async (dist: string, co2: string, trees: string, modeName: string) => {
  isGeneratingReport.value = true
  aiReport.value = ''
  
  try {
    await trajectoryApi.streamReport(
      {
        model_id: 'exp1',
        mode: modeName,
        confidence: '95.0', // 绿色出行场景使用默认高置信度即可
        scene: 'green',
        distance: dist,
        co2: co2
      },
      // onMessage 回调：接收源源不断的文字流
      (text) => {
        aiReport.value = text;
        nextTick(() => { if (scrollBox.value) scrollBox.value.scrollTop = scrollBox.value.scrollHeight })
      },
      // onDone 回调
      () => {
        isGeneratingReport.value = false;
      },
      // onError 回调
      (error) => {
        console.error(error);
        aiReport.value = `**生成文案失败**\n请检查网络连接或后端服务状态。`;
        isGeneratingReport.value = false;
      }
    );
  } catch (error) {
    aiReport.value = `**生成文案失败**\n请检查网络连接。`;
    isGeneratingReport.value = false;
  }
}

const exportToPDF = () => {
  const element = document.getElementById('green-pdf-poster')
  if (!element) return
  const opt = {
    margin: 0, filename: `${userName.value}的绿色出行认证.pdf`,
    image: { type: 'jpeg' as const, quality: 1.0 },
    html2canvas: { scale: 3, useCORS: true, backgroundColor: '#f0f4f8' },
    jsPDF: { unit: 'px' as const, format: [800, 1000] as [number, number], orientation: 'portrait' as const }
  }
  ElMessage.success('正在为您生成专属纪念海报，请稍候...')
  html2pdf().set(opt).from(element).save()
}

const initLeafletMap = () => {
  if (!mapContainer.value) return
  map = L.map(mapContainer.value).setView([39.9042, 116.4074], 12)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OSM', maxZoom: 19, className: 'dark-map-tiles' 
  }).addTo(map)
}

const renderLeafletTrajectory = (points: any[], isGreen: boolean) => {
  if (!map) return
  if (currentPolyline) map.removeLayer(currentPolyline)
  markers.forEach(m => map!.removeLayer(m))
  markers = []

  const latLngs = points.map(p => {
    const lat = p.lat || p.latitude || p.y || (Array.isArray(p) ? p[1] : undefined);
    const lng = p.lng || p.lon || p.longitude || p.x || (Array.isArray(p) ? p[0] : undefined);
    return [lat, lng];
  }).filter(p => p[0] !== undefined && p[1] !== undefined) as [number, number][];

  if (latLngs.length === 0) return;

  const lineColor = isGreen ? '#67C23A' : '#F56C6C';
  currentPolyline = L.polyline(latLngs, { color: lineColor, weight: 5, opacity: 0.85, lineCap: 'round', lineJoin: 'round'}).addTo(map);

  const startPoint = latLngs[0] as L.LatLngExpression;
  const endPoint = latLngs[latLngs.length - 1] as L.LatLngExpression;
  const startMarker = L.circleMarker(startPoint, { radius: 7, fillColor: '#52C41A', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(map);
  const endMarker = L.marker(endPoint, {
    icon: L.divIcon({ className: 'custom-end-marker', html: '<div style="width: 14px; height: 14px; background: #F5222D; border: 2px solid #fff; border-radius: 2px; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>', iconSize: [14, 14], iconAnchor: [7, 7] }),
  }).addTo(map);

  markers.push(startMarker, endMarker);
  map.fitBounds(currentPolyline.getBounds(), { padding: [50, 50], animate: true, duration: 1 });
}

onMounted(() => { 
  nextTick(() => { initLeafletMap() }) 
  // 检查是否是从历史记录穿越回来的
  if (route.query.id) {
    restoreHistoryData(route.query.id as string)
  }
})
onUnmounted(() => { if (map) { map.remove(); map = null } })
</script>

<style scoped>
/* 保持你的完美样式不变，直接沿用你发我的即可 */
.green-container { height: calc(100vh - 70px); padding: 20px; box-sizing: border-box; }
.full-height { height: 100%; }
.panel { border-radius: 12px; height: 100%; display: flex; flex-direction: column; overflow: hidden; }
.dark-panel { background: rgba(26, 31, 46, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); }
.panel-header { padding: 16px 20px; font-size: 16px; font-weight: 600; color: #e5eaf3; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: space-between;}
.header-left { display: flex; align-items: center; gap: 8px;}
.panel-content { padding: 20px; flex: 1; overflow-y: auto; position: relative;}
.flex-col { display: flex; flex-direction: column; gap: 20px; }
.flex-1 { flex: 1; }
.map-wrapper { padding: 0; background: #0b0d12; position: relative; }
.leaflet-map { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }
:deep(.dark-map-tiles) { filter: brightness(0.7) invert(1) contrast(1.2) hue-rotate(200deg); }
:deep(.custom-end-marker) { background: transparent; border: none; }
.map-legend { position: absolute; bottom: 20px; right: 20px; background: rgba(0,0,0,0.7); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(4px); z-index: 1000;}
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #e5eaf3; margin-bottom: 8px;}
.line { width: 20px; height: 4px; border-radius: 2px; }
.line.green { background: #67C23A; box-shadow: 0 0 5px #67C23A;}
.line.red { background: #F56C6C; box-shadow: 0 0 5px #F56C6C;}
.section-box { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 16px; border: 1px solid rgba(255,255,255,0.03); transition: all 0.3s;}
.box-title { font-size: 14px; color: #a3a6ad; margin-bottom: 16px; font-weight: 500; display: flex; align-items: center;}
.blur-mask { opacity: 0.3; pointer-events: none; filter: blur(2px); }
:deep(.el-upload-dragger) { background: rgba(255,255,255,0.02); border-color: rgba(255,255,255,0.1); transition: all 0.3s; padding: 20px;}
:deep(.el-upload-dragger:hover) { border-color: #67C23A; background: rgba(103, 194, 58, 0.05);}
.text-center { text-align: center; margin-top: 10px; }
.carbon-dashboard { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.carbon-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 16px 12px; display: flex; flex-direction: column; align-items: center; gap: 10px; text-align: center;}
.icon-wrap { font-size: 24px; }
.data-wrap .val { font-size: 24px; font-weight: bold; font-family: monospace; }
.data-wrap .unit { font-size: 12px; margin-left: 2px;}
.data-wrap .label { font-size: 12px; color: #909399; margin-top: 4px;}
.text-green { color: #67C23A; }
.text-blue { color: #4A90E2; }
.tree-card { background: rgba(103, 194, 58, 0.05); border-color: rgba(103, 194, 58, 0.2);}
.report-box { border-color: rgba(74, 144, 226, 0.3); background: rgba(74, 144, 226, 0.05); display: flex; flex-direction: column;}
.ai-title { color: #4A90E2; }
.ai-content { flex: 1; font-size: 14px; line-height: 1.8; color: #e5eaf3; background: rgba(0,0,0,0.3); padding: 16px; border-radius: 6px; border-left: 3px solid #4A90E2; overflow-y: auto; scrollbar-width: thin;}
.empty-text { color: #909399; font-style: italic; text-align: center; margin-top: 20px;}
.typing-cursor { display: inline-block; width: 8px; height: 14px; background-color: #4A90E2; vertical-align: middle; margin-left: 4px; animation: blink 1s step-end infinite;}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

.pdf-export-wrapper { position: absolute; top: -9999px; left: -9999px; z-index: -1; }
.pdf-poster { width: 800px; min-height: 1000px; background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%); padding: 60px; font-family: 'Helvetica Neue', 'PingFang SC', 'Microsoft YaHei', sans-serif; color: #334155; box-sizing: border-box; position: relative; overflow: hidden; }
.pdf-poster::before { content: ''; position: absolute; top: -100px; right: -100px; width: 300px; height: 300px; background: rgba(56, 189, 248, 0.2); border-radius: 50%; filter: blur(50px); }
.poster-header { border-bottom: 2px solid #cbd5e1; padding-bottom: 20px; margin-bottom: 40px; }
.poster-title { font-size: 38px; font-weight: 800; color: #0f172a; margin-bottom: 10px; letter-spacing: 2px;}
.poster-subtitle { font-size: 18px; color: #64748b; display: flex; justify-content: space-between; align-items: flex-end;}
.poster-subtitle strong { color: #10b981; font-size: 22px; margin-left: 5px; }
.poster-stats { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 40px; }
.stat-box { flex: 1; background: #ffffff; padding: 30px 20px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); text-align: center; border-top: 5px solid #38bdf8; }
.stat-box:nth-child(2) { border-color: #10b981; }
.stat-value { font-size: 36px; font-weight: bold; color: #0ea5e9; font-family: monospace; line-height: 1.2;}
.stat-value small { font-size: 18px; }
.stat-label { font-size: 16px; color: #64748b; margin-top: 10px; }
.poster-ai-section { background: #ffffff; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); position: relative; }
.ai-badge { position: absolute; top: -15px; left: 40px; background: #E6A23C; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 14px; }
.pdf-ai-text { font-size: 18px; line-height: 2; color: #334155; }
.poster-footer { margin-top: 60px; text-align: center; color: #94a3b8; font-size: 14px; position: relative; }
.watermark { font-size: 60px; font-weight: 900; color: rgba(0,0,0,0.03); position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); white-space: nowrap; pointer-events: none; }
</style>