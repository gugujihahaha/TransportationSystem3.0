<template>
  <div class="congestion-container">
    <el-row :gutter="20" class="full-height">
      
      <el-col :span="14" class="full-height">
        <div class="panel dark-panel map-panel">
          <div class="panel-header">
            <div class="header-left">
              <el-icon><Location /></el-icon> 拥堵时空溯源与特征分析
            </div>
            <div class="header-actions">
              <el-tag size="small" type="warning" effect="dark">驱动引擎: {{ activeEngine.toUpperCase() }}</el-tag>
            </div>
          </div>
          <div class="panel-content map-wrapper" v-loading="isAnalyzing" element-loading-text="时空档案还原与引擎推理中..." element-loading-background="rgba(11, 13, 18, 0.8)">
            
            <div ref="mapContainer" class="leaflet-map"></div>
            
            <div class="map-legend">
              <div class="legend-item"><span class="line red"></span> 拥堵源流 (私家车/网约车)</div>
              <div class="legend-item"><span class="line green"></span> 公共交通 (公交/地铁)</div>
              <div class="legend-item"><span class="line blue"></span> 慢行系统 (骑行/步行)</div>
            </div>
          </div>
        </div>
      </el-col>

      <el-col :span="10" class="full-height">
        <div class="panel dark-panel control-panel">
          <div class="panel-header">
            <el-icon><DataBoard /></el-icon> 溯源控制台
          </div>
          <div class="panel-content flex-col">
            
            <div class="section-box">
              <div class="box-title">1. 导入待测轨迹 (支持北京全域)</div>
              <el-upload class="trajectory-upload" drag action="#" :auto-upload="false" :show-file-list="false" :on-change="handleUpload">
                <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                <div class="el-upload__text">拖拽轨迹文件到此处，或 <em>点击上传</em></div>
              </el-upload>
            </div>

            <div class="section-box" :class="{ 'blur-mask': !hasData && !currentFile }">
              <div class="box-title">2. 动态消融引擎切换</div>
              <div class="engine-list">
                <div v-for="(name, key) in engines" :key="key" 
                     class="engine-item" :class="{ active: activeEngine === key }"
                     @click="switchEngine(key as string)">
                  <div class="engine-name">{{ key.toUpperCase() }}</div>
                  <div class="engine-desc">{{ name }}</div>
                </div>
              </div>
            </div>

            <div class="section-box" :class="{ 'blur-mask': !hasData }">
              <div class="carbon-dashboard">
                <div class="carbon-card">
                  <div class="data-wrap">
                    <div class="val text-blue" style="font-size: 20px;">{{ currentMode }}</div>
                    <div class="label">当前判定模态</div>
                  </div>
                </div>
                <div class="carbon-card">
                  <div class="data-wrap">
                    <div class="val text-green">{{ confidence }}<span class="unit">%</span></div>
                    <div class="label">引擎置信度</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="section-box report-box flex-1" :class="{ 'blur-mask': !hasData }">
              <div class="box-title ai-title" style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                <div><el-icon><Tickets /></el-icon> 智能研判深度溯源</div>
                <el-button type="success" size="small" @click="exportToPDF" :disabled="!aiReport || isGeneratingReport" style="background: #10b981; border: none; font-weight: bold;">
                  📑 导出研判简报
                </el-button>
              </div>
              <div class="ai-content" ref="scrollBox">
                <div v-if="!aiReport && !isGeneratingReport" class="empty-text">等待引擎分析完成...</div>
                <div v-if="isGeneratingReport && !aiReport" class="empty-text" style="color: #E6A23C; animation: pulse 1s infinite;">
                  连接大模型中，正在生成深度洞察...
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
      <div id="congestion-pdf-report" class="pdf-poster">
        <div class="poster-header">
          <div class="poster-title">📑 城市交通拥堵溯源与治理研判报告</div>
          <div class="poster-subtitle">
            <span>系统研判员：<strong>{{ userName }}</strong></span>
            <span>生成日期：{{ currentDate }}</span>
          </div>
        </div>
        
        <div class="poster-stats">
          <div class="stat-box">
            <div class="stat-value" style="color: #F56C6C;">{{ currentMode }}</div>
            <div class="stat-label">溯源交通流判定</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">{{ activeEngine.toUpperCase() }}</div>
            <div class="stat-label">驱动推断引擎</div>
          </div>
          <div class="stat-box">
            <div class="stat-value" style="color: #67C23A;">{{ confidence }}<small>%</small></div>
            <div class="stat-label">多模态置信度</div>
          </div>
        </div>

        <div class="poster-ai-section">
          <div class="ai-badge" style="background: #4A90E2;">大模型深度洞察</div>
          <div class="pdf-ai-text" v-html="formattedReport"></div>
        </div>

        <div class="poster-footer">
          <p>本文档由 TrafficRec 交通大模型系统提供技术支持</p>
          <div class="watermark" style="color: rgba(74,144,226,0.05);">TrafficRec Analysis</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, computed } from 'vue'
import { useRoute } from 'vue-router'
import { Location, DataBoard, UploadFilled, Tickets } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { trajectoryApi } from '../api/trajectory'
import { useAuthStore } from '@/stores/auth'
import html2pdf from 'html2pdf.js'

const route = useRoute()
const authStore = useAuthStore()
const userName = computed(() => authStore.username || '交通系统研究员')
const currentDate = new Date().toLocaleDateString()

const hasData = ref(false)
const isAnalyzing = ref(false)
const isGeneratingReport = ref(false)
const aiReport = ref('')
const currentFile = ref<any>(null)
const scrollBox = ref<HTMLElement | null>(null)

const activeEngine = ref('exp1')
const currentMode = ref('--')
const confidence = ref('0.0')

const engines = { exp1: '纯轨迹基线模型', exp2: '+ OSM 路网拓扑增强', exp3: '+ 气象特征解耦', exp4: 'Focal Loss 终极优化' }
const modeColors: Record<string, string> = { car: '#F56C6C', taxi: '#F56C6C', bus: '#67C23A', subway: '#67C23A', bike: '#4A90E2', walk: '#4A90E2', unknown: '#909399' }

const mapContainer = ref<HTMLElement | null>(null)
let map: L.Map | null = null
let currentPolyline: L.Polyline | null = null
let markers: L.Layer[] = []

const initMap = () => {
  if (!mapContainer.value) return
  map = L.map(mapContainer.value).setView([39.9042, 116.4074], 12)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OSM', maxZoom: 19, className: 'dark-map-tiles' }).addTo(map)
}

const renderTrajectory = (points: any[], mode: string) => {
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
  const color = modeColors[mode.toLowerCase()] || '#4A90E2'
  currentPolyline = L.polyline(latLngs, { color: color, weight: 6, opacity: 0.9, lineCap: 'round', lineJoin: 'round'}).addTo(map);

  const startPoint = latLngs[0] as L.LatLngExpression;
  const endPoint = latLngs[latLngs.length - 1] as L.LatLngExpression;
  const startMarker = L.circleMarker(startPoint, { radius: 7, fillColor: '#52C41A', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(map);
  const endMarker = L.marker(endPoint, { icon: L.divIcon({ className: 'custom-end-marker', html: '<div style="width: 14px; height: 14px; background: #F5222D; border: 2px solid #fff; border-radius: 2px; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>', iconSize: [14, 14], iconAnchor: [7, 7] }) }).addTo(map);

  markers.push(startMarker, endMarker);
  map.fitBounds(currentPolyline.getBounds(), { padding: [50, 50], animate: true, duration: 1 });
}

const handleUpload = async (file: any) => {
  currentFile.value = file.raw
  await executeAnalysis(activeEngine.value)
}

const switchEngine = async (key: string) => {
  if (!currentFile.value && !hasData.value) return // 只有在没数据且没文件时才阻止
  activeEngine.value = key
  // 如果是从历史记录进来的，只改变显示，提示用户需重新上传文件来跑新模型
  if (!currentFile.value) {
      ElMessage.warning('这是历史记录的快照。若要测试其他引擎，请重新上传轨迹文件。');
      return;
  }
  await executeAnalysis(key)
}

const executeAnalysis = async (modelId: string) => {
  isAnalyzing.value = true; hasData.value = false; aiReport.value = ''
  try {
    // 重点修改：向后端发送 scene = 'congestion'
    const res: any = await trajectoryApi.predict(currentFile.value, modelId, 'congestion')
    currentMode.value = translateMode(res.predicted_mode || 'unknown')
    confidence.value = res.confidence ? (res.confidence * 100).toFixed(1) : '95.2'

    if (res.points && res.points.length > 0) renderTrajectory(res.points, res.predicted_mode || 'unknown')

    hasData.value = true
    ElMessage.success(`引擎 ${modelId.toUpperCase()} 推断完成`)
    generateAIReport(modelId, currentMode.value, confidence.value)
  } catch (e) {
    ElMessage.error('推断引擎连接失败')
  } finally {
    isAnalyzing.value = false
  }
}

// 核心时空穿梭功能：根据 ID 还原页面
const restoreHistoryData = async (id: string) => {
  isAnalyzing.value = true; hasData.value = false; aiReport.value = '';
  try {
    const record = await trajectoryApi.getHistoryById(id);
    activeEngine.value = record.model_id || 'exp1';
    currentMode.value = translateMode(record.predicted_mode || 'unknown');
    confidence.value = record.confidence ? (record.confidence * 100).toFixed(1) : '95.2';
    hasData.value = true;

    // 重新在地图上绘制坐标！
    if (record.points && record.points !== '[]') {
      const pts = typeof record.points === 'string' ? JSON.parse(record.points) : record.points;
      renderTrajectory(pts, record.predicted_mode || 'unknown');
    }

    generateAIReport(activeEngine.value, currentMode.value, confidence.value);
    ElMessage.success(`时空档案已还原！引擎快照：${activeEngine.value.toUpperCase()}`);
  } catch (error) {
    ElMessage.error('无法读取历史档案数据，文件可能已损坏');
  } finally {
    isAnalyzing.value = false;
  }
}

const translateMode = (mode: string) => {
  const map: Record<string, string> = { 'car': '私家车', 'bus': '公交车', 'walk': '步行', 'bike': '自行车', 'subway': '地铁', 'taxi': '出租车' }
  return map[mode.toLowerCase()] || mode.toUpperCase()
}

const formattedReport = computed(() => {
  if (!aiReport.value) return ''
  return aiReport.value
    .replace(/### (.*?)(?:\n|$)/g, '<h3 style="color: #E6A23C; margin: 14px 0 8px; font-size: 16px;">$1</h3>')
    .replace(/## (.*?)(?:\n|$)/g, '<h2 style="color: #333; margin: 18px 0 10px; font-size: 18px; border-bottom: 1px solid #eee; padding-bottom: 5px;">$1</h2>')
    .replace(/\*\*(.*?)\*\*/g, '<strong style="color: #E6A23C;">$1</strong>')
    .replace(/\n/g, '<br/>')
})
const generateAIReport = async (modelId: string, modeName: string, confValue: string) => {
  isGeneratingReport.value = true; 
  aiReport.value = '';

  try {
    // 调用安全的后端流式接口
    await trajectoryApi.streamReport(
      {
        model_id: modelId,
        mode: modeName,
        confidence: confValue,
        scene: 'congestion'
      },
      (text) => {
        aiReport.value = text;
        nextTick(() => { if (scrollBox.value) scrollBox.value.scrollTop = scrollBox.value.scrollHeight })
      },
      () => {
        isGeneratingReport.value = false;
      },
      (error) => {
        console.error(error);
        aiReport.value = `**生成报告失败**\n请检查后端大模型服务。`;
        isGeneratingReport.value = false;
      }
    );
  } catch (error) {
    aiReport.value = `**生成报告失败**\n请检查网络连接。`;
    isGeneratingReport.value = false;
  }
}

const exportToPDF = () => {
  const element = document.getElementById('congestion-pdf-report') 
  if (!element) return
  const opt = {
    margin: 0, filename: `交通分析报告_${new Date().getTime()}.pdf`,
    image: { type: 'jpeg' as const, quality: 1.0 },
    html2canvas: { scale: 3, useCORS: true, backgroundColor: '#ffffff' },
    jsPDF: { unit: 'px' as const, format: [800, 1000] as [number, number], orientation: 'portrait' as const }
  }
  ElMessage.success('正在为您生成专属报告，请稍候...')
  html2pdf().set(opt).from(element).save()
}

onMounted(() => { 
  nextTick(() => { initMap() }) 
  if (route.query.id) restoreHistoryData(route.query.id as string)
})
onUnmounted(() => { if (map) { map.remove(); map = null } })
</script>

<style scoped>
.congestion-container { height: calc(100vh - 70px); padding: 20px; box-sizing: border-box; }
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
.line.red { background: #F56C6C; box-shadow: 0 0 5px #F56C6C;}
.line.green { background: #67C23A; box-shadow: 0 0 5px #67C23A;}
.line.blue { background: #4A90E2; box-shadow: 0 0 5px #4A90E2;}
.section-box { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 16px; border: 1px solid rgba(255,255,255,0.03); transition: all 0.3s;}
.box-title { font-size: 14px; color: #a3a6ad; margin-bottom: 16px; font-weight: 500; display: flex; align-items: center;}
.blur-mask { opacity: 0.3; pointer-events: none; filter: blur(2px); }
:deep(.el-upload-dragger) { background: rgba(255,255,255,0.02); border-color: rgba(255,255,255,0.1); transition: all 0.3s; padding: 20px;}
:deep(.el-upload-dragger:hover) { border-color: #E6A23C; background: rgba(230, 162, 60, 0.05);}
.engine-list { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.engine-item { padding: 12px; border-radius: 6px; cursor: pointer; text-align: center; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s; }
.engine-item:hover { background: rgba(230, 162, 60, 0.05); border-color: rgba(230, 162, 60, 0.3); }
.engine-item.active { background: rgba(230, 162, 60, 0.15); border-color: #E6A23C; }
.engine-name { font-weight: bold; color: #fff; font-size: 13px;}
.engine-desc { font-size: 12px; color: #909399; margin-top: 4px;}
.engine-item.active .engine-name, .engine-item.active .engine-desc { color: #E6A23C; }
.carbon-dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.carbon-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 16px 12px; text-align: center;}
.data-wrap .val { font-size: 24px; font-weight: bold; font-family: monospace; }
.data-wrap .unit { font-size: 14px; margin-left: 2px;}
.data-wrap .label { font-size: 12px; color: #909399; margin-top: 4px;}
.text-green { color: #67C23A; }
.text-blue { color: #4A90E2; }
.report-box { border-color: rgba(230, 162, 60, 0.3); background: rgba(230, 162, 60, 0.05); display: flex; flex-direction: column;}
.ai-title { color: #E6A23C; }
.ai-content { flex: 1; font-size: 14px; line-height: 1.8; color: #e5eaf3; background: rgba(0,0,0,0.3); padding: 16px; border-radius: 6px; border-left: 3px solid #E6A23C; overflow-y: auto; scrollbar-width: thin;}
.empty-text { color: #909399; font-style: italic; text-align: center; margin-top: 20px;}
.typing-cursor { display: inline-block; width: 8px; height: 14px; background-color: #E6A23C; vertical-align: middle; margin-left: 4px; animation: blink 1s step-end infinite;}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

.pdf-export-wrapper { position: absolute; top: -9999px; left: -9999px; z-index: -1; }
.pdf-poster { width: 800px; min-height: 1000px; background: #ffffff; padding: 60px; font-family: 'Helvetica Neue', 'PingFang SC', 'Microsoft YaHei', sans-serif; color: #334155; box-sizing: border-box; position: relative; overflow: hidden; border-top: 15px solid #2b3a4a; }
.pdf-poster::after { content: ''; position: absolute; top: 0; right: 0; width: 300px; height: 150px; background-image: linear-gradient(rgba(74, 144, 226, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(74, 144, 226, 0.1) 1px, transparent 1px); background-size: 20px 20px; z-index: 0; }
.poster-header { border-bottom: 2px solid #cbd5e1; padding-bottom: 20px; margin-bottom: 40px; position: relative; z-index: 1; }
.poster-title { font-size: 34px; font-weight: 800; color: #1e293b; margin-bottom: 10px; letter-spacing: 1px;}
.poster-subtitle { font-size: 16px; color: #64748b; display: flex; justify-content: space-between; align-items: flex-end;}
.poster-subtitle strong { color: #4A90E2; font-size: 18px; margin-left: 5px; }
.poster-stats { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 40px; position: relative; z-index: 1; }
.stat-box { flex: 1; background: #f8fafc; padding: 25px 20px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center; }
.stat-value { font-size: 30px; font-weight: bold; color: #334155; line-height: 1.2;}
.stat-value small { font-size: 16px; }
.stat-label { font-size: 14px; color: #64748b; margin-top: 10px; font-weight: bold;}
.poster-ai-section { background: #ffffff; padding: 40px; border-radius: 8px; border: 1px solid #e2e8f0; position: relative; z-index: 1; }
.ai-badge { position: absolute; top: -12px; left: 30px; color: white; padding: 5px 15px; border-radius: 4px; font-weight: bold; font-size: 14px; }
.pdf-ai-text { font-size: 16px; line-height: 2; color: #334155; text-align: justify; }
.poster-footer { margin-top: 60px; text-align: center; color: #94a3b8; font-size: 14px; position: relative; }
.watermark { font-size: 70px; font-weight: 900; position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); white-space: nowrap; pointer-events: none; }
</style>