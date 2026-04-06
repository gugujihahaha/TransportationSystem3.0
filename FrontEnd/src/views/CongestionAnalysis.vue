<template>
  <div class="cyber-battle-station">
    
    <div class="side-panel left-panel">
      
      <div class="cyber-card header-card">
        <div class="cyber-title-box">
          <h2 class="neon-text-primary">拥堵溯源与特征分析控制台</h2>
        </div>
        <div class="sys-status blink-status"></div>
      </div>

      <div class="cyber-card content-card flex-col">
        <div class="module-box">
          <div class="module-header">
            <span class="bracket">[</span> <span class="neon-text-secondary">01</span> <span class="bracket">]</span> 轨迹数据流输入
          </div>
          <el-upload 
            class="neon-upload-area" 
            drag 
            action="#" 
            :auto-upload="false" 
            :show-file-list="false" 
            :on-change="handleUpload"
          >
            <div class="upload-content">
              <el-icon class="upload-icon"><UploadFilled /></el-icon>
              <div class="upload-text">拖拽或点击载入时空矩阵</div>
              <div class="upload-sub">CSV / JSON / PLT</div>
            </div>
          </el-upload>
        </div>

        <div class="module-box" :class="{ 'disabled-mask': !hasData && !currentFile }">
          <div class="module-header">
            <span class="bracket">[</span> <span class="neon-text-secondary">02</span> <span class="bracket">]</span> 驱动引擎消融矩阵
          </div>
          <div class="engine-grid">
            <div v-for="(name, key) in engines" :key="key" 
                 class="engine-btn" :class="{ 'engine-active': activeEngine === key }"
                 @click="switchEngine(key as string)">
              <div class="engine-id">{{ key.toUpperCase() }}</div>
              <div class="engine-name">{{ name }}</div>
              <div class="active-glow-bar"></div>
            </div>
          </div>
        </div>

        <div class="module-box flex-1" :class="{ 'disabled-mask': !hasData }">
          <div class="module-header">
            <span class="bracket">[</span> <span class="neon-text-secondary">03</span> <span class="bracket">]</span> 实时推断特征
          </div>
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-val" :style="{ color: getModeColor(currentMode), textShadow: `0 0 15px ${getModeColor(currentMode)}` }">
                {{ currentMode }}
              </div>
              <div class="metric-label">交通模态锁定</div>
            </div>
            <div class="metric-card">
              <div class="metric-val text-neon-cyan">{{ confidence }}<span class="unit">%</span></div>
              <div class="metric-label">引擎置信度</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="center-panel">
      <div class="map-frame" v-loading="isAnalyzing" element-loading-text="时空特征提取与神经推理中..." element-loading-background="rgba(2, 11, 24, 0.85)">
        <div ref="mapContainer" class="leaflet-map"></div>
        
        <div class="frame-corner top-left"></div>
        <div class="frame-corner top-right"></div>
        <div class="frame-corner bottom-left"></div>
        <div class="frame-corner bottom-right"></div>

        <div class="map-legend">
          <div class="legend-item"><span class="line red-glow"></span> 拥堵源流 (私家车/网约车)</div>
          <div class="legend-item"><span class="line green-glow"></span> 公共交通 (公交/地铁)</div>
          <div class="legend-item"><span class="line blue-glow"></span> 慢行系统 (骑行/步行)</div>
        </div>
      </div>
    </div>

    <div class="side-panel right-panel">
      <div class="cyber-card ai-card">
        <div class="ai-header">
          <div class="ai-title">
            <el-icon class="pulse-icon"><Tickets /></el-icon> AI 智能深度研判
          </div>
          <el-button type="primary" size="small" class="neon-btn" @click="exportToPDF" :disabled="!aiReport || isGeneratingReport">
            导出简报
          </el-button>
        </div>
        
        <div class="ai-body" ref="scrollBox">
          <div v-if="!aiReport && !isGeneratingReport" class="ai-placeholder blink-text">
            [ 等待时空数据流注入 ]
          </div>
          <div v-if="isGeneratingReport && !aiReport" class="ai-processing">
            <div class="spinner"></div>
            正在建立认知引擎链路...
          </div>
          <div v-else-if="aiReport" class="ai-stream-text">
            <span v-html="formattedReport"></span>
            <span v-if="isGeneratingReport" class="type-cursor"></span>
          </div>
        </div>
      </div>
    </div>

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
  map = L.map(mapContainer.value, { zoomControl: false }).setView([39.9042, 116.4074], 12)
  L.tileLayer('http://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}', { 
    attribution: '© AutoNavi', 
    maxZoom: 19 
  }).addTo(map)
}

const renderTrajectory = (points: any[], mode: string) => {
  if (!map) return
  if (currentPolyline) map.removeLayer(currentPolyline)
  markers.forEach(m => map!.removeLayer(m))
  markers = []
  const latLngs = points.map(p => [p.lat || p.latitude, p.lng || p.longitude]).filter(p => p[0] && p[1]) as [number, number][];
  if (latLngs.length === 0) return;
  const color = modeColors[mode.toLowerCase()] || '#4A90E2'
  currentPolyline = L.polyline(latLngs, { color: color, weight: 5, opacity: 0.85, lineCap: 'round', lineJoin: 'round'}).addTo(map);
  
  const startMarker = L.circleMarker(latLngs[0] as L.LatLngExpression, { radius: 6, fillColor: '#52C41A', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(map);
  const endMarker = L.marker(latLngs[latLngs.length - 1] as L.LatLngExpression, {
    icon: L.divIcon({ className: 'custom-end-marker', html: '<div style="width: 12px; height: 12px; background: #F5222D; border: 2px solid #fff; border-radius: 2px;"></div>', iconSize: [12, 12], iconAnchor: [6, 6] })
  }).addTo(map);
  markers.push(startMarker, endMarker);
  map.fitBounds(currentPolyline.getBounds(), { padding: [50, 50], animate: true, duration: 1 });
}

const handleUpload = async (file: any) => {
  currentFile.value = file.raw
  await executeAnalysis(activeEngine.value)
}

const switchEngine = async (key: string) => {
  if (!currentFile.value && !hasData.value) return 
  activeEngine.value = key
  if (!currentFile.value) {
      ElMessage.warning('更迭引擎需重新载入数据。');
      return;
  }
  await executeAnalysis(key)
}

const executeAnalysis = async (modelId: string) => {
  isAnalyzing.value = true; hasData.value = false; aiReport.value = ''
  try {
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

const restoreHistoryData = async (id: string) => {
  isAnalyzing.value = true; hasData.value = false; aiReport.value = '';
  try {
    const record = await trajectoryApi.getHistoryById(id);
    activeEngine.value = record.model_id || 'exp1';
    currentMode.value = translateMode(record.predicted_mode || 'unknown');
    confidence.value = record.confidence ? (record.confidence * 100).toFixed(1) : '95.2';
    hasData.value = true;

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

const getModeColor = (mode: string) => {
  if (['私家车', '出租车', 'CAR', 'TAXI'].includes(mode)) return '#FF3366'; // 霓虹红/粉
  if (['公交车', '地铁', 'BUS', 'SUBWAY'].includes(mode)) return '#39FF14'; // 霓虹绿
  if (['步行', '自行车', 'WALK', 'BIKE'].includes(mode)) return '#1890FF'; // 亮蓝
  return '#00F0FF'; // 霓虹青
}

const formattedReport = computed(() => {
  if (!aiReport.value) return ''
  return aiReport.value
    .replace(/### (.*?)(?:\n|$)/g, '<h3 class="ai-h3"><span class="highlight-bar"></span> $1</h3>')
    .replace(/## (.*?)(?:\n|$)/g, '<h2 class="ai-h2">$1</h2>')
    .replace(/\*\*(.*?)\*\*/g, '<strong class="ai-strong">$1</strong>')
    .replace(/\n/g, '<br/>')
})

const generateAIReport = async (modelId: string, modeName: string, confValue: string) => {
  isGeneratingReport.value = true; 
  aiReport.value = '';

  try {
    await trajectoryApi.streamReport(
      { model_id: modelId, mode: modeName, confidence: confValue, scene: 'congestion' },
      (text) => {
        aiReport.value = text;
        nextTick(() => { if (scrollBox.value) scrollBox.value.scrollTop = scrollBox.value.scrollHeight })
      },
      () => { isGeneratingReport.value = false; },
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
.cyber-battle-station {
  display: flex;
  width: 100%;
  height: calc(100vh - 64px);
  background: #020B16; 
  padding: 16px;
  gap: 16px; 
  box-sizing: border-box;
  font-family: 'Helvetica Neue', Helvetica, 'Microsoft YaHei', sans-serif;
}

.disabled-mask { opacity: 0.3; pointer-events: none; transition: all 0.3s; filter: grayscale(1); }

.cyber-card {
  background: rgba(4, 18, 38, 0.7);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(0, 240, 255, 0.3);
  border-radius: 8px;
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
  display: flex;
  flex-direction: column;
}

.flex-col { display: flex; flex-direction: column; gap: 20px; }
.flex-1 { flex: 1; }

/* 文本高亮类 */
.neon-text-primary { margin: 0; font-size: 20px; font-weight: 800; color: #00F0FF; text-shadow: 0 0 10px rgba(0, 240, 255, 0.6); }
.neon-text-secondary { color: #00F0FF; font-weight: bold; text-shadow: 0 0 5px rgba(0, 240, 255, 0.4); }
.text-neon-cyan { color: #00F0FF; text-shadow: 0 0 15px rgba(0, 240, 255, 0.6); }

.side-panel { display: flex; flex-direction: column; gap: 16px; z-index: 10; }
.left-panel { width: 340px; flex-shrink: 0; }
.right-panel { width: 420px; flex-shrink: 0; }
.center-panel { flex: 1; display: flex; position: relative; z-index: 5; } /* 地图区域撑满中间 */

.header-card { padding: 16px 20px; display: flex; flex-direction: row; justify-content: space-between; align-items: center; }
.sub-title { font-size: 12px; color: #8892b0; margin-top: 4px; letter-spacing: 1px;}
.sys-status { width: 10px; height: 10px; border-radius: 50%; background: #39FF14; box-shadow: 0 0 10px #39FF14; }
.blink-status { animation: breathe 2s infinite; }

.content-card { padding: 20px; flex: 1; overflow-y: auto; }
.content-card::-webkit-scrollbar { width: 4px; }
.content-card::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.5); border-radius: 2px; }

.module-header { font-size: 14px; color: #cdd9e5; margin-bottom: 16px; display: flex; align-items: center; gap: 6px; }
.bracket { color: #888; font-family: monospace; }

:deep(.el-upload-dragger) {
  background: rgba(0, 240, 255, 0.02) !important;
  border: 1px dashed rgba(0, 240, 255, 0.4) !important;
  border-radius: 8px; padding: 24px 0; transition: all 0.3s ease;
}
:deep(.el-upload-dragger:hover) {
  border-color: #00F0FF !important; background: rgba(0, 240, 255, 0.08) !important;
  box-shadow: inset 0 0 20px rgba(0, 240, 255, 0.15);
}
.upload-content { text-align: center; }
.upload-icon { font-size: 36px; color: #00F0FF; margin-bottom: 12px; filter: drop-shadow(0 0 5px rgba(0, 240, 255, 0.5)); }
.upload-text { font-size: 14px; color: #fff; font-weight: 500; margin-bottom: 6px; }
.upload-sub { font-size: 12px; color: #8892b0; font-family: 'Consolas', monospace; }

.engine-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.engine-btn {
  background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 14px 10px; border-radius: 6px; cursor: pointer; position: relative; text-align: center; transition: all 0.2s; overflow: hidden;
}
.engine-btn:hover { background: rgba(0, 240, 255, 0.05); border-color: rgba(0, 240, 255, 0.4); }
.engine-active {
  background: rgba(0, 240, 255, 0.1); border-color: #00F0FF; box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
}
.active-glow-bar { position: absolute; left: 0; bottom: 0; width: 100%; height: 2px; background: transparent; }
.engine-active .active-glow-bar { background: #00F0FF; box-shadow: 0 0 10px #00F0FF; }
.engine-id { font-size: 14px; font-weight: bold; color: #e2e8f0; margin-bottom: 4px; font-family: 'Consolas', monospace; }
.engine-name { font-size: 11px; color: #8892b0; }
.engine-active .engine-id, .engine-active .engine-name { color: #00F0FF; }

/* 特征面板 */
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.metric-card {
  background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(0, 240, 255, 0.15);
  border-radius: 8px; padding: 20px 10px; text-align: center;
}
.metric-val { font-size: 26px; font-weight: bold; font-family: 'Consolas', monospace; line-height: 1; }
.unit { font-size: 14px; margin-left: 2px; }
.metric-label { font-size: 12px; color: #8892b0; margin-top: 10px; }

.map-frame {
  flex: 1; position: relative; border-radius: 12px;
  border: 2px solid rgba(0, 240, 255, 0.5);
  box-shadow: 0 0 30px rgba(0, 240, 255, 0.15), inset 0 0 30px rgba(0, 240, 255, 0.1);
  overflow: hidden;
}

.leaflet-map { width: 100%; height: 100%; background: #020B16; z-index: 1; }

:deep(.custom-map-tiles) {
  filter: brightness(0.9) saturate(1.1);
}

:deep(.custom-end-marker) { background: transparent; border: none; }

.frame-corner { position: absolute; width: 30px; height: 30px; border: 3px solid transparent; z-index: 10; pointer-events: none;}
.top-left { top: 0; left: 0; border-top-color: #00F0FF; border-left-color: #00F0FF; border-radius: 12px 0 0 0;}
.top-right { top: 0; right: 0; border-top-color: #00F0FF; border-right-color: #00F0FF; border-radius: 0 12px 0 0;}
.bottom-left { bottom: 0; left: 0; border-bottom-color: #00F0FF; border-left-color: #00F0FF; border-radius: 0 0 0 12px;}
.bottom-right { bottom: 0; right: 0; border-bottom-color: #00F0FF; border-right-color: #00F0FF; border-radius: 0 0 12px 0;}

/* 底部图例 */
.map-legend {
  position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
  background: rgba(2, 11, 22, 0.85); backdrop-filter: blur(10px);
  padding: 12px 24px; border-radius: 30px;
  border: 1px solid rgba(0, 240, 255, 0.3);
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.6);
  z-index: 1000; display: flex; gap: 24px;
}
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #fff; font-weight: 500;}
.line { width: 16px; height: 4px; border-radius: 2px; }
.red-glow { background: #FF3366; box-shadow: 0 0 8px #FF3366;}
.green-glow { background: #39FF14; box-shadow: 0 0 8px #39FF14;}
.blue-glow { background: #1890FF; box-shadow: 0 0 8px #1890FF;}

.ai-card { height: 100%; }
.ai-header {
  padding: 16px 20px; border-bottom: 1px solid rgba(0, 240, 255, 0.3);
  background: linear-gradient(90deg, rgba(0, 240, 255, 0.1), transparent);
  display: flex; justify-content: space-between; align-items: center;
}
.ai-title { font-size: 16px; font-weight: bold; color: #00F0FF; display: flex; align-items: center; gap: 8px; text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);}
.pulse-icon { animation: breathe 2s infinite; }
.neon-btn {
  background: transparent !important; border: 1px solid #00F0FF !important; color: #00F0FF !important;
  font-family: 'Consolas', monospace; transition: all 0.3s;
}
.neon-btn:hover:not(:disabled) { background: #00F0FF !important; color: #020B16 !important; box-shadow: 0 0 15px #00F0FF; }

.ai-body { padding: 24px; flex: 1; overflow-y: auto; }
.ai-body::-webkit-scrollbar { width: 4px; }
.ai-body::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.5); border-radius: 2px; }

.ai-placeholder { text-align: center; color: #8892b0; margin-top: 40px; font-family: 'Consolas', monospace; }
.blink-text { animation: blink 2s infinite; }

.ai-processing { display: flex; flex-direction: column; align-items: center; color: #00F0FF; margin-top: 40px; gap: 16px; }
.spinner { width: 40px; height: 40px; border: 3px solid rgba(0,240,255,0.2); border-top-color: #00F0FF; border-radius: 50%; animation: spin 1s linear infinite; box-shadow: 0 0 15px rgba(0,240,255,0.4); }

/* AI 文本渲染高亮 */
.ai-stream-text { color: #e2e8f0; line-height: 1.8; font-size: 14.5px; }
:deep(.ai-h2) { color: #00F0FF; font-size: 18px; margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px dashed rgba(0, 240, 255, 0.3); text-shadow: 0 0 8px rgba(0, 240, 255, 0.4);}
:deep(.ai-h3) { color: #E6A23C; font-size: 15px; margin: 16px 0 8px; display: flex; align-items: center; gap: 8px;}
:deep(.highlight-bar) { display: inline-block; width: 4px; height: 14px; background: #E6A23C; box-shadow: 0 0 8px #E6A23C; border-radius: 2px;}
:deep(.ai-strong) { color: #39FF14; font-weight: bold; text-shadow: 0 0 5px rgba(57, 255, 20, 0.3);}
.type-cursor { display: inline-block; width: 8px; height: 16px; background: #00F0FF; margin-left: 6px; vertical-align: middle; animation: blink 1s step-end infinite; box-shadow: 0 0 8px #00F0FF;}

@keyframes spin { to { transform: rotate(360deg); } }
@keyframes breathe { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.6; transform: scale(1.1); } }
@keyframes blink { 50% { opacity: 0; } }

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