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
          <div class="panel-content map-wrapper" v-loading="isAnalyzing" element-loading-text="PyTorch 引擎推理中..." element-loading-background="rgba(11, 13, 18, 0.8)">
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
                <div><el-icon><Tickets /></el-icon> 星火 AI 绿色出行摘要</div>
                <el-button type="success" size="small" @click="exportToPDF" :disabled="!aiReport || isGeneratingReport" style="background: #10b981; border: none;">
                  📄 导出PDF
                </el-button>
              </div>
              <div class="ai-content" id="green-pdf-content" ref="scrollBox">
                <div v-if="!aiReport && !isGeneratingReport" class="empty-text">等待模型分析完成...</div>
                <div v-if="isGeneratingReport && !aiReport" class="empty-text" style="color: #4A90E2; animation: pulse 1s infinite;">
                  连接星火大模型中，正在生成绿色出行摘要...
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
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, nextTick, computed } from 'vue'
import { Compass, Bicycle, UploadFilled, Guide, WindPower, Sunny, Tickets } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { trajectoryApi } from '../api/trajectory'
import html2pdf from 'html2pdf.js'

import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// --- 状态与变量 ---
const hasData = ref(false)
const isAnalyzing = ref(false)
const isGeneratingReport = ref(false)
const aiReport = ref('')
const scrollBox = ref<HTMLElement | null>(null)

// Leaflet 地图变量
const mapContainer = ref<HTMLElement | null>(null)
let map: L.Map | null = null
let currentPolyline: L.Polyline | null = null
let markers: L.Layer[] = []

// 统计数据
const stats = reactive({
  greenDistance: '0.00',
  co2Saved: '0.00',
  treesPlanted: '0.0'
})

// --- 核心业务逻辑 ---
const handleFileUpload = async (uploadFile: any) => {
  const file = uploadFile.raw;
  if (!file) return;

  isAnalyzing.value = true;
  hasData.value = false;
  aiReport.value = '';
  
  try {
    const response = await trajectoryApi.predict(file, 'exp1'); 
    const result = response as any; 
    
    const mode = result.predicted_mode || 'unknown';
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

    // 触发打字机
    generateAIReport(calcGreenDist, calcCo2, calcTrees, translateMode(mode));
    ElMessage.success(`模型解析完毕！真实判定为：${translateMode(mode)}`);

  } catch (error) {
    ElMessage.error('模型推理失败，请检查服务状态');
  } finally {
    isAnalyzing.value = false;
  }
}

const translateMode = (mode: string) => {
  const map: Record<string, string> = { 'car': '私家车', 'bus': '公交车', 'walk': '步行', 'bike': '自行车', 'subway': '地铁', 'train': '火车' };
  return map[mode.toLowerCase()] || mode;
}

// 格式化 Markdown
const formattedReport = computed(() => {
  if (!aiReport.value) return ''
  return aiReport.value
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
})

// 🚀 原生 Fetch 对接【讯飞星火】流式打字机
const generateAIReport = async (dist: string, co2: string, trees: string, modeName: string) => {
  isGeneratingReport.value = true
  aiReport.value = ''
  
  let prompt = ''
  if (Number(dist) > 0) {
    prompt = `你是一个低碳出行专家。系统最新判定本次出行为：${modeName}。\n绿色里程：${dist}km\n累计减少碳排放：${co2}kg\n相当于种树：${trees}棵\n请写一段富有鼓舞性的个人绿色出行摘要，包含数据评价和环保建议，使用Markdown格式，字数150字左右。`
  } else {
    prompt = `你是一个低碳出行专家。系统最新判定本次出行为高碳排交通方式：${modeName}。\n请给出鼓励其下次选择公共交通或慢行系统以减少城市碳足迹的委婉建议，使用Markdown格式，字数150字左右。`
  }

  try {
    // 星火兼容 OpenAI 的 API 地址
    const response = await fetch('/spark-api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ZOawgFgAMWrgzoramwRS:BkjUHBXpuOrXCpVQfFtJ` 
      },
      body: JSON.stringify({
        model: 'lite', // 星火免费模型
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7,
        stream: true
      })
    })

    if (!response.body) throw new Error('流式响应不可用')

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ') && !line.includes('[DONE]')) {
          try {
            const data = JSON.parse(line.substring(6))
            const text = data.choices[0].delta.content || ''
            aiReport.value += text
            
            // 自动滚动到底部
            nextTick(() => {
              if (scrollBox.value) {
                scrollBox.value.scrollTop = scrollBox.value.scrollHeight
              }
            })
          } catch (e) {}
        }
      }
    }
  } catch (error) {
    console.error(error)
    aiReport.value = `**生成报告失败**\n请检查网络连接或 API Key 是否有效。`
  } finally {
    isGeneratingReport.value = false
  }
}

// 📄 一键导出 PDF（处理深色模式适配）
const exportToPDF = () => {
  const element = document.getElementById('green-pdf-content')
  if (!element) return

  const opt = {
    margin:       10,
    filename:     `绿色出行洞察报告_${new Date().getTime()}.pdf`,
    image:        { type: 'jpeg' as const, quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#ffffff' },
    jsPDF:        { unit: 'mm' as const, format: 'a4' as const, orientation: 'portrait' as const }
  }

  const clone = element.cloneNode(true) as HTMLElement
  
  clone.style.backgroundColor = 'white'
  clone.style.color = 'black'
  clone.style.padding = '20px'
  clone.style.height = 'auto'
  clone.style.overflow = 'visible'
  
  clone.querySelectorAll('*').forEach((child: any) => {
    child.style.color = 'black'
  })

  const cursor = clone.querySelector('.typing-cursor')
  if (cursor) cursor.remove()
  
  html2pdf().set(opt).from(clone).save()
}

// ==========================================
// 🗺️ Leaflet 真实 GIS 地图渲染引擎
// ==========================================
const initLeafletMap = () => {
  if (!mapContainer.value) return
  map = L.map(mapContainer.value).setView([39.9042, 116.4074], 12)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
    className: 'dark-map-tiles' 
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
    return [lat, lng] as [number, number];
  }).filter(p => p[0] !== undefined && p[1] !== undefined);

  if (latLngs.length === 0) return;

  const lineColor = isGreen ? '#67C23A' : '#F56C6C';
  
  currentPolyline = L.polyline(latLngs, {
    color: lineColor,
    weight: 5,
    opacity: 0.85,
    lineCap: 'round',
    lineJoin: 'round'
  }).addTo(map);

  const startPoint = latLngs[0] as [number, number];
  const endPoint = latLngs[latLngs.length - 1] as [number, number];
  
  const startMarker = L.circleMarker(startPoint, {
    radius: 7, fillColor: '#52C41A', color: '#fff', weight: 2, fillOpacity: 1,
  }).addTo(map);
  startMarker.bindTooltip('起点', { permanent: false, direction: 'top' });

  const endMarker = L.marker(endPoint, {
    icon: L.divIcon({
      className: 'custom-end-marker',
      html: '<div style="width: 14px; height: 14px; background: #F5222D; border: 2px solid #fff; border-radius: 2px; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>',
      iconSize: [14, 14], iconAnchor: [7, 7],
    }),
  }).addTo(map);
  endMarker.bindTooltip('终点', { permanent: false, direction: 'top' });

  markers.push(startMarker, endMarker);
  map.fitBounds(currentPolyline.getBounds(), { padding: [50, 50], animate: true, duration: 1 });
}

onMounted(async () => {
  await nextTick()
  initLeafletMap()
})

onUnmounted(() => {
  if (map) { map.remove(); map = null }
})
</script>

<style scoped>
.green-container { 
  height: calc(100vh - 70px); 
  padding: 20px; 
  box-sizing: border-box;
}
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
.legend-item:last-child { margin-bottom: 0;}
.line { width: 20px; height: 4px; border-radius: 2px; }
.line.green { background: #67C23A; box-shadow: 0 0 5px #67C23A;}
.line.red { background: #F56C6C; box-shadow: 0 0 5px #F56C6C;}

.section-box { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 16px; border: 1px solid rgba(255,255,255,0.03); transition: all 0.3s;}
.box-title { font-size: 14px; color: #a3a6ad; margin-bottom: 16px; font-weight: 500; display: flex; align-items: center;}
.blur-mask { opacity: 0.3; pointer-events: none; filter: blur(2px); }

:deep(.el-upload-dragger) { background: rgba(255,255,255,0.02); border-color: rgba(255,255,255,0.1); transition: all 0.3s; padding: 20px;}
:deep(.el-upload-dragger:hover) { border-color: #67C23A; background: rgba(103, 194, 58, 0.05);}
:deep(.el-upload.is-disabled .el-upload-dragger) { cursor: not-allowed; border-color: #333; background: #111;}
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
.ai-content { 
  flex: 1; font-size: 14px; line-height: 1.8; color: #e5eaf3; background: rgba(0,0,0,0.3); 
  padding: 16px; border-radius: 6px; border-left: 3px solid #4A90E2;
  overflow-y: auto; scrollbar-width: thin;
}
.empty-text { color: #909399; font-style: italic; text-align: center; margin-top: 20px;}
:deep(strong) { color: #67C23A; }

.typing-cursor {
  display: inline-block;
  width: 8px;
  height: 14px;
  background-color: #4A90E2;
  vertical-align: middle;
  margin-left: 4px;
  animation: blink 1s step-end infinite;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>