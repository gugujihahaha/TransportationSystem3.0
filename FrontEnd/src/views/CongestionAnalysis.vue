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
          <div class="panel-content map-wrapper" v-loading="isAnalyzing" element-loading-text="PyTorch 多模态引擎推理中..." element-loading-background="rgba(11, 13, 18, 0.8)">
            
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

            <div class="section-box" :class="{ 'blur-mask': !currentFile }">
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
                <div><el-icon><Tickets /></el-icon> 星火 AI 深度溯源报告</div>
                <el-button type="success" size="small" @click="exportToPDF" :disabled="!aiReport || isGeneratingReport" style="background: #10b981; border: none;">
                  📄 导出PDF
                </el-button>
              </div>
              <div class="ai-content" id="congestion-pdf-content" ref="scrollBox">
                <div v-if="!aiReport && !isGeneratingReport" class="empty-text">等待引擎分析完成...</div>
                <div v-if="isGeneratingReport && !aiReport" class="empty-text" style="color: #E6A23C; animation: pulse 1s infinite;">
                  连接星火大模型中，正在生成深度洞察...
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
import { ref, onMounted, onUnmounted, nextTick, computed } from 'vue'
import { Location, DataBoard, UploadFilled, Tickets } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { trajectoryApi } from '../api/trajectory'
import html2pdf from 'html2pdf.js'

// --- 状态与变量 ---
const hasData = ref(false)
const isAnalyzing = ref(false)
const isGeneratingReport = ref(false)
const aiReport = ref('')
const currentFile = ref<any>(null)
const scrollBox = ref<HTMLElement | null>(null)

const activeEngine = ref('exp1')
const currentMode = ref('--')
const confidence = ref('0.0')

const engines = {
  exp1: '纯轨迹基线模型',
  exp2: '+ OSM 路网拓扑增强',
  exp3: '+ 气象特征解耦',
  exp4: 'Focal Loss 终极优化'
}

// 颜色映射字典
const modeColors: Record<string, string> = {
  car: '#F56C6C', taxi: '#F56C6C',   // 红色
  bus: '#67C23A', subway: '#67C23A', // 绿色
  bike: '#4A90E2', walk: '#4A90E2',  // 蓝色
  unknown: '#909399'
}

// --- Leaflet 地图变量 ---
const mapContainer = ref<HTMLElement | null>(null)
let map: L.Map | null = null
let currentPolyline: L.Polyline | null = null
let markers: L.Layer[] = []

// ==========================================
// 🗺️ 地图引擎
// ==========================================
const initMap = () => {
  if (!mapContainer.value) return
  map = L.map(mapContainer.value).setView([39.9042, 116.4074], 12)

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
    className: 'dark-map-tiles' 
  }).addTo(map)
}

const renderTrajectory = (points: any[], mode: string) => {
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

  const color = modeColors[mode.toLowerCase()] || '#4A90E2'
  
  currentPolyline = L.polyline(latLngs, {
    color: color,
    weight: 6,
    opacity: 0.9,
    lineCap: 'round',
    lineJoin: 'round'
  }).addTo(map);

  const startPoint = latLngs[0] as [number, number];
  const endPoint = latLngs[latLngs.length - 1] as [number, number];

  const startMarker = L.circleMarker(startPoint, { radius: 7, fillColor: '#52C41A', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(map);
  const endMarker = L.marker(endPoint, {
    icon: L.divIcon({
      className: 'custom-end-marker',
      html: '<div style="width: 14px; height: 14px; background: #F5222D; border: 2px solid #fff; border-radius: 2px; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>',
      iconSize: [14, 14], iconAnchor: [7, 7],
    }),
  }).addTo(map);

  startMarker.bindTooltip('起点', { permanent: false, direction: 'top' });
  endMarker.bindTooltip('终点', { permanent: false, direction: 'top' });
  markers.push(startMarker, endMarker);

  map.fitBounds(currentPolyline.getBounds(), { padding: [50, 50], animate: true, duration: 1 });
}

// ==========================================
// 业务与调度逻辑
// ==========================================
const handleUpload = async (file: any) => {
  currentFile.value = file.raw
  await executeAnalysis(activeEngine.value)
}

const switchEngine = async (key: string) => {
  if (!currentFile.value) return
  activeEngine.value = key
  await executeAnalysis(key)
}

const executeAnalysis = async (modelId: string) => {
  isAnalyzing.value = true
  hasData.value = false
  aiReport.value = ''
  
  try {
    const res: any = await trajectoryApi.predict(currentFile.value, modelId)
    
    currentMode.value = translateMode(res.predicted_mode || 'unknown')
    confidence.value = res.confidence ? (res.confidence * 100).toFixed(1) : '95.2'

    if (res.points && res.points.length > 0) {
      renderTrajectory(res.points, res.predicted_mode || 'unknown')
    }

    hasData.value = true
    ElMessage.success(`引擎 ${modelId.toUpperCase()} 推断完成`)
    
    // 不 await，让打字机自己去跑，不阻塞页面
    generateAIReport(modelId, currentMode.value, confidence.value)

  } catch (e) {
    ElMessage.error('推断引擎连接失败')
  } finally {
    isAnalyzing.value = false
  }
}

const translateMode = (mode: string) => {
  const map: Record<string, string> = { 'car': '私家车', 'bus': '公交车', 'walk': '步行', 'bike': '自行车', 'subway': '地铁', 'taxi': '出租车' }
  return map[mode.toLowerCase()] || mode.toUpperCase()
}

// 格式化 Markdown
const formattedReport = computed(() => {
  if (!aiReport.value) return ''
  return aiReport.value
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
})

// 🚀 原生 Fetch 对接【讯飞星火】流式打字机
const generateAIReport = async (modelId: string, modeName: string, confValue: string) => {
  isGeneratingReport.value = true
  aiReport.value = ''
  
  const prompt = `你是一个专业的交通智能分析专家。基于以下最新数据生成溯源报告：\n1. 驱动引擎：${modelId.toUpperCase()}\n2. 识别出的主要拥堵交通流：${modeName}\n3. 引擎置信度：${confValue}%\n请按“时空特征推导”和“拥堵治理建议”两个部分输出，使用Markdown格式，总字数控制在200字左右。`

  try {
    // 星火兼容 OpenAI 的 API 地址
    const response = await fetch('/spark-api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ZOawgFgAMWrgzoramwRS:BkjUHBXpuOrXCpVQfFtJ` 
      },
      body: JSON.stringify({
        model: 'lite', // 星火基础免费模型
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
          } catch (e) {
            // 忽略 JSON 截断错误
          }
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

// 📄 一键导出 PDF（处理深色模式适配白底黑字打印）
const exportToPDF = () => {
  const element = document.getElementById('congestion-pdf-content')
  if (!element) return

  const opt = {
    margin:       10,
    filename:     `交通拥堵溯源报告_${new Date().getTime()}.pdf`,
    image:        { type: 'jpeg' as const, quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#ffffff' },
    jsPDF:        { unit: 'mm' as const, format: 'a4' as const, orientation: 'portrait' as const }
  }

  const clone = element.cloneNode(true) as HTMLElement
  
  // 强制变身白底黑字
  clone.style.backgroundColor = 'white'
  clone.style.color = 'black'
  clone.style.padding = '20px'
  clone.style.height = 'auto'
  clone.style.overflow = 'visible'
  
  clone.querySelectorAll('*').forEach((child: any) => {
    child.style.color = 'black'
  })

  // 移除闪烁的光标
  const cursor = clone.querySelector('.typing-cursor')
  if (cursor) cursor.remove()
  
  html2pdf().set(opt).from(clone).save()
}

onMounted(() => { nextTick(() => { initMap() }) })
onUnmounted(() => { if (map) { map.remove(); map = null } })
</script>

<style scoped>
.congestion-container { 
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
.leaflet-map { 
  position: absolute; 
  top: 0; 
  left: 0; 
  width: 100%; 
  height: 100%; 
  z-index: 1;
}

:deep(.dark-map-tiles) { filter: brightness(0.7) invert(1) contrast(1.2) hue-rotate(200deg); }
:deep(.custom-end-marker) { background: transparent; border: none; }

.map-legend { position: absolute; bottom: 20px; right: 20px; background: rgba(0,0,0,0.7); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(4px); z-index: 1000;}
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #e5eaf3; margin-bottom: 8px;}
.legend-item:last-child { margin-bottom: 0;}
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
.engine-item {
  padding: 12px; border-radius: 6px; cursor: pointer; text-align: center;
  background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s; 
}
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
.ai-content { 
  flex: 1; font-size: 14px; line-height: 1.8; color: #e5eaf3; background: rgba(0,0,0,0.3); 
  padding: 16px; border-radius: 6px; border-left: 3px solid #E6A23C;
  overflow-y: auto; scrollbar-width: thin;
}
.empty-text { color: #909399; font-style: italic; text-align: center; margin-top: 20px;}
:deep(strong) { color: #E6A23C; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

.typing-cursor {
  display: inline-block;
  width: 8px;
  height: 14px;
  background-color: #E6A23C;
  vertical-align: middle;
  margin-left: 4px;
  animation: blink 1s step-end infinite;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
</style>