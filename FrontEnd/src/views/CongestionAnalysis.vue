<template>
  <div class="analysis-container">
    <el-row :gutter="20" class="full-height">
      
      <el-col :span="14" class="full-height">
        <div class="panel dark-panel map-panel">
          <div class="panel-header">
            <div class="header-left">
              <el-icon><MapLocation /></el-icon> 空间轨迹与拥堵路段渲染
            </div>
            <div class="header-actions">
              <el-tag size="small" type="info" effect="dark">当前地域约束: 北京市</el-tag>
            </div>
          </div>
          <div class="panel-content map-wrapper" v-loading="isAnalyzing" element-loading-text="模型特征提取与时空推理中..." element-loading-background="rgba(15, 17, 23, 0.8)">
            <div ref="mapChartRef" class="echarts-map"></div>
            
            <div class="map-legend">
              <div class="legend-item"><span class="dot red"></span> 严重拥堵 (< 10km/h)</div>
              <div class="legend-item"><span class="dot yellow"></span> 缓行 (10-30km/h)</div>
              <div class="legend-item"><span class="dot green"></span> 畅通 (> 30km/h)</div>
            </div>
          </div>
        </div>
      </el-col>

      <el-col :span="10" class="full-height">
        <div class="panel dark-panel control-panel">
          <div class="panel-header">
            <el-icon><Cpu /></el-icon> 多模态拥堵溯源控制台
          </div>
          <div class="panel-content flex-col">

            <div class="section-box upload-box">
              <div class="box-title" style="margin-bottom: 8px;">第一步：上传待分析的北京路段轨迹</div>
              <el-upload
                class="mini-upload"
                action="#"
                :auto-upload="false"
                :show-file-list="false"
                :on-change="handleFileUpload"
                :disabled="isAnalyzing"
              >
                <el-button type="primary" size="small" plain :icon="UploadFilled">
                  {{ currentFile ? '已加载: ' + currentFile.name : '点击选择轨迹文件 (.csv/.plt)' }}
                </el-button>
              </el-upload>
            </div>
            
            <div class="section-box" :class="{ 'blur-mask': !currentFile }">
              <div class="box-title">第二步：选择多模态解析模型</div>
              <div class="model-selector">
                <div 
                  v-for="mod in ['exp1', 'exp2', 'exp3', 'exp4']" 
                  :key="mod"
                  class="model-btn"
                  :class="{ active: activeModel === mod }"
                  @click="runAnalysis(mod)"
                >
                  <span class="name">{{ mod.toUpperCase() }}</span>
                  <span class="desc">{{ getModelDesc(mod) }}</span>
                </div>
              </div>
            </div>

            <div class="section-box flex-1" :class="{ 'blur-mask': !hasResult }">
              <div class="box-title">第三步：该路段交通方式构成解析</div>
              <div ref="pieChartRef" class="echarts-pie"></div>
            </div>

            <div class="section-box report-box" :class="{ 'blur-mask': !hasResult }">
              <div class="box-title ai-title">
                <el-icon><Document /></el-icon> 溯源智能洞察
              </div>
              <div class="ai-content">
                <div v-if="!aiReportText" class="empty-text">等待运行模型分析...</div>
                <div v-else class="typing-text" v-html="aiReportText"></div>
              </div>
            </div>

          </div>
        </div>
      </el-col>

    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { MapLocation, Cpu, Document, UploadFilled } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { trajectoryApi } from '../api/trajectory'

// --- 状态变量 ---
const activeModel = ref('')
const isAnalyzing = ref(false)
const hasResult = ref(false)
const aiReportText = ref('')
const currentFile = ref<any>(null)

// --- ECharts 引用 ---
const mapChartRef = ref<HTMLElement | null>(null)
const pieChartRef = ref<HTMLElement | null>(null)
let mapChart: echarts.ECharts | null = null
let pieChart: echarts.ECharts | null = null

const getModelDesc = (mod: string) => {
  if (mod === 'exp1') return '纯轨迹基线'
  if (mod === 'exp2') return '+ OSM路网'
  if (mod === 'exp3') return '+ 天气环境'
  if (mod === 'exp4') return '动态损失优化'
  return ''
}

// 接收用户上传的文件
const handleFileUpload = (uploadFile: any) => {
  currentFile.value = uploadFile.raw
  ElMessage.success('轨迹文件加载成功，请选择下方模型进行拥堵溯源分析')
  runAnalysis('exp4')
}

// 辅助翻译英文类别到中文
const translateMode = (mode: string) => {
  const map: Record<string, string> = { 'car': '私家车', 'bus': '公交车', 'walk': '步行', 'bike': '自行车', 'subway': '地铁', 'train': '火车' }
  return map[mode.toLowerCase()] || mode
}

// 兜底假数据 (仅在后端解析失败时使用)
const generateFallbackPieData = (modelId: string) => {
  if (modelId === 'exp2') return [{ value: 45, name: '私家车' }, { value: 30, name: '公交车' }, { value: 15, name: '自行车' }, { value: 10, name: '步行' }]
  if (modelId === 'exp3') return [{ value: 50, name: '私家车' }, { value: 35, name: '公交车' }, { value: 5, name: '自行车' }, { value: 10, name: '步行' }]
  return [{ value: 55, name: '私家车' }, { value: 38, name: '公交车' }, { value: 2, name: '自行车' }, { value: 5, name: '步行' }]
}

// 动态生成不同模型的 AI 报告
const generateAIReport = (modelId: string, modeName: string, confValue: number) => {
  let report = ''
  if (modelId === 'exp1') {
    report = `<b>[Exp1 纯轨迹基线]</b><br/>模型完全依赖运动学特征进行判定。当前判定该路段主要交通流为 <b>${modeName}</b>，置信概率为 <b>${confValue}%</b>。由于缺乏路网上下文，该置信度在复杂路况下可能存在虚高现象。`
  } else if (modelId === 'exp2') {
    report = `<b>[Exp2 OSM语义融合]</b><br/>引入空间拓扑后，模型能够捕捉轨迹与公交站点/路口的几何关系。判定结果：<b>${modeName}</b>，置信概率为 <b>${confValue}%</b>。您可以观察到置信度相比 Exp1 的显著变化。`
  } else if (modelId === 'exp3') {
    report = `<b>[Exp3 气象环境解耦]</b><br/>当前时空的气象环境特征已注入。判定结果：<b>${modeName}</b> (概率 <b>${confValue}%</b>)。恶劣天气特征可能会降低非机动车的预测权重，使模型判定更加谨慎。`
  } else if (modelId === 'exp4') {
    report = `<b>[Exp4 终极推断]</b><br/>Focal Loss 优化完成。最终判定该拥堵源交通流为：<b>${modeName}</b> (确信度 <b>${confValue}%</b>)。模型已有效抑制对常见类别的过度拟合，展现出最真实的分类边界。`
  }
  simulateTyping(report)
}
// --- ECharts 真实地图轨迹渲染 ---
const renderRealTrajectory = (points: any[]) => {
  if (!mapChart) return

  const echartsData = points.map(p => {
    if (Array.isArray(p)) return [p[0], p[1]]; 
    return [p.lng || p.lon || p.x || p[0], p.lat || p.y || p[1]];
  }).filter(p => p[0] !== undefined && p[1] !== undefined);

  mapChart.setOption({
    xAxis: { type: 'value', scale: true, show: false }, // 自适应真实经纬度
    yAxis: { type: 'value', scale: true, show: false },
    series: [
      {
        type: 'line', 
        data: echartsData,
        lineStyle: { color: '#F56C6C', width: 4 }, // 拥堵溯源页面，线画成红色更符合主题
        smooth: false, 
        symbol: 'none', 
        name: '真实拥堵轨迹'
      }
    ]
  }, true);
}
// --- 核心：调用真实后端进行推理对比 ---
const runAnalysis = async (modelId: string) => {
  if (!currentFile.value) {
    ElMessage.warning('请先在顶部上传待分析的轨迹文件！')
    return
  }
  if (activeModel.value === modelId && hasResult.value) return 
  
  activeModel.value = modelId
  isAnalyzing.value = true
  aiReportText.value = ''
  
  try {
    const response = await trajectoryApi.predict(currentFile.value, modelId)
    const result = response as any
    
    console.log(`========== [${modelId}] 模型真实返回数据 ==========`, result)
    
    // 1. 获取真实的分类结果和置信度
    const mode = result.predicted_mode || 'unknown'
    const modeName = translateMode(mode)
    const confValue = result.confidence ? Number((result.confidence * 100).toFixed(1)) : 100
    
    // 2. 将饼图改造为：展示模型在该推断上的“置信度分布”
    const pieData = [
      { name: `模型确信是 [${modeName}]`, value: confValue },
      { name: '其他交通方式概率 (不确定性)', value: Number((100 - confValue).toFixed(1)) }
    ]

    hasResult.value = true
    updatePieChart(pieData, modeName, confValue)
    
    // 3. 把那 1000 多个真实的经纬度点画到地图上！
    if (result.points && result.points.length > 0) {
      renderRealTrajectory(result.points)
    }

    // 4. 将真实的置信度数字拼接到报告中
    generateAIReport(modelId, modeName, confValue) 
    ElMessage.success(`${modelId.toUpperCase()} 推理完成！置信度: ${confValue}%`)

  } catch (error) {
    ElMessage.error(`调用 ${modelId} 模型失败，请检查服务状态`)
    console.error('Analysis Error:', error)
  } finally {
    isAnalyzing.value = false
  }
}

// --- 图表渲染 ---
const initMapChart = () => {
  if (!mapChartRef.value) return
  mapChart = echarts.init(mapChartRef.value)
  const trajectoryPoints = Array.from({ length: 100 }, (_, i) => {
    const x = 116.46 + Math.random() * 0.005; const y = 39.90 + (i * 0.0002) + Math.random() * 0.001;
    const speed = (i > 40 && i < 70) ? Math.random() * 8 : 20 + Math.random() * 30; 
    let color = '#67C23A' 
    if (speed < 10) color = '#F56C6C' 
    else if (speed < 30) color = '#E6A23C' 
    return { value: [x, y], itemStyle: { color } }
  })

  mapChart.setOption({
    backgroundColor: 'transparent', xAxis: { type: 'value', scale: true, show: false }, yAxis: { type: 'value', scale: true, show: false },
    series: [
      { type: 'scatter', symbolSize: 6, data: trajectoryPoints, animationDelay: (idx: number) => idx * 20 },
      { type: 'line', data: trajectoryPoints.map(p => p.value), lineStyle: { color: 'rgba(255,255,255,0.2)', width: 2, type: 'dashed' }, smooth: true, symbol: 'none' }
    ]
  })
}

// --- ECharts 饼图 (重构为置信度展示) ---
const updatePieChart = (data: { name: string; value: number }[], modeName: string, confValue: number) => {
  if (!pieChartRef.value) return
  if (!pieChart) pieChart = echarts.init(pieChartRef.value)
  
  // 根据不同的置信度给主颜色：高于85%绿色，70-85%蓝色，低于70%橙色
  const mainColor = confValue > 85 ? '#67C23A' : (confValue > 70 ? '#4A90E2' : '#E6A23C')
  
  pieChart.setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c}%' },
    legend: { bottom: '0', textStyle: { color: '#e5eaf3' } },
    color: [mainColor, '#909399'], // 主色调 vs 灰色(不确定性)
    series: [{
      type: 'pie', radius: ['50%', '75%'], avoidLabelOverlap: false,
      itemStyle: { borderRadius: 8, borderColor: '#1a1f2e', borderWidth: 2 },
      label: { show: true, position: 'center', formatter: `${modeName}\n${confValue}%`, color: mainColor, fontSize: 16, fontWeight: 'bold' },
      data: data, animationType: 'scale', animationEasing: 'elasticOut'
    }]
  })
}

let typingTimer: any = null
const simulateTyping = (text: string) => {
  if (typingTimer) clearInterval(typingTimer)
  aiReportText.value = text 
}

const handleResize = () => { mapChart?.resize(); pieChart?.resize() }

onMounted(async () => {
  await nextTick()
  initMapChart()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  mapChart?.dispose()
  pieChart?.dispose()
})
</script>

<style scoped>
/* 样式部分保持一致 */
.analysis-container { height: 100%; padding-bottom: 20px;}
.full-height { height: 100%; }
.panel { border-radius: 12px; height: 100%; display: flex; flex-direction: column; overflow: hidden; }
.dark-panel { background: rgba(26, 31, 46, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); }

.panel-header { padding: 16px 20px; font-size: 16px; font-weight: 600; color: #e5eaf3; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: space-between;}
.header-left { display: flex; align-items: center; gap: 8px;}
.panel-content { padding: 20px; flex: 1; overflow-y: auto; position: relative;}
.flex-col { display: flex; flex-direction: column; gap: 16px; }
.flex-1 { flex: 1; min-height: 220px;}

.map-wrapper { padding: 0; background: #0f1117; background-image: radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px); background-size: 20px 20px;}
.echarts-map { width: 100%; height: 100%; }
.map-legend { position: absolute; bottom: 20px; right: 20px; background: rgba(0,0,0,0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(4px);}
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #e5eaf3; margin-bottom: 8px;}
.legend-item:last-child { margin-bottom: 0;}
.dot { width: 10px; height: 10px; border-radius: 50%; }
.dot.red { background: #F56C6C; box-shadow: 0 0 5px #F56C6C;}
.dot.yellow { background: #E6A23C; box-shadow: 0 0 5px #E6A23C;}
.dot.green { background: #67C23A; box-shadow: 0 0 5px #67C23A;}

.section-box { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 16px; border: 1px solid rgba(255,255,255,0.03); display: flex; flex-direction: column; transition: all 0.3s;}
.box-title { font-size: 14px; color: #a3a6ad; margin-bottom: 12px; font-weight: 500;}
.blur-mask { opacity: 0.3; pointer-events: none; filter: blur(2px); }

/* 上传区样式 */
.upload-box { padding: 12px 16px; border-color: rgba(74, 144, 226, 0.2); background: rgba(74, 144, 226, 0.02);}
:deep(.mini-upload .el-upload) { width: 100%; }
:deep(.mini-upload .el-button) { width: 100%; justify-content: flex-start;}

.model-selector { display: flex; gap: 12px; }
.model-btn { flex: 1; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 10px 6px; text-align: center; cursor: pointer; transition: all 0.3s; display: flex; flex-direction: column; gap: 4px;}
.model-btn .name { font-size: 14px; font-weight: bold; color: #e5eaf3;}
.model-btn .desc { font-size: 11px; color: #909399;}
.model-btn:hover:not(.disabled) { background: rgba(74, 144, 226, 0.1); border-color: rgba(74, 144, 226, 0.4);}
.model-btn.active { background: rgba(74, 144, 226, 0.2); border-color: #4A90E2; box-shadow: 0 0 10px rgba(74,144,226,0.2);}
.model-btn.active .name { color: #4A90E2;}
.model-btn.disabled { cursor: not-allowed; opacity: 0.4; background: repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(255,255,255,0.05) 5px, rgba(255,255,255,0.05) 10px);}

.echarts-pie { width: 100%; height: 100%; }

.report-box { border-color: rgba(103, 194, 58, 0.3); background: rgba(103, 194, 58, 0.05); }
.ai-title { color: #67C23A; display: flex; align-items: center; gap: 6px;}
.ai-content { min-height: 70px; font-size: 13px; line-height: 1.6; color: #e5eaf3;}
.empty-text { color: #909399; font-style: italic; text-align: center; margin-top: 20px;}
.typing-text { padding: 8px; background: rgba(0,0,0,0.3); border-radius: 6px; border-left: 3px solid #67C23A;}
:deep(b) { color: #4A90E2; }
</style>