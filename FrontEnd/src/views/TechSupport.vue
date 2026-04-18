<template>
  <div class="tech-support-container">
    <!-- <PageHeader title="多模态大模型底座" subtitle="全景透视四阶时空图网络的推演精度、特征离散度及鲁棒性指标" /> -->

    <el-skeleton :loading="initialLoading" animated :rows="15" class="relative-z">
      <template #default>
        <div class="tech-layout">
          
          <div class="cyber-sidebar">
            <div class="sidebar-header">
              <span class="header-text">中枢控制序列</span>
              <div class="deco-line"></div>
            </div>
            
            <div class="nav-list">
              <div 
                v-for="(item, index) in navItems" 
                :key="index"
                class="nav-item"
                :class="{ 'is-active': activeMenu === item.id }"
                @click="handleSelect(item.id)"
              >
                <div class="nav-idx">0{{ index + 1 }}</div>
                <div class="nav-text">{{ item.name }}</div>
                <div class="nav-corner"></div>
              </div>
            </div>
          </div>

          <div class="cyber-content">
            
            <div v-if="activeMenu === '1'" class="fui-panel fade-in">
              <div class="panel-header">
                <h3 class="panel-title">递进实验模型宏观性能演进</h3>
              </div>
              
              <div ref="overviewChartRef" class="chart-box-large"></div>
              
              <div class="table-wrapper">
                <table class="fui-table">
                  <thead>
                    <tr>
                      <th>演进阶段</th>
                      <th>全局准确率</th>
                      <th>宏平均 F1 分数</th>
                      <th>加权平均 F1 分数</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, idx) in overviewTableData" :key="idx">
                      <td><span class="exp-tag">{{ row.exp }}</span></td>
                      <td class="text-green font-bold">{{ row.accuracy }}</td>
                      <td class="text-cyan font-bold">{{ row.macroF1 }}</td>
                      <td class="text-orange font-bold">{{ row.weightedF1 }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div v-if="activeMenu === '2'" class="fui-panel fade-in">
              <div class="panel-header">
                <h3 class="panel-title">细粒度交通方式性能雷达诊断</h3>
              </div>
              
              <div class="version-tabs">
                <div class="tab-label">切换网络层级：</div>
                <div 
                  v-for="exp in availableExps" :key="exp"
                  class="version-tab" 
                  :class="{ 'active': selectedExpCat === exp }"
                  @click="selectedExpCat = exp; renderCategoryChart()"
                >
                  {{ exp.toUpperCase() }}
                </div>
              </div>
              
              <div v-if="!evalData[selectedExpCat]" class="empty-state"><div class="bracket-alert">[ 未检索到诊断日志 ]</div></div>
              <div v-else ref="categoryChartRef" class="chart-box-large"></div>
            </div>

            <div v-if="activeMenu === '3'" class="fui-panel fade-in">
              <div class="panel-header">
                <h3 class="panel-title">跨模态特征混淆交叉矩阵</h3>
              </div>
              
              <div class="version-tabs">
                <div class="tab-label">切换网络层级：</div>
                <div 
                  v-for="exp in ['exp1', 'exp2', 'exp3', 'exp4']" :key="exp"
                  class="version-tab" 
                  :class="{ 'active': selectedExpCM === exp }"
                  @click="selectedExpCM = exp; renderCMChart()"
                >
                  {{ exp.toUpperCase() }}
                </div>
              </div>
              
              <div v-if="csvLoading" class="empty-state">
                <div class="loader-bar"><div class="loader-fill"></div></div>
                <p class="text-cyan mt-4 font-bold">引擎正在解析千万级预测张量，请稍候...</p>
              </div>
              <div v-else-if="csvError" class="empty-state"><div class="bracket-alert text-red">[ 数据断流，无法构建张量矩阵 ]</div></div>
              <div v-else ref="cmChartRef" class="chart-box-xl heatmap-box"></div>
            </div>

            <div v-if="activeMenu === '4'" class="fui-panel fade-in">
              <div class="panel-header">
                <h3 class="panel-title">底层推理置信度离散分布特征</h3>
              </div>

              <div class="version-tabs">
                <div class="tab-label">切换网络层级：</div>
                <div 
                  v-for="exp in ['exp1', 'exp2', 'exp3', 'exp4']" :key="exp"
                  class="version-tab" 
                  :class="{ 'active': selectedExpConf === exp }"
                  @click="selectedExpConf = exp; renderConfidenceChart()"
                >
                  {{ exp.toUpperCase() }}
                </div>
              </div>

              
              <div v-if="csvLoading" class="empty-state">
                <div class="loader-bar"><div class="loader-fill"></div></div>
                <p class="text-cyan mt-4 font-bold">引擎正在解析千万级预测张量，请稍候...</p>
              </div>
              <div v-else-if="csvError" class="empty-state"><div class="bracket-alert text-red">[ 数据断流，无法构建分布模型 ]</div></div>
              <div v-else ref="confChartRef" class="chart-box-large"></div>
            </div>

          </div>
        </div>
      </template>
    </el-skeleton>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import Papa from 'papaparse'
import PageHeader from '@/components/PageHeader.vue'
const initialLoading = ref(true)
const csvLoading = ref(false)
const csvError = ref(false)

const activeMenu = ref('1')
const navItems = [
  { id: '1', name: '宏观性能演进' },
  { id: '2', name: '多模态分类诊断' },
  { id: '3', name: '特征混淆矩阵' },
  { id: '4', name: '置信度离散分布' }
]

const availableExps = ['exp1', 'exp2', 'exp3', 'exp4']

const selectedExpCat = ref('exp4')
const selectedExpCM = ref('exp3')
const selectedExpConf = ref('exp3')

const evalData = reactive({})
const csvCache = reactive({})

const overviewChartRef = ref(null)
const categoryChartRef = ref(null)
const cmChartRef = ref(null)
const confChartRef = ref(null)
let currentChartInstance = null

const modeDict = {
  'Walk': '步行', 'Bike': '骑行', 'Bus': '常规公交', 
  'Car & taxi': '小汽车', 'Subway': '城市轨道', 'Train': '铁路客运'
}

// ==================== Exp4 数据 ====================
const exp4ConfusionMatrix = [
  [133,   2,   0,   0,   0,   0],   // Walk true
  [  2, 144,   4,   0,   0,   1],   // Bike true
  [  0,   5, 107,  15,   0,   0],   // Bus true
  [  0,   0,  14,  25,   0,   1],   // Car & taxi true
  [  0,   0,   0,   0,  18,   0],   // Subway true
  [  0,   0,   0,   0,   0,  24]    // Train true
]

// 各类别的置信度五数概括（min, Q1, median, Q3, max）
const exp4ConfidenceStats = {
  'Walk': [0.270, 0.560, 0.640, 0.700, 0.743],
  'Bike': [0.276, 0.540, 0.610, 0.650, 0.697],
  'Bus': [0.280, 0.470, 0.550, 0.630, 0.752],
  'Car & taxi': [0.344, 0.470, 0.550, 0.620, 0.756],
  'Subway': [0.379, 0.660, 0.780, 0.820, 0.841],
  'Train': [0.344, 0.760, 0.800, 0.810, 0.825]
}

const transportModes = Object.keys(modeDict)

const loadEvalReports = async () => {
  for (const exp of availableExps) {
    try {
      const res = await fetch(`/evaluation_report_${exp}.json`)
      if (res.ok) evalData[exp] = await res.json()
    } catch (err) { console.warn(`加载 ${exp} 失败`) }
  }
}

const loadCSVData = async (exp) => {
  if (exp === 'exp4') return null
  if (csvCache[exp]) return csvCache[exp]
  
  csvLoading.value = true
  csvError.value = false
  try {
    const res = await fetch(`/predictions_with_geo_${exp}.csv`)
    if (!res.ok) throw new Error('Fetch failed')
    const text = await res.text()
    
    return new Promise((resolve) => {
      Papa.parse(text, {
        header: true, dynamicTyping: true, skipEmptyLines: true,
        complete: (results) => {
          csvCache[exp] = results.data
          resolve(results.data)
        },
        error: () => { csvError.value = true; resolve(null) }
      })
    })
  } catch (err) {
    csvError.value = true
    return null
  } finally {
    csvLoading.value = false
  }
}

const getPercentile = (data, p) => {
  const index = (data.length - 1) * p
  const lower = Math.floor(index)
  const upper = lower + 1
  const weight = index % 1
  if (upper >= data.length) return data[lower]
  return data[lower] * (1 - weight) + data[upper] * weight
}

const calcBoxplotStats = (values) => {
  if (!values || values.length === 0) return []
  values.sort((a, b) => a - b)
  return [ values[0], getPercentile(values, 0.25), getPercentile(values, 0.5), getPercentile(values, 0.75), values[values.length - 1] ]
}

const overviewTableData = computed(() => {
  return availableExps.map(exp => {
    const d = evalData[exp]
    return {
      exp: `${exp.toUpperCase()}`,
      accuracy: d ? d.accuracy?.toFixed(4) : '数据载入中...',
      macroF1: d ? d['macro avg']?.['f1-score']?.toFixed(4) : '-',
      weightedF1: d ? d['weighted avg']?.['f1-score']?.toFixed(4) : '-'
    }
  })
})

const destroyCurrentChart = () => {
  if (currentChartInstance) {
    currentChartInstance.dispose()
    currentChartInstance = null
  }
}

// ================= 图表 1: 宏观总览 =================
const renderOverviewChart = () => {
  destroyCurrentChart()
  if (!overviewChartRef.value) return
  
  const exps = overviewTableData.value.map(item => item.exp)
  const accData = overviewTableData.value.map(item => isNaN(item.accuracy) ? 0 : parseFloat(item.accuracy))
  const f1Data = overviewTableData.value.map(item => isNaN(item.macroF1) ? 0 : parseFloat(item.macroF1))

  currentChartInstance = echarts.init(overviewChartRef.value)
  currentChartInstance.setOption({
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(6, 12, 28, 0.95)', borderColor: '#00f0ff', textStyle: { color: '#fff', fontSize: 16, fontWeight: 'bold' }, padding: 15 },
    legend: { data: ['全局准确率', '宏平均 F1 分数'], textStyle: { color: '#cbd5e1', fontSize: 15, fontWeight: 'bold' }, top: 0, itemWidth: 20, itemHeight: 12 },
    grid: { left: '2%', right: '2%', bottom: '5%', top: 60, containLabel: true },
    xAxis: { type: 'category', data: exps, axisLabel: { color: '#00f0ff', fontSize: 15, fontWeight: 'bold', margin: 15 }, axisLine: { lineStyle: { color: '#1e293b', width: 2 } }, axisTick: { show: false } },
    yAxis: { type: 'value', min: 0.7, max: 1.0, axisLabel: { color: '#94a3b8', fontSize: 14, fontWeight: 'bold' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)', type: 'dashed', width: 1.5 } } },
    series: [
      { name: '全局准确率', type: 'bar', barWidth: '20%', itemStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1, [{offset:0,color:'#00FF88'},{offset:1,color:'rgba(0,255,136,0.1)'}]), borderRadius: [4,4,0,0] }, data: accData },
      { name: '宏平均 F1 分数', type: 'bar', barWidth: '20%', itemStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1, [{offset:0,color:'#00f0ff'},{offset:1,color:'rgba(0,240,255,0.1)'}]), borderRadius: [4,4,0,0] }, data: f1Data }
    ]
  })
}

// ================= 图表 2: 雷达图  =================
const renderCategoryChart = () => {
  destroyCurrentChart()
  const d = evalData[selectedExpCat.value]
  if (!d || !categoryChartRef.value) return

  const indicators = []
  const precisionData = []
  const recallData = []
  const f1Data = []

  transportModes.forEach(mode => {
    if (d[mode]) {
      indicators.push({ name: modeDict[mode], max: 1.0 })
      precisionData.push(d[mode].precision || 0)
      recallData.push(d[mode].recall || 0)
      f1Data.push(d[mode]['f1-score'] || 0)
    }
  })

  currentChartInstance = echarts.init(categoryChartRef.value)
  currentChartInstance.setOption({
    tooltip: { trigger: 'item', backgroundColor: 'rgba(6, 12, 28, 0.95)', borderColor: '#00f0ff', textStyle: { color: '#fff', fontSize: 15, fontWeight: 'bold' }, padding: 15 },
    legend: { data: ['精确率', '召回率', 'F1 分数'], textStyle: { color: '#cbd5e1', fontSize: 15, fontWeight: 'bold' }, bottom: 0, itemWidth: 20, itemHeight: 12 },
    radar: {
      indicator: indicators, shape: 'polygon', radius: '65%', center: ['50%', '45%'],
      splitArea: { areaStyle: { color: ['rgba(0, 240, 255, 0.02)', 'rgba(0, 240, 255, 0.05)', 'rgba(0, 240, 255, 0.08)', 'rgba(0, 240, 255, 0.15)'] } },
      axisLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.5)', width: 1.5 } },
      splitLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.5)', width: 1.5 } },
      axisName: { color: '#00f0ff', fontSize: 15, fontWeight: 'bold', textShadow: '0 0 8px #00f0ff' }
    },
    series: [{
      type: 'radar',
      data: [
        { value: precisionData, name: '精确率', itemStyle: { color: '#FFDD00' }, lineStyle: { width: 3 } },
        { value: recallData, name: '召回率', itemStyle: { color: '#CC33FF' }, lineStyle: { width: 3 } },
        { value: f1Data, name: 'F1 分数', itemStyle: { color: '#00FF88' }, lineStyle: { width: 3 }, areaStyle: { color: 'rgba(0, 255, 136, 0.4)' } }
      ]
    }]
  })
}

// ================= 图表 3: 混淆矩阵  =================
const renderCMChart = async () => {
  destroyCurrentChart()
  if (selectedExpCM.value === 'exp4') {
    if (!cmChartRef.value) return
    const matrix = exp4ConfusionMatrix
    let maxVal = 0
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        if (matrix[i][j] > maxVal) maxVal = matrix[i][j]
      }
    }
    const heatmapData = []
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        heatmapData.push([j, i, matrix[i][j]])
      }
    }
    const chineseModes = transportModes.map(m => modeDict[m])
    currentChartInstance = echarts.init(cmChartRef.value)
    currentChartInstance.setOption({
      tooltip: { 
      position: 'top', backgroundColor: 'rgba(6, 12, 28, 0.95)', borderColor: '#00f0ff', padding: 15,
      textStyle: { color: '#fff', fontSize: 15 }, 
      formatter: (p) => `<div style="font-weight:bold;margin-bottom:8px;">实际: ${chineseModes[p.data[1]]} <span style="color:#64748b">➡</span> 预测: ${chineseModes[p.data[0]]}</div>落入特征数: <span style="color:#00f0ff; font-size:18px;">${p.data[2]}</span>` 
    },
      grid: { top: 20, bottom: 80, left: 100, right: 30 },
      xAxis: { type: 'category', data: chineseModes, axisLabel: { color: '#cbd5e1', fontSize: 14, fontWeight: 'bold', margin: 15}, splitArea: { show: true }, axisTick: { show: false } },
      yAxis: { type: 'category', data: chineseModes, axisLabel: { color: '#cbd5e1', fontSize: 14, fontWeight: 'bold', margin: 15 }, splitArea: { show: true }, axisTick: { show: false } },
      visualMap: { min: 0, max: maxVal, calculable: true, orient: 'horizontal', left: 'center', bottom: 10, inRange: { color: ['#0A0F1E', '#00f0ff', '#00FF88', '#FFDD00'] }, textStyle: { color: '#e2e8f0', fontSize: 13, fontWeight: 'bold' }, itemWidth: 20 },
      series: [{ type: 'heatmap', data: heatmapData, label: { show: true, color: '#fff', fontSize: 15, fontWeight: 'bold', textShadow: '0 0 3px #000' }, itemStyle: { borderColor: '#050a14', borderWidth: 4 } }]
    })
    return
  }
  const csvData = await loadCSVData(selectedExpCM.value)
  if (!csvData || csvData.length === 0 || !cmChartRef.value) return

  const matrix = Array(6).fill(0).map(() => Array(6).fill(0))
  let maxVal = 0

  csvData.forEach(row => {
    if (row.true_label && row.pred_label) {
      const tIdx = transportModes.indexOf(row.true_label)
      const pIdx = transportModes.indexOf(row.pred_label)
      if (tIdx >= 0 && pIdx >= 0) {
        matrix[tIdx][pIdx]++
        if (matrix[tIdx][pIdx] > maxVal) maxVal = matrix[tIdx][pIdx]
      }
    }
  })

  const heatmapData = []
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) { heatmapData.push([j, i, matrix[i][j]]) }
  }
  const chineseModes = transportModes.map(m => modeDict[m])

  currentChartInstance = echarts.init(cmChartRef.value)
  currentChartInstance.setOption({
    tooltip: { 
      position: 'top', backgroundColor: 'rgba(6, 12, 28, 0.95)', borderColor: '#00f0ff', padding: 15,
      textStyle: { color: '#fff', fontSize: 15 }, 
      formatter: (p) => `<div style="font-weight:bold;margin-bottom:8px;">实际: ${chineseModes[p.data[1]]} <span style="color:#64748b">➡</span> 预测: ${chineseModes[p.data[0]]}</div>落入特征数: <span style="color:#00f0ff; font-size:18px;">${p.data[2]}</span>` 
    },
    grid: { top: 20, bottom: 80, left: 100, right: 30 },
    xAxis: { type: 'category', data: chineseModes, axisLabel: { color: '#cbd5e1', fontSize: 14, fontWeight: 'bold', margin: 15 }, splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)','rgba(0,0,0,0)'] } }, axisTick: { show: false } },
    yAxis: { type: 'category', data: chineseModes, axisLabel: { color: '#cbd5e1', fontSize: 14, fontWeight: 'bold', margin: 15 }, splitArea: { show: true }, axisTick: { show: false } },
    visualMap: { min: 0, max: maxVal, calculable: true, orient: 'horizontal', left: 'center', bottom: 10, inRange: { color: ['#0A0F1E', '#00f0ff', '#00FF88', '#FFDD00'] }, textStyle: { color: '#e2e8f0', fontSize: 13, fontWeight: 'bold' }, itemWidth: 20 },
    series: [{ type: 'heatmap', data: heatmapData, label: { show: true, color: '#fff', fontSize: 15, fontWeight: 'bold', textShadow: '0 0 3px #000' }, itemStyle: { borderColor: '#050a14', borderWidth: 4 } }]
  })
}

// ================= 图表 4: 置信度箱线图 =================
const renderConfidenceChart = async () => {
  destroyCurrentChart()
  if (selectedExpConf.value === 'exp4') {
    if (!confChartRef.value) return
    const chineseModes = transportModes.map(m => modeDict[m])
    const boxplotData = transportModes.map(mode => exp4ConfidenceStats[mode])
    currentChartInstance = echarts.init(confChartRef.value)
    currentChartInstance.setOption({
      tooltip: { trigger: 'item', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(6, 12, 28, 0.95)', borderColor: '#00f0ff', padding: 15, textStyle: { color: '#fff', fontSize: 14 } },
      grid: { top: 30, left: 60, right: 30, bottom: 40 },
      xAxis: { type: 'category', data: chineseModes, axisLabel: { color: '#00f0ff', fontSize: 15, fontWeight: 'bold', margin: 15 }, axisLine: { lineStyle: { color: '#1e293b', width: 2 } }, axisTick: { show: false } },
      yAxis: { type: 'value', min: 0, max: 1.0, axisLabel: { color: '#94a3b8', fontSize: 14, fontWeight: 'bold' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)', type: 'dashed', width: 1.5 } } },
      series: [{
        name: '离散特征', type: 'boxplot', data: boxplotData,
        itemStyle: { color: 'rgba(0,240,255,0.15)', borderColor: '#00f0ff', borderWidth: 2 },
        boxWidth: [25, 45]
      }]
    })
    return
  }
  const csvData = await loadCSVData(selectedExpConf.value)
  if (!csvData || csvData.length === 0 || !confChartRef.value) return

  const confMap = {}
  transportModes.forEach(m => confMap[m] = [])

  csvData.forEach(row => {
    if (row.pred_label && row.confidence !== undefined && confMap[row.pred_label]) {
      confMap[row.pred_label].push(row.confidence)
    }
  })

  const boxplotData = transportModes.map(m => calcBoxplotStats(confMap[m]))
  const chineseModes = transportModes.map(m => modeDict[m])

  currentChartInstance = echarts.init(confChartRef.value)
  currentChartInstance.setOption({
    tooltip: { trigger: 'item', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(6, 12, 28, 0.95)', borderColor: '#00f0ff', padding: 15, textStyle: { color: '#fff', fontSize: 14 } },
    grid: { top: 30, left: 60, right: 30, bottom: 40 },
    xAxis: { type: 'category', data: chineseModes, axisLabel: { color: '#00f0ff', fontSize: 15, fontWeight: 'bold', margin: 15 }, axisLine: { lineStyle: { color: '#1e293b', width: 2 } }, axisTick: { show: false } },
    yAxis: { type: 'value', min: 0, max: 1.0, axisLabel: { color: '#94a3b8', fontSize: 14, fontWeight: 'bold' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)', type: 'dashed', width: 1.5 } } },
    series: [{
      name: '离散特征', type: 'boxplot', data: boxplotData,
      itemStyle: { color: 'rgba(0,240,255,0.15)', borderColor: '#00f0ff', borderWidth: 2 },
      boxWidth: [25, 45]
    }]
  })
}

const handleSelect = async (index) => {
  activeMenu.value = index
  await nextTick()
  if (index === '1') renderOverviewChart()
  if (index === '2') renderCategoryChart()
  if (index === '3') renderCMChart()
  if (index === '4') renderConfidenceChart()
}

const handleResize = () => { if (currentChartInstance) currentChartInstance.resize() }

onMounted(async () => {
  await loadEvalReports()
  initialLoading.value = false
  await nextTick()
  renderOverviewChart()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  destroyCurrentChart()
})
</script>

<style scoped>
.tech-support-container {
  position: relative;
  min-height: calc(100vh - 64px);
  background-color: transparent;
  padding: 30px 40px; 
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  color: #e2e8f0;
  overflow: hidden;
}
.relative-z { position: relative; z-index: 10; }

.cyber-grid-bg {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: 
    linear-gradient(rgba(0, 240, 255, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 240, 255, 0.05) 1px, transparent 1px);
  background-size: 40px 40px;
  opacity: 0.6;
}

.cyber-header {
  display: flex; justify-content: space-between; align-items: flex-end;
  margin-bottom: 40px; border-bottom: 2px solid rgba(0, 240, 255, 0.15); padding-bottom: 15px;
}
.header-left { display: flex; align-items: stretch; gap: 20px; }
.header-deco-block { width: 10px; height: 45px; background: #00f0ff; box-shadow: 0 0 15px #00f0ff; }
.main-title { font-size: 28px; font-weight: 800; margin: 0 0 6px 0; letter-spacing: 2px; }
.sub-title { font-size: 15px; color: #94a3b8; margin: 0; letter-spacing: 1px;}
.text-glow-cyan { color: #fff; text-shadow: 0 0 15px rgba(0, 240, 255, 0.8), 0 0 30px rgba(0, 240, 255, 0.4); }

.header-right { display: flex; gap: 15px; }
.status-badge {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 16px; background: rgba(0, 240, 255, 0.08); border: 1px solid rgba(0, 240, 255, 0.3);
  font-size: 14px; font-weight: bold; color: #e2e8f0; letter-spacing: 1px;
}
.blink-dot { width: 8px; height: 8px; background: #00FF88; border-radius: 50%; box-shadow: 0 0 10px #00FF88; animation: blink 1s infinite alternate; }
@keyframes blink { 0% { opacity: 0.3; } 100% { opacity: 1; } }

.tech-layout { display: flex; gap: 40px; align-items: flex-start; }

.cyber-sidebar { width: 280px; flex-shrink: 0; display: flex; flex-direction: column; gap: 15px; }
.sidebar-header { display: flex; align-items: center; gap: 15px; margin-bottom: 10px; }
.header-text { font-size: 16px; color: #00f0ff; font-weight: 800; letter-spacing: 2px; }
.deco-line { flex: 1; height: 2px; background: rgba(0, 240, 255, 0.3); }

.nav-list { display: flex; flex-direction: column; gap: 12px; }
.nav-item {
  position: relative; padding: 20px 24px; cursor: pointer;
  display: flex; align-items: center; gap: 15px;
  background: rgba(10, 15, 30, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 4px; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.nav-idx { font-family: 'Din', monospace; font-size: 18px; font-weight: bold; color: #475569; transition: color 0.3s; }
.nav-text { font-size: 16px; font-weight: bold; color: #94a3b8; transition: color 0.3s; letter-spacing: 1px; z-index: 2; }
.nav-corner { position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: #00f0ff; box-shadow: 0 0 15px #00f0ff; transform: scaleY(0); transition: transform 0.3s; }

.nav-item:hover { border-color: rgba(0, 240, 255, 0.3); background: rgba(0, 240, 255, 0.05); transform: translateX(5px); }
.nav-item.is-active { background: rgba(0, 240, 255, 0.1); border-color: rgba(0, 240, 255, 0.6); transform: translateX(10px); box-shadow: 0 8px 25px rgba(0,0,0,0.5); }
.nav-item.is-active .nav-idx { color: #00f0ff; text-shadow: 0 0 10px #00f0ff; }
.nav-item.is-active .nav-text { color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.4); }
.nav-item.is-active .nav-corner { transform: scaleY(1); }

.cyber-content { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.fui-panel {
  position: relative; flex: 1; min-height: 650px;
  background: rgba(6, 12, 28, 0.65); backdrop-filter: blur(12px);
  border: 1px solid rgba(0, 240, 255, 0.2); padding: 40px;
  box-shadow: inset 0 0 40px rgba(0, 240, 255, 0.03), 0 10px 40px rgba(0,0,0,0.6);
  display: flex; flex-direction: column; border-radius: 8px;
}

.panel-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid rgba(255,255,255,0.05); padding-bottom: 20px; margin-bottom: 35px; }
.panel-title { font-size: 22px; color: #fff; font-weight: 800; margin: 0; letter-spacing: 1px; display: flex; align-items: center;}
.panel-title::before { content: ''; display: inline-block; width: 6px; height: 22px; background: #00f0ff; margin-right: 15px; box-shadow: 0 0 12px #00f0ff; }
.panel-deco { font-family: 'Din', monospace; font-size: 14px; color: #475569; letter-spacing: 3px; font-weight: bold;}

.chart-box-large { flex: 1; width: 100%; min-height: 350px; }
.chart-box-xl { flex: 1; width: 100%; min-height: 450px; } 

.version-tabs { display: flex; align-items: center; gap: 15px; }
.tab-label { font-size: 15px; color: #94a3b8; font-weight: bold; margin-right: 10px; }
.version-tab {
  padding: 5px 24px; font-size: 15px; font-weight: bold; color: #94a3b8; cursor: pointer;
  background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1);
}
.version-tab:hover { background: rgba(0, 240, 255, 0.05); color: #e2e8f0; }
.version-tab.active { 
  background: rgba(0, 240, 255, 0.15); border-color: #00f0ff; color: #00f0ff; 
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.3); text-shadow: 0 0 8px #00f0ff; 
}
.version-tab::before {  content: ''; display: block; transform: skewX(15deg); }

.table-wrapper { margin-top: 40px; background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; overflow: hidden;}
.fui-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 16px; }
.fui-table th { background: rgba(0, 240, 255, 0.08); color: #00f0ff; padding: 20px 24px; font-weight: 800; border-bottom: 2px solid rgba(0, 240, 255, 0.3); letter-spacing: 1px; font-size: 15px;}
.fui-table td { padding: 18px 24px; border-bottom: 1px dashed rgba(255,255,255,0.08); color: #e2e8f0; }
.fui-table tbody tr { transition: background 0.3s; }
.fui-table tbody tr:hover { background: rgba(0, 240, 255, 0.05); }

.exp-tag { display: inline-block; padding: 4px 12px; background: rgba(0, 240, 255, 0.1); border: 1px solid #00f0ff; color: #00f0ff; font-weight: bold; border-radius: 4px; font-size: 14px; letter-spacing: 1px;}
.text-green { color: #00FF88; text-shadow: 0 0 8px rgba(0,255,136,0.4);}
.text-cyan { color: #00f0ff; text-shadow: 0 0 8px rgba(0,240,255,0.4);}
.text-orange { color: #FF9900; text-shadow: 0 0 8px rgba(255,153,0,0.4);}
.text-red { color: #FF0033; }
.font-bold { font-weight: 900; font-family: 'Din', monospace, sans-serif; font-size: 18px; }

.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 400px; }
.bracket-alert { font-family: monospace; color: #64748b; font-size: 16px; font-weight: bold; letter-spacing: 3px; animation: pulseOpacity 2s infinite; }
@keyframes pulseOpacity { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.loader-bar { width: 250px; height: 4px; background: rgba(255,255,255,0.1); position: relative; overflow: hidden; margin-bottom: 20px; border-radius: 2px;}
.loader-fill { position: absolute; left: -80px; width: 80px; height: 100%; background: #00f0ff; box-shadow: 0 0 15px #00f0ff; animation: loadBar 1.2s infinite ease-in-out; }
@keyframes loadBar { 100% { left: 100%; } }

.fade-in { animation: fadeIn 0.5s cubic-bezier(0.4, 0, 0.2, 1) forwards; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

@media (max-width: 1200px) {
  .tech-layout { flex-direction: column; }
  .cyber-sidebar { width: 100%; flex-direction: row; align-items: center; }
  .nav-list { flex-direction: row; flex-wrap: wrap; }
  .nav-item { padding: 12px 20px; }
}
</style>