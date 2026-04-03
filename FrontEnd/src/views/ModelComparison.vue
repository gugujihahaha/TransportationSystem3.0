<template>
  <div class="model-comparison">

    <div class="experiments-header">
      <h2 class="main-title"><span class="text-glow">消融实验与多模态对比</span></h2>
      <p class="description">基于真实后台评估报告，深度剖析不同特征组合与损失函数对交通方式识别性能的影响</p>
    </div>

    <div class="highlight-section">
      <div class="comparison-section highlight-card">
        <div class="section-title-wrap">
          <h3><span class="icon">📈</span> 核心评估指标演进 (Accuracy vs Macro-F1)</h3>
          <div class="insight-badge text-gold">
            💡 科研亮点：Exp4 牺牲部分全局准确率，换取了整体宏平均(F1)的提升，攻克长尾难题。
          </div>
        </div>
        <div ref="evolutionChartRef" class="chart-container"></div>
      </div>

      <div class="comparison-section highlight-card">
        <div class="section-title-wrap">
          <h3><span class="icon">🎯</span> 长尾弱势类别突破 (Exp1 基线 vs Exp4 终极)</h3>
          <div class="insight-badge text-cyan">
            💡 靶向分析：Focal Loss 的引入使“单车”、“地铁”等极度不平衡样本的 F1 获得了质的飞跃。
          </div>
        </div>
        <div ref="longTailChartRef" class="chart-container"></div>
      </div>
    </div>
    <div class="experiments-cards">
      <div v-for="exp in experiments" :key="exp.id" class="experiment-card"
        :class="{ active: selectedExperiment?.id === exp.id, completed: exp.status === 'completed' }"
        @click="selectExperiment(exp)">
        <div class="card-header">
          <h3>{{ exp.name }}</h3>
          <el-tag :type="getStatusType(exp.status)" size="small">
            {{ getStatusText(exp.status) }}
          </el-tag>
        </div>

        <div class="card-body">
          <p class="description">{{ exp.description }}</p>
          <div class="features">
            <span v-for="(feature, index) in exp.features" :key="index" class="feature-tag">
              {{ feature }}
            </span>
          </div>
        </div>

        <div v-if="exp.status === 'completed'" class="card-footer">
          <el-button type="primary" class="cyber-btn" @click.stop="showDetails(exp)">
            查看混淆矩阵与详情
          </el-button>
        </div>
      </div>
    </div>

    <div class="comparison-section">
      <h3>全局准确率对比</h3>
      <div ref="accuracyChartRef" class="chart-container"></div>
    </div>

    <div class="comparison-section">
      <h3>各类别 F1 分数全景雷达</h3>
      <div class="radar-selector">
        <el-checkbox-group v-model="selectedExpsForRadar" @change="updateRadarChart">
          <el-checkbox-button v-for="exp in experiments" :key="exp.id" :label="exp.id">
            {{ exp.name }}
          </el-checkbox-button>
        </el-checkbox-group>
      </div>
      <div ref="radarChartRef" class="chart-container"></div>
    </div>

    <div class="comparison-section">
      <h3>各交通方式在不同实验中的 F1 分数</h3>
      <div class="mode-selector">
        <el-checkbox-group v-model="selectedModesForBar" @change="updateModeBarChart">
          <el-checkbox-button v-for="mode in modeList" :key="mode.key" :label="mode.key">
            {{ mode.label }}
          </el-checkbox-button>
        </el-checkbox-group>
      </div>
      <div ref="modeBarChartRef" class="chart-container"></div>
    </div>

    <div class="comparison-section">
      <h3>综合评估指标对比走势</h3>
      <div ref="metricsChartRef" class="chart-container"></div>
    </div>

    <div class="comparison-section">
      <h3>实验环境与配置白皮书</h3>
      <el-alert type="info" :closable="false"
        style="margin-bottom: 16px; background: rgba(74, 144, 226, 0.1); border: 1px solid rgba(74, 144, 226, 0.3); color: #e2e8f0;">
        <template #title>
          <strong style="color: #00f0ff;">数据划分与训练策略</strong>
        </template>
        <ul style="margin-top: 12px; padding-left: 20px; line-height: 1.8;">
          <li>训练/验证/测试集划分：70% / 10% / 20%</li>
          <li>随机种子：random_state = 42</li>
          <li>分层采样：stratify 保证类别分布一致</li>
          <li>最优模型保存标准：验证集损失最小</li>
        </ul>
      </el-alert>
      <el-table :data="comparisonData" border stripe class="custom-dark-table">
        <el-table-column prop="feature" label="架构配置项" width="200" fixed />
        <el-table-column prop="exp1" label="Exp1 (基线)" width="220" />
        <el-table-column prop="exp2" label="Exp2 (+OSM拓扑)" width="220" />
        <el-table-column prop="exp3" label="Exp3 (+气象解耦)" width="220" />
        <el-table-column prop="exp4" label="Exp4 (Focal优化)" width="240">
          <template #default="scope">
            <span v-html="scope.row.exp4"></span>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <el-dialog v-model="showDetailsDialog" :title="`${selectedExp?.name} 深度溯源评估报告`" width="1200px"
      custom-class="cyber-dialog">
      <div v-loading="loading" class="exp-details">
        <div class="details-layout">
          <div class="details-left">
            <h4 style="color: #00f0ff;">模型预测混淆矩阵</h4>
            <div class="confusion-matrix-container">
              <img v-if="confusionMatrixUrl" :src="confusionMatrixUrl" alt="混淆矩阵" class="confusion-matrix-img" />
              <el-empty v-else description="正在拉取后台评估图谱..." />
            </div>
          </div>

          <div class="details-right">
            <div class="accuracy-display">
              <div class="accuracy-label">当前模型总体准确率 (Accuracy)</div>
              <div class="accuracy-value">
                {{ currentReport?.accuracy ? (currentReport.accuracy * 100).toFixed(2) : 'N/A' }}%
              </div>
            </div>

            <div class="f1-display mb-20">
              <h5 style="color: #ffb800; font-size: 16px;">Macro 平均 F1 分数 (平衡指标)</h5>
              <div class="f1-value" style="color: #ffb800; font-size: 32px;">
                {{ calculateMacroF1() }}
              </div>
            </div>

            <div class="metrics-chart">
              <h5 style="color: #e2e8f0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">各类别 P /
                R /
                F1 指标拆解</h5>
              <div ref="singleMetricsChartRef" class="chart-container" style="height: 300px;"></div>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
defineOptions({
  name: 'ModelComparison'
})

import { ref, computed, onMounted, nextTick, watch, onUnmounted } from 'vue'
import { useExperimentStore } from '@/stores/experiment'
import { experimentApi } from '@/api/experiment'
import * as echarts from 'echarts'
import type { ECharts } from 'echarts'
import type { ExperimentInfo, EvaluationReport } from '@/types'

const experimentStore = useExperimentStore()

const experiments = computed(() => experimentStore.experiments)
const selectedExperiment = computed(() => experimentStore.selectedExperiment)
const currentReport = computed(() => experimentStore.currentReport)
const loading = computed(() => experimentStore.loading)

const showDetailsDialog = ref(false)
const selectedExp = ref<ExperimentInfo | null>(null)
const confusionMatrixUrl = ref('')
const selectedExpsForRadar = ref<string[]>(['exp1', 'exp2', 'exp3', 'exp4'])

// 原有的 Ref
const accuracyChartRef = ref<HTMLElement>()
const radarChartRef = ref<HTMLElement>()
const modeBarChartRef = ref<HTMLElement>()
const metricsChartRef = ref<HTMLElement>()
const singleMetricsChartRef = ref<HTMLElement>()

// 新增的高光图表 Ref
const evolutionChartRef = ref<HTMLElement>()
const longTailChartRef = ref<HTMLElement>()

let accuracyChart: ECharts | null = null
let radarChart: ECharts | null = null
let modeBarChart: ECharts | null = null
let metricsChart: ECharts | null = null
let singleMetricsChart: ECharts | null = null
let evolutionChart: ECharts | null = null
let longTailChart: ECharts | null = null

const allReports = ref<Record<string, EvaluationReport>>({})

const modeList = ref<{ key: string; label: string }[]>([])
const selectedModesForBar = ref<string[]>([])

const modeNames: Record<string, string> = {
  Walk: '步行',
  Bike: '自行车',
  'Car & taxi': '机动车',
  Bus: '公交',
  Subway: '地铁',
  Train: '火车',
  Airplane: '飞机',
}

const expColors = {
  exp1: '#4A90E2',
  exp2: '#52C41A',
  exp3: '#FA8C16',
  exp4: '#722ED1',
}

const expNames = {
  exp1: 'Exp1 (纯轨迹基线)',
  exp2: 'Exp2 (+OSM拓扑)',
  exp3: 'Exp3 (+气象解耦)',
  exp4: 'Exp4 (Focal 终极)',
}

onMounted(async () => {
  await experimentStore.loadExperiments()
  await loadAllReports()
  await nextTick()

  // 初始化您原有的图表
  initAccuracyChart()
  initRadarChart()
  initModeBarChart()
  initMetricsChart()

  // 初始化新增的高光图表（基于真实数据）
  initEvolutionChart()
  initLongTailChart()

  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  accuracyChart?.dispose()
  radarChart?.dispose()
  modeBarChart?.dispose()
  metricsChart?.dispose()
  singleMetricsChart?.dispose()
  evolutionChart?.dispose()
  longTailChart?.dispose()
})

const handleResize = () => {
  accuracyChart?.resize()
  radarChart?.resize()
  modeBarChart?.resize()
  metricsChart?.resize()
  singleMetricsChart?.resize()
  evolutionChart?.resize()
  longTailChart?.resize()
}

watch(() => modeList.value, () => {
  nextTick(() => {
    if (radarChart) updateRadarChart()
    if (modeBarChart) updateModeBarChart()
  })
})

// 加载后台数据
async function loadAllReports() {
  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']
  for (const expId of expIds) {
    try {
      const report = await experimentApi.getExperimentReport(expId)
      allReports.value[expId] = report
    } catch (error) {
      console.error(`加载 ${expId} 报告失败:`, error)
    }
  }

  const allModes = new Set<string>()
  Object.values(allReports.value).forEach(report => {
    if (report?.f1_score) {
      Object.keys(report.f1_score).forEach(mode => allModes.add(mode))
    }
  })

  modeList.value = Array.from(allModes).map(key => ({
    key,
    label: modeNames[key] || key
  })).sort((a, b) => {
    const order = ['Walk', 'Bike', 'Bus', 'Car & taxi', 'Subway', 'Train', 'Airplane']
    return order.indexOf(a.key) - order.indexOf(b.key)
  })

  selectedModesForBar.value = modeList.value.map(m => m.key)
}

// ========== 新增：科研高光图表 (动态读取真实报告) ==========

function initEvolutionChart() {
  if (!evolutionChartRef.value) return
  if (evolutionChart) evolutionChart.dispose()
  evolutionChart = echarts.init(evolutionChartRef.value)

  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']

  // 从加载的真实报告中提取 Acc 和 F1
  const accData = expIds.map(id => {
    const r = allReports.value[id]
    return r ? parseFloat((r.accuracy * 100).toFixed(2)) : 0
  })

  const f1Data = expIds.map(id => {
    const r = allReports.value[id]
    if (!r || !r.f1_score) return 0
    const scores = Object.values(r.f1_score)
    const macroF1 = scores.reduce((sum, val) => sum + val, 0) / scores.length
    return parseFloat((macroF1 * 100).toFixed(2))
  })

  const option = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(0,0,0,0.8)', textStyle: { color: '#fff' } },
    legend: { data: ['Accuracy (全局准确率)', 'Macro-F1 (宏平均)'], textStyle: { color: '#cbd5e1' }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '5%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: expIds.map(id => expNames[id as keyof typeof expNames]),
      axisLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.3)' } },
      axisLabel: { color: '#94a3b8', fontSize: 13 }
    },
    yAxis: {
      type: 'value',
      min: (value: any) => Math.floor(value.min - 2),
      max: (value: any) => Math.ceil(value.max + 2),
      axisLabel: { formatter: '{value}%', color: '#94a3b8' },
      splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } }
    },
    series: [
      {
        name: 'Accuracy (全局准确率)',
        type: 'line',
        data: accData,
        smooth: true,
        symbolSize: 8,
        itemStyle: { color: '#00f0ff' },
        lineStyle: { width: 3, shadowBlur: 10, shadowColor: '#00f0ff' }
      },
      {
        name: 'Macro-F1 (宏平均)',
        type: 'line',
        data: f1Data,
        smooth: true,
        symbolSize: 8,
        itemStyle: { color: '#ffb800' },
        lineStyle: { width: 3, shadowBlur: 10, shadowColor: '#ffb800' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(255, 184, 0, 0.4)' },
            { offset: 1, color: 'rgba(255, 184, 0, 0)' }
          ])
        }
      }
    ]
  }
  evolutionChart.setOption(option)
}

function initLongTailChart() {
  if (!longTailChartRef.value) return
  if (longTailChart) longTailChart.dispose()
  longTailChart = echarts.init(longTailChartRef.value)

  // 靶向分析容易混淆/样本少的类别
  const targetModes = ['Bike', 'Bus', 'Subway', 'Train']

  const exp1Data = targetModes.map(mode => {
    const r = allReports.value['exp1']
    return r && r.f1_score[mode] ? parseFloat((r.f1_score[mode] * 100).toFixed(2)) : 0
  })

  const exp4Data = targetModes.map(mode => {
    const r = allReports.value['exp4']
    return r && r.f1_score[mode] ? parseFloat((r.f1_score[mode] * 100).toFixed(2)) : 0
  })

  const option = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(0,0,0,0.8)', textStyle: { color: '#fff' } },
    legend: { data: ['Exp1 基线 F1', 'Exp4 Focal优化 F1'], textStyle: { color: '#cbd5e1' }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '5%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: targetModes.map(m => modeNames[m] || m),
      axisLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.3)' } },
      axisLabel: { color: '#94a3b8', fontSize: 13 }
    },
    yAxis: {
      type: 'value',
      name: 'F1-Score (%)',
      nameTextStyle: { color: '#94a3b8' },
      min: (value: any) => Math.max(0, Math.floor(value.min - 10)),
      splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)', type: 'dashed' } },
      axisLabel: { color: '#94a3b8' }
    },
    series: [
      {
        name: 'Exp1 基线 F1',
        type: 'bar',
        barWidth: '20%',
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(77, 144, 226, 0.8)' },
            { offset: 1, color: 'rgba(77, 144, 226, 0.2)' }
          ]),
          borderRadius: [4, 4, 0, 0]
        },
        data: exp1Data
      },
      {
        name: 'Exp4 Focal优化 F1',
        type: 'bar',
        barWidth: '20%',
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(0, 255, 136, 0.9)' },
            { offset: 1, color: 'rgba(0, 255, 136, 0.3)' }
          ]),
          borderRadius: [4, 4, 0, 0],
          shadowBlur: 10,
          shadowColor: 'rgba(0, 255, 136, 0.5)'
        },
        data: exp4Data,
        label: { show: true, position: 'top', color: '#00ff88', fontWeight: 'bold', formatter: '{c}%' }
      }
    ]
  }
  longTailChart.setOption(option)
}

// ========== 恢复您的原始功能代码 ==========

function initAccuracyChart() {
  if (!accuracyChartRef.value) return
  if (accuracyChart) accuracyChart.dispose()
  accuracyChart = echarts.init(accuracyChartRef.value)

  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']
  const accuracies = expIds.map(id => {
    const report = allReports.value[id]
    return report ? parseFloat((report.accuracy * 100).toFixed(2)) : 0
  })

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params: any) => `<strong>${params[0].axisValue}</strong><br/>${params[0].marker} 准确率: ${params[0].data}%`,
    },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: {
      type: 'category',
      data: expIds.map(id => expNames[id as keyof typeof expNames]),
      axisLabel: { interval: 0, color: '#e2e8f0' },
    },
    yAxis: { type: 'value', name: '准确率 (%)', min: 0, max: 100, splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } } },
    series: [
      {
        name: '准确率', type: 'bar', data: accuracies, barWidth: '40%',
        itemStyle: {
          color: (params: any) => ['#4A90E2', '#52C41A', '#FA8C16', '#722ED1'][params.dataIndex],
          borderRadius: [4, 4, 0, 0]
        },
        label: { show: true, position: 'top', color: '#fff', formatter: '{c}%' }
      },
    ],
  }
  accuracyChart.setOption(option)
}

function initRadarChart() {
  if (!radarChartRef.value) return
  if (radarChart) radarChart.dispose()
  radarChart = echarts.init(radarChartRef.value)
  updateRadarChart()
}

function updateRadarChart() {
  if (!radarChart) return

  const selectedExpIds = selectedExpsForRadar.value
  if (selectedExpIds.length === 0) {
    radarChart.setOption({ series: [], radar: { indicator: [] } })
    return
  }

  const modes = modeList.value.map(m => m.key)
  const modeLabels = modes.map(m => modeNames[m] || m)

  const series = selectedExpsForRadar.value.map(expId => {
    const report = allReports.value[expId]
    const data = modes.map(m => report?.f1_score?.[m] ? parseFloat((report.f1_score[m] * 100).toFixed(2)) : 0)
    return {
      name: expNames[expId as keyof typeof expNames],
      value: data,
      lineStyle: { color: expColors[expId as keyof typeof expColors] },
      itemStyle: { color: expColors[expId as keyof typeof expColors] },
      areaStyle: { color: expColors[expId as keyof typeof expColors], opacity: 0.2 },
    }
  })

  const option = {
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(0,0,0,0.8)',
      textStyle: { color: '#fff' }
    },
    legend: { data: selectedExpsForRadar.value.map(id => expNames[id as keyof typeof expNames]), bottom: 0, textStyle: { color: '#cbd5e1' } },
    radar: {
      indicator: modeLabels.map(label => ({ name: label, max: 100 })),
      radius: '65%',
      axisName: { color: '#00f0ff' },
      splitLine: { lineStyle: { color: ['rgba(0, 240, 255, 0.1)', 'rgba(0, 240, 255, 0.2)', 'rgba(0, 240, 255, 0.4)'] } },
      splitArea: { show: false }
    },
    series: [{ name: 'F1分数', type: 'radar', data: series }],
  }
  radarChart.setOption(option)
}

function initModeBarChart() {
  if (!modeBarChartRef.value) return
  if (modeBarChart) modeBarChart.dispose()
  modeBarChart = echarts.init(modeBarChartRef.value)
  updateModeBarChart()
}

function updateModeBarChart() {
  if (!modeBarChart) return
  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']
  const selectedModeKeys = selectedModesForBar.value

  const series = expIds.map(expId => {
    const report = allReports.value[expId]
    const data = selectedModeKeys.map(modeKey => report?.f1_score?.[modeKey] ? parseFloat((report.f1_score[modeKey] * 100).toFixed(2)) : 0)
    return {
      name: expNames[expId as keyof typeof expNames],
      type: 'bar',
      data: data,
      itemStyle: { color: expColors[expId as keyof typeof expColors], borderRadius: [2, 2, 0, 0] }
    }
  })

  const option = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(0,0,0,0.8)', textStyle: { color: '#fff' } },
    legend: { data: expIds.map(id => expNames[id as keyof typeof expNames]), top: 0, textStyle: { color: '#cbd5e1' } },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
    xAxis: { type: 'category', data: selectedModeKeys.map(key => modeNames[key] || key), axisLabel: { color: '#e2e8f0' } },
    yAxis: { type: 'value', name: 'F1分数 (%)', min: 0, max: 100, splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } } },
    series: series,
  }
  modeBarChart.setOption(option)
}

function initMetricsChart() {
  if (!metricsChartRef.value) return
  if (metricsChart) metricsChart.dispose()
  metricsChart = echarts.init(metricsChartRef.value)

  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']
  const metrics = ['准确率', '精确率', '召回率', 'F1分数']

  const seriesData = metrics.map((metric) => {
    const data = expIds.map(id => {
      const report = allReports.value[id]
      if (!report) return 0
      if (metric === '准确率') return parseFloat((report.accuracy * 100).toFixed(2))

      const values = Object.values(metric === '精确率' ? report.precision || {} : metric === '召回率' ? report.recall || {} : report.f1_score || {})
      const avg = values.reduce((sum, v) => sum + v, 0) / values.length
      return parseFloat((avg * 100).toFixed(2))
    })

    return { name: metric, type: 'line', data: data, smooth: true, symbolSize: 8 }
  })

  const option = {
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(0,0,0,0.8)', textStyle: { color: '#fff' } },
    legend: { data: metrics, top: 0, textStyle: { color: '#cbd5e1' } },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
    xAxis: { type: 'category', data: expIds.map(id => expNames[id as keyof typeof expNames]), axisLabel: { color: '#e2e8f0' } },
    yAxis: { type: 'value', name: '百分比 (%)', min: 70, max: 100, splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } } },
    series: seriesData,
  }
  metricsChart.setOption(option)
}

// 弹窗逻辑
watch(showDetailsDialog, async (show) => {
  if (show && selectedExp.value) {
    await loadExperimentData(selectedExp.value.id)
  } else {
    if (confusionMatrixUrl.value) {
      URL.revokeObjectURL(confusionMatrixUrl.value)
      confusionMatrixUrl.value = ''
    }
    if (singleMetricsChart) {
      singleMetricsChart.dispose()
      singleMetricsChart = null
    }
  }
})

async function loadExperimentData(expId: string) {
  confusionMatrixUrl.value = ''
  if (singleMetricsChart) { singleMetricsChart.dispose(); singleMetricsChart = null }

  // 核心：请求原有的后台报告和混淆矩阵图片！
  await Promise.all([
    experimentStore.loadExperimentReport(expId),
    loadConfusionMatrix(expId),
  ])

  await nextTick()
  initSingleMetricsChart()
}

async function loadConfusionMatrix(expId: string) {
  try {
    const blob = await experimentApi.getExperimentConfusionMatrix(expId)
    confusionMatrixUrl.value = URL.createObjectURL(blob)
  } catch (error) {
    console.error('加载混淆矩阵失败:', error)
  }
}

function showDetails(exp: ExperimentInfo) {
  selectedExp.value = exp
  showDetailsDialog.value = true
}

function selectExperiment(exp: any) {
  experimentStore.selectExperiment(exp)
}

function getStatusType(status: string): string {
  const statusMap: Record<string, string> = { completed: 'success', not_trained: 'info', training: 'warning' }
  return statusMap[status] || 'info'
}

function getStatusText(status: string): string {
  const statusMap: Record<string, string> = { completed: '已解析', not_trained: '未就绪', training: '运算中' }
  return statusMap[status] || '未知'
}

function initSingleMetricsChart() {
  if (!singleMetricsChartRef.value || !currentReport.value) return
  if (singleMetricsChart) singleMetricsChart.dispose()
  singleMetricsChart = echarts.init(singleMetricsChartRef.value)

  const report = currentReport.value
  const modes = Object.keys(report.precision || {})

  const precisionData = modes.map(m => parseFloat(((report.precision?.[m] || 0) * 100).toFixed(2)))
  const recallData = modes.map(m => parseFloat(((report.recall?.[m] || 0) * 100).toFixed(2)))
  const f1Data = modes.map(m => parseFloat(((report.f1_score?.[m] || 0) * 100).toFixed(2)))

  const option = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(0,0,0,0.8)', textStyle: { color: '#fff' } },
    legend: { data: ['精确率', '召回率', 'F1分数'], top: 0, textStyle: { color: '#cbd5e1' } },
    xAxis: { type: 'category', data: modes.map(m => modeNames[m] || m), axisLabel: { interval: 0, rotate: 30, color: '#e2e8f0' } },
    yAxis: { type: 'value', max: 100, splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } } },
    series: [
      { name: '精确率', type: 'bar', data: precisionData, itemStyle: { color: '#4A90E2', borderRadius: [2, 2, 0, 0] } },
      { name: '召回率', type: 'bar', data: recallData, itemStyle: { color: '#52C41A', borderRadius: [2, 2, 0, 0] } },
      { name: 'F1分数', type: 'bar', data: f1Data, itemStyle: { color: '#FA8C16', borderRadius: [2, 2, 0, 0] } },
    ],
  }
  singleMetricsChart.setOption(option)
}

function calculateMacroF1(): string {
  if (!currentReport.value || !currentReport.value.f1_score) return 'N/A'
  const f1Scores = Object.values(currentReport.value.f1_score)
  const macroF1 = (f1Scores.reduce((sum, val) => sum + val, 0) / f1Scores.length) * 100
  return macroF1.toFixed(2) + '%'
}

// 原始表格数据
const comparisonData = [
  { feature: '轨迹特征', exp1: 'traj_9', exp2: 'traj_21', exp3: 'traj_21', exp4: 'traj_21' },
  { feature: '段级统计', exp1: 'stats_18', exp2: 'stats_18', exp3: 'stats_18', exp4: 'stats_18' },
  { feature: 'OSM空间', exp1: '-', exp2: '✓', exp3: '✓', exp4: '✓' },
  { feature: '天气特征', exp1: '-', exp2: '-', exp3: '✓', exp4: '✓' },
  { feature: '损失函数', exp1: 'CrossEntropyLoss', exp2: 'CrossEntropyLoss', exp3: 'CrossEntropyLoss', exp4: '<span style="color:#ffb800; font-weight:bold;">FocalLoss (γ=2.0)</span>' },
  { feature: '模型结构', exp1: 'Bi-LSTM', exp2: 'Bi-LSTM', exp3: '双路Attention拼接', exp4: '双路Attention拼接' },
]
</script>

<style scoped>
.model-comparison {
  padding: 40px 5%;
  background-color: #070b19;
  flex: 1;
  min-height: 100vh;
  overflow-y: auto;
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif;
}

/* 头部发光标题 */
.experiments-header {
  text-align: center;
  margin-bottom: 40px;
}

.main-title {
  font-size: 3rem;
  font-weight: 900;
  margin-bottom: 10px;
}

.text-glow {
  color: #fff;
  text-shadow: 0 0 20px rgba(0, 240, 255, 0.6);
}

.description {
  color: #94a3b8;
  font-size: 1.1rem;
}

/* ========== 高光区域样式 ========== */
.highlight-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
  margin-bottom: 40px;
}

.highlight-card {
  border-color: rgba(0, 240, 255, 0.3) !important;
  box-shadow: 0 10px 30px rgba(0, 240, 255, 0.05);
}

.section-title-wrap {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 20px;
}

.insight-badge {
  background: rgba(0, 0, 0, 0.4);
  padding: 8px 15px;
  border-radius: 6px;
  font-size: 0.95rem;
  font-weight: bold;
  border-left: 3px solid;
}

.text-gold {
  color: #ffb800;
  border-color: #ffb800;
}

.text-cyan {
  color: #00f0ff;
  border-color: #00f0ff;
}

/* ========== 原有功能通用样式改造为科技风 ========== */
.experiments-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
  margin-bottom: 40px;
}

.experiment-card {
  background: rgba(13, 20, 40, 0.5);
  border-radius: 12px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.3s;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.experiment-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 20px rgba(0, 240, 255, 0.2);
  border-color: rgba(0, 240, 255, 0.5);
}

.experiment-card.active {
  border-color: #00f0ff;
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.card-header h3 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: #fff;
}

.description {
  font-size: 0.95rem;
  color: #cbd5e1;
  line-height: 1.6;
  margin-bottom: 12px;
}

.features {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.feature-tag {
  background: rgba(0, 240, 255, 0.1);
  color: #00f0ff;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 0.85rem;
  border: 1px solid rgba(0, 240, 255, 0.2);
}

.card-footer {
  text-align: center;
  margin-top: 15px;
}

.cyber-btn {
  background: transparent;
  border: 1px solid #00f0ff;
  color: #00f0ff;
  padding: 8px 20px;
  border-radius: 4px;
}

.cyber-btn:hover {
  background: #00f0ff;
  color: #000;
}

.comparison-section {
  background: rgba(13, 20, 40, 0.5);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 30px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.comparison-section h3 {
  margin: 0 0 20px 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: #fff;
  border-bottom: 1px solid rgba(0, 240, 255, 0.3);
  padding-bottom: 12px;
  display: flex;
  align-items: center;
}

.comparison-section h3 .icon {
  margin-right: 10px;
}

.radar-selector,
.mode-selector {
  margin-bottom: 20px;
}

.chart-container {
  width: 100%;
  height: 400px;
}

/* 表格暗色主题适配 */
:deep(.custom-dark-table) {
  background-color: transparent !important;
  color: #e2e8f0;
}

:deep(.custom-dark-table th),
:deep(.custom-dark-table tr) {
  background-color: rgba(13, 20, 40, 0.8) !important;
  border-color: rgba(255, 255, 255, 0.1) !important;
}

:deep(.custom-dark-table td) {
  border-color: rgba(255, 255, 255, 0.1) !important;
}

:deep(.custom-dark-table--striped .el-table__body tr.el-table__row--striped td) {
  background-color: rgba(255, 255, 255, 0.02) !important;
}

:deep(.el-table--enable-row-hover .el-table__body tr:hover > td) {
  background-color: rgba(0, 240, 255, 0.1) !important;
}

/* 弹窗暗色适配 */
:deep(.cyber-dialog) {
  background: #070b19;
  border: 1px solid #00f0ff;
}

:deep(.cyber-dialog .el-dialog__title) {
  color: #fff;
  font-weight: bold;
}

.exp-details {
  min-height: 400px;
}

.details-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.confusion-matrix-container {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.confusion-matrix-img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  border: 1px solid rgba(0, 240, 255, 0.3);
}

.accuracy-display {
  text-align: center;
  padding: 24px;
  background: rgba(0, 240, 255, 0.1);
  border: 1px solid #00f0ff;
  border-radius: 12px;
  margin-bottom: 24px;
}

.accuracy-label {
  font-size: 1rem;
  color: #00f0ff;
  margin-bottom: 8px;
}

.accuracy-value {
  font-size: 3rem;
  font-weight: 700;
  color: #fff;
  font-family: monospace;
}

.f1-display {
  text-align: center;
  padding: 20px;
  background: rgba(255, 184, 0, 0.05);
  border-radius: 8px;
  border: 1px dashed rgba(255, 184, 0, 0.3);
}

@media (max-width: 1200px) {
  .highlight-section {
    grid-template-columns: 1fr;
  }

  .experiments-cards {
    grid-template-columns: repeat(2, 1fr);
  }

  .details-layout {
    grid-template-columns: 1fr;
  }
}
</style>