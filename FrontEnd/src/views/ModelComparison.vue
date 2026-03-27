<template>
  <div class="model-comparison">
    <div class="experiments-header">
      <h2>实验对比</h2>
      <p>对比不同特征组合对交通方式识别性能的影响</p>
    </div>

    <div class="experiments-cards">
      <div
        v-for="exp in experiments"
        :key="exp.id"
        class="experiment-card"
        :class="{ active: selectedExperiment?.id === exp.id, completed: exp.status === 'completed' }"
        @click="selectExperiment(exp)"
      >
        <div class="card-header">
          <h3>{{ exp.name }}</h3>
          <el-tag :type="getStatusType(exp.status)" size="small">
            {{ getStatusText(exp.status) }}
          </el-tag>
        </div>

        <div class="card-body">
          <p class="description">{{ exp.description }}</p>
          <div class="features">
            <span
              v-for="(feature, index) in exp.features"
              :key="index"
              class="feature-tag"
            >
              {{ feature }}
            </span>
          </div>
        </div>

        <div v-if="exp.status === 'completed'" class="card-footer">
          <el-button type="primary" @click="showDetails(exp)">
            查看详情
          </el-button>
        </div>
      </div>
    </div>

    <el-dialog v-model="showDetailsDialog" :title="`${selectedExp?.name} 详细评估结果`" width="1200px">
      <div v-loading="loading" class="exp-details">
        <div class="details-layout">
          <div class="details-left">
            <h4>混淆矩阵</h4>
            <div class="confusion-matrix-container">
              <img
                v-if="confusionMatrixUrl"
                :src="confusionMatrixUrl"
                alt="混淆矩阵"
                class="confusion-matrix-img"
              />
              <el-empty v-else description="暂无混淆矩阵数据" />
            </div>
          </div>

          <div class="details-right">
            <div class="accuracy-display">
              <div class="accuracy-label">总体准确率</div>
              <div class="accuracy-value">
                {{ currentReport?.accuracy ? (currentReport.accuracy * 100).toFixed(2) : 'N/A' }}%
              </div>
            </div>

            <div class="metrics-chart">
              <h5>各类别性能指标</h5>
              <div ref="metricsChartRef" class="chart-container"></div>
            </div>

            <div class="f1-display">
              <h5>Macro 平均 F1 分数</h5>
              <div class="f1-value">
                {{ calculateMacroF1() }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>

    <div class="comparison-table">
      <h3>实验对比说明</h3>
      <el-table :data="comparisonData" border>
        <el-table-column prop="feature" label="特征维度" width="180" />
        <el-table-column prop="exp1" label="Exp1 (纯轨迹)" width="180" />
        <el-table-column prop="exp2" label="Exp2 (+OSM)" width="180" />
        <el-table-column prop="exp3" label="Exp3 (+天气)" width="180" />
        <el-table-column prop="exp4" label="Exp4 (对比学习)" width="180" />
      </el-table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { useExperimentStore } from '@/stores/experiment'
import { experimentApi } from '@/api/experiment'
import * as echarts from 'echarts'
import type { ECharts } from 'echarts'
import type { ExperimentInfo } from '@/types'

const experimentStore = useExperimentStore()

const experiments = computed(() => experimentStore.experiments)
const selectedExperiment = computed(() => experimentStore.selectedExperiment)
const currentReport = computed(() => experimentStore.currentReport)
const loading = computed(() => experimentStore.loading)

const showDetailsDialog = ref(false)
const selectedExp = ref<ExperimentInfo | null>(null)
const confusionMatrixUrl = ref('')
const metricsChartRef = ref<HTMLElement>()
let metricsChart: ECharts | null = null

onMounted(() => {
  experimentStore.loadExperiments()
})

watch(showDetailsDialog, async (show) => {
  if (show && selectedExp.value) {
    await loadExperimentData(selectedExp.value.id)
  } else {
    if (confusionMatrixUrl.value) {
      URL.revokeObjectURL(confusionMatrixUrl.value)
      confusionMatrixUrl.value = ''
    }
    if (metricsChart) {
      metricsChart.dispose()
      metricsChart = null
    }
  }
})

async function loadExperimentData(expId: string) {
  confusionMatrixUrl.value = ''
  if (metricsChart) {
    metricsChart.dispose()
    metricsChart = null
  }
  
  await Promise.all([
    experimentStore.loadExperimentReport(expId),
    loadConfusionMatrix(expId),
  ])
  
  await nextTick()
  initMetricsChart()
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
  const statusMap: Record<string, string> = {
    completed: 'success',
    not_trained: 'info',
    training: 'warning',
  }
  return statusMap[status] || 'info'
}

function getStatusText(status: string): string {
  const statusMap: Record<string, string> = {
    completed: '已完成',
    not_trained: '未训练',
    training: '训练中',
  }
  return statusMap[status] || '未知'
}

function initMetricsChart() {
  if (!metricsChartRef.value || !currentReport.value) {
    return
  }

  if (metricsChart) {
    metricsChart.dispose()
  }

  metricsChart = echarts.init(metricsChartRef.value)

  const report = currentReport.value
  const modes = Object.keys(report.precision || {})
  
  const precisionData = modes.map(m => report.precision?.[m] || 0)
  const recallData = modes.map(m => report.recall?.[m] || 0)
  const f1Data = modes.map(m => report.f1_score?.[m] || 0)

  const modeNames: Record<string, string> = {
    Walk: '步行',
    Bike: '自行车',
    'Car & taxi': '汽车/出租',
    Bus: '公交',
    Subway: '地铁',
    Train: '火车',
    Airplane: '飞机',
  }

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
    },
    legend: {
      data: ['精确率', '召回率', 'F1分数'],
      top: 10,
    },
    xAxis: {
      type: 'category',
      data: modes.map(m => modeNames[m] || m),
      axisLabel: {
        interval: 0,
        rotate: 30,
      },
    },
    yAxis: {
      type: 'value',
      max: 1,
      axisLabel: {
        formatter: '{value}',
      },
    },
    series: [
      {
        name: '精确率',
        type: 'bar',
        data: precisionData,
        itemStyle: {
          color: '#4A90E2',
        },
      },
      {
        name: '召回率',
        type: 'bar',
        data: recallData,
        itemStyle: {
          color: '#52C41A',
        },
      },
      {
        name: 'F1分数',
        type: 'bar',
        data: f1Data,
        itemStyle: {
          color: '#FA8C16',
        },
      },
    ],
  }

  metricsChart.setOption(option)
}

function calculateMacroF1(): string {
  if (!currentReport.value || !currentReport.value.f1_score) return 'N/A'
  
  const f1Scores = Object.values(currentReport.value.f1_score)
  const macroF1 = f1Scores.reduce((sum, val) => sum + val, 0) / f1Scores.length
  
  return macroF1.toFixed(3)
}

const comparisonData = [
  {
    feature: '轨迹特征',
    exp1: '9维',
    exp2: '9维',
    exp3: '9维',
    exp4: '9维',
  },
  {
    feature: '段级统计',
    exp1: '18维',
    exp2: '18维',
    exp3: '18维',
    exp4: '18维',
  },
  {
    feature: 'OSM空间',
    exp1: '-',
    exp2: '11维',
    exp3: '15维',
    exp4: '15维',
  },
  {
    feature: '天气特征',
    exp1: '-',
    exp2: '-',
    exp3: '12维',
    exp4: '12维',
  },
  {
    feature: '损失函数',
    exp1: 'Cross Entropy',
    exp2: 'Cross Entropy',
    exp3: 'Cross Entropy',
    exp4: 'Focal Loss',
  },
  {
    feature: '模型结构',
    exp1: 'Bi-LSTM',
    exp2: 'Bi-LSTM',
    exp3: 'Bi-LSTM',
    exp4: 'Bi-LSTM + CCA',
  },
]
</script>

<style scoped>
.model-comparison {
  padding: 24px;
  background: #0f1117;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.experiments-header {
  margin-bottom: 32px;
  text-align: center;
}

.experiments-header h2 {
  margin: 0 0 12px 0;
  font-size: 28px;
  color: #fff;
}

.experiments-header p {
  margin: 0;
  font-size: 14px;
  color: #909399;
}

.experiments-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
  margin-bottom: 40px;
}

.experiment-card {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.3s;
  border: 2px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
}

.experiment-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.experiment-card.active {
  border-color: #4A90E2;
}

.experiment-card.completed {
  border-left: 4px solid #67C23A;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.card-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}

.card-body {
  margin-bottom: 16px;
}

.description {
  font-size: 14px;
  color: #e5e8eb;
  line-height: 1.6;
  margin-bottom: 12px;
}

.features {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.feature-tag {
  background: rgba(74, 144, 226, 0.15);
  color: #4A90E2;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 12px;
}

.card-footer {
  text-align: center;
}

.exp-details {
  min-height: 400px;
}

.details-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.details-left h4,
.details-right h4 {
  margin: 0 0 16px 0;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
}

.confusion-matrix-container {
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.confusion-matrix-img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}

.accuracy-display {
  text-align: center;
  padding: 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  margin-bottom: 24px;
}

.accuracy-label {
  font-size: 14px;
  color: #fff;
  margin-bottom: 8px;
}

.accuracy-value {
  font-size: 48px;
  font-weight: 700;
  color: #fff;
}

.metrics-chart {
  margin-bottom: 24px;
}

.chart-container {
  width: 100%;
  height: 400px;
}

.f1-display {
  text-align: center;
  padding: 20px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.f1-display h5 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #909399;
}

.f1-value {
  font-size: 36px;
  font-weight: 600;
  color: #4A90E2;
}

.comparison-table {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.comparison-table h3 {
  margin: 0 0 16px 0;
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}
</style>
