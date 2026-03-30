<template>
  <div class="dashboard-container">
    <el-row :gutter="20" class="full-height">
      
      <el-col :span="6" class="full-height">
        <div class="panel dark-panel feature-panel">
          <div class="panel-header">
            <el-icon><DataLine /></el-icon> 多模态数据与 49维特征体系
          </div>
          <div class="panel-content">
            <div class="data-source-cards">
              <div class="source-card">
                <span class="label">轨迹点量级</span>
                <span class="value">百万级</span>
              </div>
              <div class="source-card">
                <span class="label">覆盖路网</span>
                <span class="value">北京 OSM</span>
              </div>
              <div class="source-card">
                <span class="label">气象跨度</span>
                <span class="value">5年历史</span>
              </div>
            </div>
            
            <div class="tree-container">
              <el-tree
                :data="featureTreeData"
                :props="defaultProps"
                default-expand-all
                class="custom-tree"
              >
                <template #default="{ node, data }">
                  <span class="custom-tree-node">
                    <el-icon v-if="data.icon" :class="data.color"><component :is="data.icon" /></el-icon>
                    <span class="node-label">{{ node.label }}</span>
                    <el-tag v-if="data.dim" size="small" class="dim-tag" effect="dark" type="info">{{ data.dim }}维</el-tag>
                  </span>
                </template>
              </el-tree>
            </div>
          </div>
        </div>
      </el-col>

      <el-col :span="12" class="full-height">
        <div class="panel middle-panel">
          <div class="experiment-selector">
            <div 
              v-for="exp in experiments" 
              :key="exp.id"
              class="exp-step"
              :class="{ active: activeExp === exp.id }"
              @click="switchExperiment(exp.id)"
            >
              <div class="step-marker"></div>
              <div class="step-info">
                <h4>{{ exp.name }}</h4>
                <p>{{ exp.desc }}</p>
              </div>
            </div>
          </div>

          <div class="charts-container">
            <div class="chart-wrapper dark-panel">
              <div class="panel-header small">全局性能演进 (Accuracy & Macro-F1)</div>
              <div ref="trendChartRef" class="echarts-box"></div>
            </div>
            <div class="chart-wrapper dark-panel mt-15">
              <div class="panel-header small">长尾类别追踪 (火车 / 地铁 F1-Score)</div>
              <div ref="longTailChartRef" class="echarts-box"></div>
            </div>
          </div>
        </div>
      </el-col>

      <el-col :span="6" class="full-height">
        <div class="panel dark-panel insight-panel">
          <div class="panel-header">
            <el-icon><Aim /></el-icon> 实验结论与深度洞察
          </div>
          <div class="panel-content insight-content">
            
            <transition name="fade-slide" mode="out-in">
              <div :key="activeExp" class="active-insight">
                <div class="insight-badge">{{ currentExpData.name }}</div>
                <h3 class="insight-title">{{ currentExpData.insightTitle }}</h3>
                
                <div class="insight-metrics">
                  <div class="metric-item">
                    <span class="label">Accuracy</span>
                    <span class="value text-blue">{{ currentExpData.acc }}%</span>
                  </div>
                  <div class="metric-item">
                    <span class="label">Macro-F1</span>
                    <span class="value text-orange">{{ currentExpData.f1 }}%</span>
                  </div>
                </div>

                <div class="insight-desc">
                  <ul>
                    <li v-for="(point, index) in currentExpData.insightPoints" :key="index">
                      {{ point }}
                    </li>
                  </ul>
                </div>

                <div class="insight-footer" v-if="currentExpData.id !== 'exp1'">
                  <el-icon><WarningFilled /></el-icon> 
                  <span class="footer-text">受限于依赖特征，该模型限定北京区域应用</span>
                </div>
                <div class="insight-footer success" v-else>
                  <el-icon><CircleCheckFilled /></el-icon> 
                  <span class="footer-text">纯轨迹特征驱动，具备全国跨城泛化能力</span>
                </div>
              </div>
            </transition>

          </div>
        </div>
      </el-col>

    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, onUnmounted, nextTick } from 'vue'
import { DataLine, Aim, Location, Cloudy, TrendCharts, WarningFilled, CircleCheckFilled } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
// 引入你提供的真实接口文件 (请确保路径正确，根据你的项目结构可能是 ../api/experiment 或 @/api/experiment)
import { experimentApi } from '../api/experiment'
import { ElMessage } from 'element-plus'

// --- 状态管理 ---
const activeExp = ref('exp4')
const isLoading = ref(true) // 新增加载状态

// 四个实验的基础信息 (保持静态，因为这是页面的固定导航骨架)
const experiments = [
  { id: 'exp1', name: 'Exp1: 基础基线', desc: '纯轨迹特征' },
  { id: 'exp2', name: 'Exp2: 空间语义', desc: '+ OSM路网' },
  { id: 'exp3', name: 'Exp3: 环境耦合', desc: '+ 气象数据' },
  { id: 'exp4', name: 'Exp4: 算法突破', desc: '+ 损失优化' }
]

// 动态响应式数据源 (替换了原本写死的假数据)
const expDetails = ref<Record<string, any>>({
  'exp1': { id: 'exp1', name: 'Exp 1 纯轨迹基线模型', acc: 0, f1: 0, insightTitle: '验证运动学特征的跨城泛化底座', insightPoints: ['仅提取速度、加速度等 27 维特征。', '极易混淆公交与私家车。', '长尾类别识别率极低。'] },
  'exp2': { id: 'exp2', name: 'Exp 2 融合 OSM 空间语义', acc: 0, f1: 0, insightTitle: '打破孤岛，建立空间坐标系映射', insightPoints: ['引入 15 维路网拓扑与 POI 距离特征。', '准确率实现显著跃升。', '解决“低速私家车”与“公交车”混淆问题。'] },
  'exp3': { id: 'exp3', name: 'Exp 3 耦合气象环境数据', acc: 0, f1: 0, insightTitle: '增强复杂天气扰动下的模型鲁棒性', insightPoints: ['注入历史气象特征。', '提升了恶劣天气下出行方式改变的解释力。', '有效纠正了雨雪天气下非机动车误判。'] },
  'exp4': { id: 'exp4', name: 'Exp 4 动态损失函数优化', acc: 0, f1: 0, insightTitle: '攻克样本不均衡，平衡长尾精度', insightPoints: ['特征维度保持 49 维不变，引入 Focal Loss。', '整体 Macro-F1 逼近理想关口。', '火车与地铁等长尾类别的 F1-Score 获得极大提升。'] }
})

const currentExpData = computed(() => expDetails.value[activeExp.value])

// 图表动态数据
const chartData = ref({
  categories: ['Exp1', 'Exp2', 'Exp3', 'Exp4'],
  acc: [0, 0, 0, 0],
  f1: [0, 0, 0, 0],
  trainF1: [0, 0, 0, 0],
  subwayF1: [0, 0, 0, 0]
})

// 图标映射
const iconMap: Record<string, any> = { TrendCharts, Location, Aim, Cloudy }
const featureTreeData = [
  { label: '49维 多模态特征体系', icon: 'TrendCharts', color: 'text-blue',
    children: [
      { label: '轨迹运动学特征', dim: 27, icon: 'Location', color: 'text-blue', children: [{ label: '点级特征 (9维)' }, { label: '段级统计特征 (18维)' }] },
      { label: 'OSM 空间语义特征', dim: 15, icon: 'Aim', color: 'text-orange', children: [{ label: '道路类型独热编码' }, { label: '路口与红绿灯密度' }, { label: '周边POI距离' }] },
      { label: '气象环境特征', dim: 7, icon: 'Cloudy', color: 'text-blue', children: [{ label: '温湿度与降水量' }, { label: '风速与能见度' }] }
    ]
  }
]
const defaultProps = { children: 'children', label: 'label' }

function switchExperiment(id: string) {
  if (isLoading.value) return;
  activeExp.value = id
}

// ECharts 实例化
const trendChartRef = ref<HTMLElement | null>(null)
const longTailChartRef = ref<HTMLElement | null>(null)
let trendChart: echarts.ECharts | null = null
let longTailChart: echarts.ECharts | null = null

const initCharts = () => {
  if (trendChartRef.value && longTailChartRef.value) {
    trendChart = echarts.init(trendChartRef.value)
    longTailChart = echarts.init(longTailChartRef.value)
    updateCharts()
  }
}

const updateCharts = () => {
  if (!trendChart || !longTailChart || isLoading.value) return

  const expIndex = experiments.findIndex(e => e.id === activeExp.value)
  
  trendChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { top: 40, right: 30, bottom: 30, left: 40 },
    xAxis: { type: 'category', data: chartData.value.categories, axisLine: { lineStyle: { color: '#606266' } } },
    yAxis: [
      { type: 'value', min: 80, max: 100, splitLine: { lineStyle: { color: '#303133', type: 'dashed' } } },
      { type: 'value', min: 75, max: 100, splitLine: { show: false } }
    ],
    series: [
      { name: 'Accuracy', type: 'line', yAxisIndex: 0, data: chartData.value.acc, smooth: true, symbolSize: 8, itemStyle: { color: '#4A90E2' } },
      { 
        name: 'Macro-F1', type: 'bar', yAxisIndex: 1, barWidth: '40%',
        data: chartData.value.f1.map((val, idx) => ({
          value: val,
          itemStyle: { color: idx === expIndex ? '#FA8C16' : 'rgba(250, 140, 22, 0.3)' }
        }))
      }
    ]
  })

  longTailChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['火车 F1', '地铁 F1'], textStyle: { color: '#909399' } },
    grid: { top: 40, right: 20, bottom: 30, left: 40 },
    xAxis: { type: 'category', data: chartData.value.categories, axisLine: { lineStyle: { color: '#606266' } } },
    yAxis: { type: 'value', min: 40, max: 100, splitLine: { lineStyle: { color: '#303133', type: 'dashed' } } },
    series: [
      { name: '火车 F1', type: 'line', data: chartData.value.trainF1, smooth: true, itemStyle: { color: '#F56C6C' } },
      { name: '地铁 F1', type: 'line', data: chartData.value.subwayF1, smooth: true, itemStyle: { color: '#67C23A' } }
    ]
  })
}

// --- 核心：请求真实后端数据 ---
const loadBackendData = async () => {
  isLoading.value = true;
  try {
    const ids = ['exp1', 'exp2', 'exp3', 'exp4'];
    
    // 并发请求四个实验的报告接口
    const reports = await Promise.all(
      ids.map(id => experimentApi.getExperimentReport(id).catch(err => {
        console.warn(`获取 ${id} 报告失败`, err);
        return null; // 容错处理
      }))
    );

    reports.forEach((rawReport, index) => {
      if (!rawReport) return;
      const expId = ids[index];
      if (!expId) return; // 修复 TS2538: expId 可能为 undefined 的报错
      
      // 修复 TS7053: 将返回值断言为 any，绕过由于动态 key（如 "macro avg", "火车"）引起的类型报错
      const report = rawReport as any;
      
      // 提取核心指标
      const acc = report.accuracy ? Number((report.accuracy * 100).toFixed(1)) : 0;
      const macroF1 = report['macro avg'] && report['macro avg']['f1-score'] ? Number((report['macro avg']['f1-score'] * 100).toFixed(1)) : 0;
      
      // 更新文字卡片数据
      expDetails.value[expId].acc = acc;
      expDetails.value[expId].f1 = macroF1;

      // 更新图表数据
      chartData.value.acc[index] = acc;
      chartData.value.f1[index] = macroF1;

      // 提取长尾类别指标
      const trainData = report['train'] || report['火车'] || report['Train'] || { 'f1-score': 0 };
      const subwayData = report['subway'] || report['地铁'] || report['Subway'] || { 'f1-score': 0 };
      
      chartData.value.trainF1[index] = Number((trainData['f1-score'] * 100).toFixed(1));
      chartData.value.subwayF1[index] = Number((subwayData['f1-score'] * 100).toFixed(1));
    });

    updateCharts();
  } catch (error) {
    ElMessage.error('无法连接到后端引擎，请确保 PyTorch 后端已启动');
    console.error('API Error:', error);
  } finally {
    isLoading.value = false;
  }
}

watch(activeExp, () => {
  updateCharts()
})

const handleResize = () => {
  trendChart?.resize()
  longTailChart?.resize()
}

onMounted(async () => {
  await nextTick()
  initCharts() // 先渲染空图表壳子
  window.addEventListener('resize', handleResize)
  
  // 发起真实请求
  await loadBackendData()
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  trendChart?.dispose()
  longTailChart?.dispose()
})
</script>

<style scoped>
.dashboard-container { height: 100%; padding-bottom: 20px;}
.full-height { height: 100%; }
.panel { border-radius: 12px; height: 100%; display: flex; flex-direction: column; overflow: hidden; }
.dark-panel { background: rgba(26, 31, 46, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); }

.panel-header { padding: 16px 20px; font-size: 16px; font-weight: 600; color: #e5eaf3; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; gap: 8px; }
.panel-header.small { font-size: 14px; padding: 12px 16px; border: none;}
.panel-content { padding: 20px; flex: 1; overflow-y: auto; }
.panel-content::-webkit-scrollbar { width: 4px; }
.panel-content::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

/* 左栏样式 */
.data-source-cards { display: flex; gap: 12px; margin-bottom: 24px; }
.source-card { flex: 1; background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.05);}
.source-card .label { font-size: 12px; color: #909399; margin-bottom: 4px; }
.source-card .value { font-size: 14px; color: #e5eaf3; font-weight: bold; }

.tree-container { background: rgba(0,0,0,0.1); border-radius: 8px; padding: 12px; }
.custom-tree { background: transparent; color: #e5eaf3; }
:deep(.el-tree-node__content:hover) { background-color: rgba(74, 144, 226, 0.1); }
:deep(.el-tree-node:focus > .el-tree-node__content) { background-color: transparent; }
.custom-tree-node { display: flex; align-items: center; font-size: 14px; gap: 8px; }
.text-blue { color: #4A90E2; }
.text-orange { color: #FA8C16; }
.dim-tag { margin-left: auto; transform: scale(0.9); }

/* 中栏样式 */
.middle-panel { gap: 20px; }
.experiment-selector { display: flex; gap: 12px; background: rgba(26, 31, 46, 0.5); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);}
.exp-step { flex: 1; display: flex; flex-direction: column; gap: 8px; cursor: pointer; padding: 12px; border-radius: 8px; transition: all 0.3s; position: relative;}
.exp-step::after { content: ''; position: absolute; right: -6px; top: 20px; width: 12px; height: 1px; background: rgba(255,255,255,0.1); }
.exp-step:last-child::after { display: none; }
.step-marker { height: 4px; width: 100%; background: rgba(255,255,255,0.1); border-radius: 2px; transition: all 0.3s; }
.step-info h4 { margin: 0; font-size: 14px; color: #909399; transition: color 0.3s;}
.step-info p { margin: 4px 0 0 0; font-size: 12px; color: #606266; }
.exp-step:hover { background: rgba(255,255,255,0.02); }
.exp-step.active .step-marker { background: #4A90E2; box-shadow: 0 0 8px #4A90E2; }
.exp-step.active .step-info h4 { color: #e5eaf3; font-weight: bold;}
.exp-step.active .step-info p { color: #4A90E2; }

.charts-container { display: flex; flex-direction: column; flex: 1; gap: 16px; }
.chart-wrapper { flex: 1; display: flex; flex-direction: column; padding: 10px; border-radius: 12px;}
.echarts-box { flex: 1; width: 100%; min-height: 200px; }

/* 右栏解读样式 */
.insight-content { display: flex; flex-direction: column; }
.active-insight { display: flex; flex-direction: column; gap: 20px; height: 100%;}
.insight-badge { display: inline-block; padding: 4px 12px; background: rgba(74, 144, 226, 0.15); color: #4A90E2; border-radius: 4px; font-size: 13px; font-weight: bold; width: fit-content; border: 1px solid rgba(74, 144, 226, 0.3);}
.insight-title { margin: 0; font-size: 18px; color: #e5eaf3; line-height: 1.4; }
.insight-metrics { display: flex; gap: 20px; background: rgba(0,0,0,0.2); padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.03);}
.metric-item { display: flex; flex-direction: column; gap: 4px; }
.metric-item .label { font-size: 12px; color: #909399; }
.metric-item .value { font-size: 24px; font-weight: bold; font-family: monospace;}

.insight-desc ul { margin: 0; padding-left: 20px; color: #a3a6ad; font-size: 14px; line-height: 1.8; }
.insight-desc li { margin-bottom: 8px; }

.insight-footer { margin-top: auto; display: flex; align-items: center; gap: 8px; padding: 12px; background: rgba(245, 108, 108, 0.1); border: 1px solid rgba(245, 108, 108, 0.2); border-radius: 8px; color: #F56C6C; font-size: 13px; }
.insight-footer.success { background: rgba(103, 194, 58, 0.1); border-color: rgba(103, 194, 58, 0.2); color: #67C23A; }

/* 动画效果 */
.fade-slide-enter-active, .fade-slide-leave-active { transition: all 0.3s ease; }
.fade-slide-enter-from { opacity: 0; transform: translateX(10px); }
.fade-slide-leave-to { opacity: 0; transform: translateX(-10px); }
</style>