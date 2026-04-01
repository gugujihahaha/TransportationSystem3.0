<template>
  <div class="scrollable-container">
    
    <el-row class="mb-20">
      <el-col :span="24">
        <div class="panel dark-panel">
          <div class="panel-header">
            <el-icon><DataLine /></el-icon> TrafficRec 核心算法演进 (Exp1 - Exp4 性能消融分析)
          </div>
          <div class="panel-content chart-wrapper">
            <div ref="evolutionChartRef" class="evolution-chart"></div>
          </div>
        </div>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="content-row">
      
      <el-col :span="11" class="full-height">
        <div class="panel dark-panel flex-col">
          <div class="section-box flex-1">
            <div class="box-title"><el-icon><Cpu /></el-icon> 系统技术架构 (Pipeline)</div>
            <div class="arch-content">
              <div class="arch-layer data-layer">
                <div class="layer-title">数据层 (Data & Preprocessing)</div>
                <div class="layer-items">
                  <el-tag type="info" effect="dark">GPS 轨迹时空解析</el-tag>
                  <el-tag type="warning" effect="dark">OSM 路网拓扑查询</el-tag>
                  <el-tag type="success" effect="dark">历史气象特征对齐</el-tag>
                </div>
              </div>
              <div class="arch-arrow"><el-icon><Bottom /></el-icon></div>
              
              <div class="arch-layer engine-layer">
                <div class="layer-title">算法引擎层 (PyTorch / ML Models)</div>
                <div class="layer-items">
                  <el-tag type="danger" effect="dark">49维多模态特征融合</el-tag>
                  <el-tag type="danger" effect="dark">Focal Loss 不均衡优化</el-tag>
                  <el-tag type="danger" effect="dark">时序序列注意力机制</el-tag>
                </div>
              </div>
              <div class="arch-arrow"><el-icon><Bottom /></el-icon></div>
              
              <div class="arch-layer service-layer">
                <div class="layer-title">服务与业务层 (FastAPI & Vue3)</div>
                <div class="layer-items">
                  <el-tag effect="dark">RESTful API 预测服务</el-tag>
                  <el-tag effect="dark">ECharts 态势感知大屏</el-tag>
                  <el-tag effect="dark">高德地图轨迹映射</el-tag>
                </div>
              </div>
            </div>
          </div>
        </div>
      </el-col>

      <el-col :span="13" class="full-height">
        <div class="panel dark-panel flex-col limitation-box">
          <div class="section-box flex-1">
            <div class="box-title text-warning"><el-icon><Warning /></el-icon> 应用局限与应对声明 (Limitations)</div>
            <div class="boundary-content">
              <ul>
                <li><strong>场景边界：</strong> 当前模型（TrafficRec v3.0）的 OSM 路网特征抽取主要基于<strong>北京市</strong>的空间拓扑数据进行训练。由于不同城市的道路网格密度、公交地铁覆盖率存在显著差异，若直接将模型泛化至其他城市（如重庆的山地路网），预测精度可能会出现可预见的衰减。</li>
                <li><strong>应对机制：</strong> 系统架构已解耦特征工程与推理模块。在跨城市部署时，只需替换目标城市的 OSM 底图并进行少量样本的迁移学习（Transfer Learning）即可快速适配。</li>
                <br>
                <li><strong>极端气象的扰动：</strong> 虽已引入气象维度（温度、风速等），但在遇到极端恶劣天气（如特大暴雨导致全城拥堵、地铁停运）时，用户的历史出行习惯会被打破，多模态特征之间的联合分布会发生偏移（Distribution Shift）。</li>
                <li><strong>持续优化方向：</strong> 计划在下一代版本中引入<strong>实时气象动态惩罚权重</strong>，降低模型在此类长尾分布场景下的过拟合风险。</li>
              </ul>
            </div>
          </div>
        </div>
      </el-col>

    </el-row>
  </div>
</template>

<script setup lang="ts">
import { Cpu, Bottom, Warning, DataLine } from '@element-plus/icons-vue'
import { ref, onMounted, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'

// ECharts 实例化
const evolutionChartRef = ref<HTMLElement | null>(null)
let chartInstance: echarts.ECharts | null = null

// 基于 Exp1 - Exp4 提取的真实数据
const expData = {
  categories: ['Exp 1 (基线模型)', 'Exp 2 (特征优化)', 'Exp 3 (地铁强化)', 'Exp 4 (轻量化/消融)'],
  series: [
    { name: 'Subway (地铁)', data: [0.666, 0.736, 0.800, 0.764] },
    { name: 'Bus (公交)', data: [0.761, 0.817, 0.802, 0.787] },
    { name: 'Car & taxi (汽车)', data: [0.514, 0.550, 0.539, 0.452] },
    { name: 'Train (火车)', data: [0.871, 0.842, 0.806, 0.758] },
    { name: 'Bike (骑行)', data: [0.936, 0.932, 0.919, 0.934] },
    { name: 'Walk (步行)', data: [0.916, 0.878, 0.893, 0.824] }
  ]
}

const initChart = () => {
  if (!evolutionChartRef.value) return
  chartInstance = echarts.init(evolutionChartRef.value)

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      backgroundColor: 'rgba(3, 8, 22, 0.9)',
      borderColor: '#00f0ff',
      textStyle: { color: '#fff' }
    },
    legend: {
      data: expData.series.map(s => s.name),
      textStyle: { color: '#a0aabf' },
      top: 0
    },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: expData.categories,
      axisLabel: { color: '#a0aabf', fontSize: 13 },
      axisLine: { lineStyle: { color: '#334155' } }
    },
    yAxis: {
      type: 'value',
      name: 'F1-Score',
      nameTextStyle: { color: '#a0aabf' },
      min: 0.4,
      max: 1.0,
      axisLabel: { color: '#a0aabf' },
      splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)', type: 'dashed' } }
    },
    series: expData.series.map(item => ({
      name: item.name,
      type: 'line',
      data: item.data,
      smooth: true,
      symbol: 'circle',
      symbolSize: 8,
      lineStyle: {
        width: item.name.includes('Subway') || item.name.includes('Bus') ? 4 : 2,
        type: item.name.includes('Walk') || item.name.includes('Bike') ? 'dashed' : 'solid'
      }
    }))
  }

  chartInstance.setOption(option)
}

const handleResize = () => {
  chartInstance?.resize()
}

onMounted(() => {
  initChart()
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance?.dispose()
})
</script>

<style scoped>
/* ========== 全局滚动容器样式 ========== */
.scrollable-container {
  width: 100%;
  height: 100%;
  padding: 24px 32px;
  overflow-y: auto;
  overflow-x: hidden;
  box-sizing: border-box;
}

.scrollable-container::-webkit-scrollbar { width: 8px; }
.scrollable-container::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
.scrollable-container::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.3); border-radius: 4px; }
.scrollable-container::-webkit-scrollbar-thumb:hover { background: rgba(0, 240, 255, 0.6); }

/* ========== 布局辅助 ========== */
.mb-20 { margin-bottom: 20px; }
.content-row { min-height: 400px; }

/* ========== ECharts 图表样式 ========== */
.chart-wrapper { padding: 10px 0; }
.evolution-chart { width: 100%; height: 350px; }

/* ========== 原有样式完整保留 ========== */
.tech-container { height: 100%; padding: 20px; box-sizing: border-box; }
.full-height { height: 100%; }
.panel { background: rgba(16, 25, 43, 0.6); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); backdrop-filter: blur(10px);}
.dark-panel { color: #e5eaf3; }
.flex-col { display: flex; flex-direction: column; }
.flex-1 { flex: 1; }
.panel-header { display: flex; align-items: center; gap: 8px; font-size: 18px; font-weight: bold; color: #fff; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }

/* 系统架构相关样式 */
.section-box { display: flex; flex-direction: column;}
.box-title { font-size: 18px; font-weight: bold; color: #00f0ff; display: flex; align-items: center; gap: 8px; margin-bottom: 20px;}
.arch-content { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: space-around; padding: 20px 0;}
.arch-layer { width: 80%; padding: 20px; border-radius: 8px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; transition: all 0.3s;}
.arch-layer:hover { border-color: rgba(74, 144, 226, 0.5); background: rgba(74, 144, 226, 0.05);}
.layer-title { font-size: 13px; color: #909399; margin-bottom: 12px; font-weight: bold; letter-spacing: 1px;}
.layer-items { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;}
.arch-arrow { color: #4A90E2; font-size: 20px; animation: floatDown 2s infinite;}

@keyframes floatDown {
  0% { transform: translateY(0); opacity: 0.5; }
  50% { transform: translateY(5px); opacity: 1; }
  100% { transform: translateY(0); opacity: 0.5; }
}

/* 边界声明样式 */
.limitation-box { background: rgba(230, 162, 60, 0.05); border-color: rgba(230, 162, 60, 0.2);}
.text-warning { color: #E6A23C; }
.boundary-content ul { margin: 0; padding-left: 20px; color: #a3a6ad; font-size: 13px; line-height: 1.8; }
.boundary-content li { margin-bottom: 10px; }
.boundary-content strong { color: #e5eaf3; }
</style>