<template>
  <div class="datav-wrapper">
    <div class="datav-container">
      
      <header class="dashboard-header">
        <dv-decoration-8 class="header-dec" :color="['#00f0ff', '#00f0ff']" />
        <div class="header-title">
          <dv-decoration-11 class="title-dec" :color="['#00f0ff', '#00f0ff']">
            TrafficRec 多模态交通态势感知系统
          </dv-decoration-11>
        </div>
        <dv-decoration-8 class="header-dec" :reverse="true" :color="['#00f0ff', '#00f0ff']" />
      </header>

      <main class="dashboard-main">
        
        <section class="col-left">
          <dv-border-box-13 class="chart-box">
            <div class="box-title">消融实验 F1-Score 性能演进 (Exp1-Exp4)</div>
            <div ref="lineChartRef" class="echarts-container"></div>
          </dv-border-box-13>
          
          <dv-border-box-12 class="chart-box">
            <div class="box-title">49维多模态特征重要性分析</div>
            <div ref="radarChartRef" class="echarts-container"></div>
          </dv-border-box-12>
        </section>

        <section class="col-center">
          <dv-border-box-8 class="map-border-box" :dur="3">
            <div class="map-wrapper">
              <MapView :trajectories="mapTrajectories" :selectedTrajectory="null" />
            </div>
          </dv-border-box-8>
          
          <div class="center-bottom-stats">
             <dv-border-box-10 class="stat-card">
               <div class="stat-label">轨迹点量级</div>
               <div class="stat-value text-cyan">百万级</div>
             </dv-border-box-10>
             <dv-border-box-10 class="stat-card">
               <div class="stat-label">覆盖路网</div>
               <div class="stat-value text-cyan">北京 OSM</div>
             </dv-border-box-10>
             <dv-border-box-10 class="stat-card">
               <div class="stat-label">气象跨度</div>
               <div class="stat-value text-cyan">5年历史</div>
             </dv-border-box-10>
          </div>
        </section>

        <section class="col-right">
          <dv-border-box-10 class="chart-box scroll-board-box">
            <div class="box-title">实时态势感知推流</div>
            <dv-scroll-board :config="scrollBoardConfig" class="scroll-board" />
          </dv-border-box-10>

          <dv-border-box-12 class="chart-box insights-box">
            <div class="box-title">系统核心洞察</div>
            <div class="insight-content">
              <div class="insight-item">
                <div class="insight-badge">攻克长尾</div>
                <h4 class="insight-title text-glow">公交与私家车混淆难题</h4>
                <p class="insight-desc">Exp2 引入 OSM 空间拓扑后，公交专用道特征使 Bus 类 F1 提升至 0.817。有效解决传统仅依靠轨迹速度带来的误判问题。</p>
              </div>
              <div class="insight-item">
                <div class="insight-badge">地铁识别</div>
                <h4 class="insight-title text-glow">地下微弱信号重建</h4>
                <p class="insight-desc">Exp3 结合时序推演，Subway 类的 F1-score 突破 0.800，弥补了 GPS 信号在地下环境丢失的识别盲区。</p>
              </div>
              <div class="insight-item">
                <div class="insight-badge">绿色普惠</div>
                <h4 class="insight-title text-glow text-green">碳减排量化闭环</h4>
                <p class="insight-desc">基于 Exp4 最终模型，实现微观拥堵溯源与个人绿色出行的高精核算，构筑“识别—核算—激励”闭环。</p>
              </div>
            </div>
          </dv-border-box-12>
        </section>

      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'
import { experimentApi } from '@/api/experiment'
import MapView from '@/components/MapView.vue'

const mapTrajectories = ref([])
const lineChartRef = ref(null)
const radarChartRef = ref(null)
let lineChart = null
let radarChart = null

const scrollBoardConfig = reactive({
  header: ['轨迹ID', '推断方式', '置信度'],
  data: [
    ['TRJ-092A', '公交 (Bus)', "<span style='color:#67C23A;'>99.7%</span>"],
    ['TRJ-114B', '私家车 (Car)', "<span style='color:#E6A23C;'>85.2%</span>"],
    ['TRJ-033C', '地铁 (Subway)', "<span style='color:#00f0ff;'>96.1%</span>"],
    ['TRJ-008D', '骑行 (Bike)', "<span style='color:#67C23A;'>94.5%</span>"],
    ['TRJ-021E', '步行 (Walk)', "<span style='color:#67C23A;'>98.8%</span>"],
    ['TRJ-105F', '公交 (Bus)', "<span style='color:#67C23A;'>91.4%</span>"],
  ],
  index: true,
  columnWidth: [50, 90, 110, 80],
  align: ['center', 'center', 'center', 'center'],
  headerBGC: 'rgba(0, 240, 255, 0.15)',
  rowNum: 6,
  waitTime: 2500
})

// === 数据注入：灌入 Exp1-Exp4 的真实实验数据 ===
const initLineChart = () => {
  if (!lineChartRef.value) return
  lineChart = echarts.init(lineChartRef.value)
  const option = {
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(0,0,0,0.8)', borderColor: '#00f0ff', textStyle: { color: '#fff' } },
    legend: { data: ['Bus', 'Subway', 'Car & Taxi', 'Bike'], textStyle: { color: '#a0aabf' }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '5%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: ['Exp1(基线)', 'Exp2(+OSM)', 'Exp3(+气象)', 'Exp4(优化)'],
      axisLabel: { color: '#a0aabf' }
    },
    yAxis: {
      type: 'value',
      min: 0.4,
      axisLabel: { color: '#a0aabf' },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } }
    },
    series: [
      { name: 'Bus', type: 'line', smooth: true, itemStyle: { color: '#00f0ff' }, lineStyle: { width: 3, shadowBlur: 10 }, data: [0.761, 0.817, 0.802, 0.787] },
      { name: 'Subway', type: 'line', smooth: true, itemStyle: { color: '#b388ff' }, lineStyle: { width: 3 }, data: [0.666, 0.736, 0.800, 0.764] },
      { name: 'Car & Taxi', type: 'line', smooth: true, itemStyle: { color: '#E6A23C' }, lineStyle: { width: 2, type: 'dashed' }, data: [0.514, 0.550, 0.539, 0.452] },
      { name: 'Bike', type: 'line', smooth: true, itemStyle: { color: '#67C23A' }, lineStyle: { width: 2 }, data: [0.936, 0.932, 0.919, 0.934] }
    ]
  }
  lineChart.setOption(option)
}

const initRadarChart = () => {
  if (!radarChartRef.value) return
  radarChart = echarts.init(radarChartRef.value)
  const option = {
    radar: {
      indicator: [
        { name: '运动学', max: 100 },
        { name: '空间拓扑', max: 100 },
        { name: '时序', max: 100 },
        { name: '气象', max: 100 },
        { name: '优化', max: 100 }
      ],
      axisName: { color: '#00f0ff' },
      splitArea: { areaStyle: { color: ['rgba(0, 240, 255, 0.05)', 'rgba(0, 240, 255, 0.1)'] } }
    },
    series: [{
      type: 'radar',
      data: [{ value: [95, 85, 90, 75, 88], areaStyle: { color: 'rgba(0, 240, 255, 0.4)' }, lineStyle: { color: '#00f0ff' } }]
    }]
  }
  radarChart.setOption(option)
}

onMounted(() => {
  setTimeout(() => {
    initLineChart()
    initRadarChart()
  }, 100)
  window.addEventListener('resize', () => {
    lineChart?.resize(); radarChart?.resize();
  })
})
</script>

<style scoped>
.datav-wrapper {
  width: 100%;
  min-height: calc(100vh - 64px); 
  background-color: #030409;
  color: #fff;
  overflow-x: hidden;
}

.datav-container {
  display: flex;
  flex-direction: column;
  padding: 20px;
}

/* 标题栏：一行排版 */
.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 80px;
  flex-shrink: 0;
}
.header-dec { width: 25%; height: 50px; }
.header-title { flex: 1; text-align: center; }
.title-dec { width: 100%; height: 60px; font-size: 26px; font-weight: bold; color: #00f0ff; }

/* 主体：标准流布局 */
.dashboard-main {
  display: flex;
  gap: 20px;
  margin-top: 20px;
  min-height: 900px;
}

.col-left, .col-right { width: 28%; display: flex; flex-direction: column; gap: 20px; }
.col-center { width: 44%; display: flex; flex-direction: column; gap: 20px; }

.chart-box { flex: 1; padding: 20px; background: rgba(10, 16, 29, 0.6); border-radius: 8px; }
.box-title { font-size: 16px; color: #00f0ff; border-left: 4px solid #00f0ff; padding-left: 10px; margin-bottom: 15px; }
.echarts-container { width: 100%; height: calc(100% - 35px); min-height: 250px; }

.map-border-box { height: 600px; width: 100%; }
.map-wrapper { width: 100%; height: 100%; padding: 15px; box-sizing: border-box; }
.map-wrapper :deep(.leaflet-container) { width: 100% !important; height: 100% !important; }

.center-bottom-stats { height: 100px; display: flex; gap: 15px; }
.stat-card { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; background: rgba(0, 240, 255, 0.05); }
.stat-label { font-size: 14px; color: #a0aabf; }
.stat-value { font-size: 22px; font-weight: bold; }

.insight-content { overflow-y: auto; height: calc(100% - 35px); }
.insight-item { margin-bottom: 18px; border-left: 2px solid #00f0ff; padding-left: 12px; }
.insight-badge { font-size: 12px; color: #00f0ff; background: rgba(0, 240, 255, 0.2); padding: 2px 6px; width: fit-content; }
.insight-title { margin: 6px 0; font-size: 15px; }
.insight-desc { font-size: 13px; color: #a0aabf; line-height: 1.5; }

.text-cyan { color: #00f0ff; }
.text-green { color: #67C23A; }
</style>