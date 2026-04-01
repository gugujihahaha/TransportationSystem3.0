<template>
  <div class="datav-wrapper">
    <dv-full-screen-container class="datav-container">
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
            <div class="box-title">消融实验 F1-Score 演进 (Exp1-Exp4)</div>
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
              <MapView 
                :trajectories="mapTrajectories" 
                :selectedTrajectory="null" 
              />
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
                <p class="insight-desc">基于 Exp4 最终模型，实现微观拥堵溯源与个人绿色出行的高精核算，构筑“行为识别—智能核算—普惠激励”的全链条。</p>
              </div>
            </div>
          </dv-border-box-12>
        </section>

      </main>
    </dv-full-screen-container>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'

// 引入你修复好的无错版地图组件
import MapView from '@/components/MapView.vue'

// 修复点：定义一个响应式空数组传给地图，确保 MapView 不会因为 undefined 崩溃
const mapTrajectories = ref([])

// 图表实例引用
const lineChartRef = ref(null)
const radarChartRef = ref(null)
let lineChart = null
let radarChart = null

// DataV 轮播表配置 (模拟实时识别)
const scrollBoardConfig = reactive({
  header: ['轨迹ID', '推断方式', '置信度'],
  data: [
    ['TRJ-092A', '公交 (Bus)', "<span style='color:#67C23A;text-shadow: 0 0 5px #67C23A;'>99.7%</span>"],
    ['TRJ-114B', '私家车 (Car)', "<span style='color:#E6A23C;'>85.2%</span>"],
    ['TRJ-033C', '地铁 (Subway)', "<span style='color:#00f0ff;text-shadow: 0 0 5px #00f0ff;'>96.1%</span>"],
    ['TRJ-008D', '骑行 (Bike)', "<span style='color:#67C23A;'>94.5%</span>"],
    ['TRJ-021E', '步行 (Walk)', "<span style='color:#67C23A;'>98.8%</span>"],
    ['TRJ-105F', '公交 (Bus)', "<span style='color:#67C23A;text-shadow: 0 0 5px #67C23A;'>91.4%</span>"],
  ],
  index: true,
  columnWidth: [50, 90, 110, 80],
  align: ['center', 'center', 'center', 'center'],
  oddRowBGC: 'rgba(0, 240, 255, 0.05)',
  evenRowBGC: 'rgba(0, 0, 0, 0)',
  headerBGC: 'rgba(0, 240, 255, 0.15)',
  rowNum: 6,
  waitTime: 2500
})

// 初始化折线图 (F1-score 演进)
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
      data: ['Exp1', 'Exp2', 'Exp3', 'Exp4'],
      axisLabel: { color: '#a0aabf' },
      axisLine: { lineStyle: { color: '#334155' } }
    },
    yAxis: {
      type: 'value',
      min: 0.4,
      max: 1.0,
      axisLabel: { color: '#a0aabf' },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } }
    },
    series: [
      { name: 'Bus', type: 'line', smooth: true, itemStyle: { color: '#00f0ff' }, lineStyle: { width: 3, shadowColor: '#00f0ff', shadowBlur: 10 }, data: [0.761, 0.817, 0.802, 0.787] },
      { name: 'Subway', type: 'line', smooth: true, itemStyle: { color: '#b388ff' }, lineStyle: { width: 3, shadowColor: '#b388ff', shadowBlur: 10 }, data: [0.666, 0.736, 0.800, 0.764] },
      { name: 'Car & Taxi', type: 'line', smooth: true, itemStyle: { color: '#E6A23C' }, lineStyle: { width: 2, type: 'dashed' }, data: [0.514, 0.550, 0.539, 0.452] },
      { name: 'Bike', type: 'line', smooth: true, itemStyle: { color: '#67C23A' }, lineStyle: { width: 2 }, data: [0.936, 0.932, 0.919, 0.934] }
    ]
  }
  lineChart.setOption(option)
}

// 初始化雷达图 (多模态特征)
const initRadarChart = () => {
  if (!radarChartRef.value) return
  radarChart = echarts.init(radarChartRef.value)
  const option = {
    tooltip: {},
    radar: {
      indicator: [
        { name: '运动学特征', max: 100 },
        { name: '空间拓扑', max: 100 },
        { name: '时序依赖', max: 100 },
        { name: '气象环境', max: 100 },
        { name: '长尾优化', max: 100 }
      ],
      shape: 'polygon',
      axisName: { color: '#00f0ff', fontSize: 12 },
      splitArea: { areaStyle: { color: ['rgba(0, 240, 255, 0.05)', 'rgba(0, 240, 255, 0.1)', 'rgba(0, 240, 255, 0.15)', 'rgba(0, 240, 255, 0.2)'] } },
      axisLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.3)' } },
      splitLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.5)' } }
    },
    series: [{
      name: '特征重要性', type: 'radar',
      data: [{ value: [95, 85, 90, 75, 88], name: 'TrafficRec', areaStyle: { color: 'rgba(0, 240, 255, 0.4)' }, lineStyle: { color: '#00f0ff', width: 2 }, itemStyle: { color: '#00f0ff', borderColor: '#fff', borderWidth: 2, shadowColor: '#00f0ff', shadowBlur: 10 } }]
    }]
  }
  radarChart.setOption(option)
}

// 窗口尺寸自适应
const handleResize = () => {
  if (lineChart) lineChart.resize()
  if (radarChart) radarChart.resize()
}

onMounted(() => {
  // 增加小延迟以确保 DataV 的容器已经完全渲染，避免 echarts 找不到高度
  setTimeout(() => {
    initLineChart()
    initRadarChart()
  }, 100)
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  if (lineChart) lineChart.dispose()
  if (radarChart) radarChart.dispose()
})
</script>

<style scoped>
/* 全局暗黑赛博朋克底色 */
.datav-wrapper {
  width: 100vw;
  height: 100vh;
  background-color: #030409;
  background-image: radial-gradient(circle at 50% 50%, #0a101d 0%, #030409 100%);
  color: #fff;
  overflow: hidden;
}

.datav-container {
  display: flex;
  flex-direction: column;
  padding: 15px 20px;
  box-sizing: border-box;
}

/* 头部样式 */
.dashboard-header {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80px;
  margin-bottom: 20px;
}
.header-dec {
  width: 25%;
  height: 50px;
}
.header-title {
  width: 40%;
  text-align: center;
  font-size: 26px;
  font-weight: bold;
  letter-spacing: 2px;
}
.title-dec {
  width: 100%;
  height: 60px;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.8);
}

/* 主体三栏布局 */
.dashboard-main {
  display: flex;
  flex: 1;
  gap: 20px;
  height: calc(100% - 100px);
}
.col-left, .col-right {
  width: 28%;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.col-center {
  width: 44%;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 通用容器样式 */
.chart-box {
  flex: 1;
  position: relative;
  padding: 20px;
  background: rgba(10, 16, 29, 0.6);
  border-radius: 8px;
}
.box-title {
  font-size: 16px;
  font-weight: bold;
  color: #00f0ff;
  text-shadow: 0 0 5px rgba(0, 240, 255, 0.5);
  margin-bottom: 15px;
  padding-left: 10px;
  border-left: 4px solid #00f0ff;
}
.echarts-container {
  width: 100%;
  height: calc(100% - 35px);
}

/* 中间地图区域 */
.map-border-box {
  flex: 1;
  width: 100%;
}
.map-wrapper {
  width: 100%;
  height: 100%;
  padding: 15px; /* 避开 DataV 边框的装饰线 */
  box-sizing: border-box;
}
/* 强制让你的 MapView 继承父级宽高 */
.map-wrapper :deep(.leaflet-container) {
  width: 100% !important;
  height: 100% !important;
  border-radius: 8px;
  border: 1px solid rgba(0, 240, 255, 0.2); 
}

/* 中下部数据卡片 */
.center-bottom-stats {
  height: 100px;
  display: flex;
  justify-content: space-between;
  gap: 15px;
}
.stat-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background: rgba(0, 240, 255, 0.05);
}
.stat-label {
  font-size: 14px;
  color: #a0aabf;
  margin-bottom: 8px;
}
.stat-value {
  font-size: 22px;
  font-weight: bold;
  font-family: 'Courier New', Courier, monospace;
}

/* 右侧：推流与洞察 */
.scroll-board-box {
  flex: 1.2;
}
.scroll-board {
  width: 100%;
  height: calc(100% - 35px);
}
.insights-box {
  flex: 0.8;
}
.insight-content {
  overflow-y: auto;
  height: calc(100% - 35px);
  padding-right: 5px;
}
.insight-item {
  margin-bottom: 18px;
  background: linear-gradient(90deg, rgba(0, 240, 255, 0.1) 0%, transparent 100%);
  padding: 12px;
  border-left: 2px solid #00f0ff;
}
.insight-badge {
  display: inline-block;
  padding: 2px 8px;
  background: rgba(0, 240, 255, 0.2);
  color: #00f0ff;
  border-radius: 2px;
  font-size: 12px;
  margin-bottom: 6px;
}
.insight-title {
  margin: 0 0 6px 0;
  font-size: 15px;
  color: #fff;
}
.insight-desc {
  margin: 0;
  font-size: 13px;
  color: #a0aabf;
  line-height: 1.5;
}

/* 发光辅助类 */
.text-glow { text-shadow: 0 0 8px rgba(255, 255, 255, 0.6); }
.text-cyan { color: #00f0ff; text-shadow: 0 0 8px #00f0ff; }
.text-green { color: #67C23A; text-shadow: 0 0 8px #67C23A; }

/* 隐藏滚动条 */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.3); border-radius: 4px; }
</style>