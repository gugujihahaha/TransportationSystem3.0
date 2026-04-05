<template>
  <div class="data-insight-container">
    <div class="page-header">
      <h2 class="title">城市出行多维数据洞察</h2>
      <p class="subtitle">深度挖掘城市交通脉络，从多时空维度分析出行规律与拥堵成因</p>
    </div>

    <el-skeleton :loading="loading" animated :rows="15">
      <template #default>
        <div class="insight-layout">
          
          <div class="sidebar glass-card">
            <el-menu
              :default-active="activeMenu"
              class="insight-menu"
              @select="handleSelect"
              background-color="transparent"
              text-color="#94a3b8"
              active-text-color="#38bdf8"
            >
              <el-menu-item index="1">
                <span class="menu-icon">🕒</span> <span>时段出行分析</span>
              </el-menu-item>
              <el-menu-item index="2">
                <span class="menu-icon">🛣️</span> <span>道路类型出行分析</span>
              </el-menu-item>
              <el-menu-item index="3">
                <span class="menu-icon">⛈️</span> <span>天气影响分析</span>
              </el-menu-item>
              <el-menu-item index="4">
                <span class="menu-icon">🚗</span> <span>拥堵时段分析</span>
              </el-menu-item>
            </el-menu>
          </div>

          <div class="content-area glass-card">
            
            <div v-show="activeMenu === '1'" class="chart-section fade-in">
              <h3 class="section-title">时段出行分析</h3>
              <div class="insight-alert">
                <span class="icon">💡</span>
                <span class="text">洞察：早高峰小汽车占比最高，夜间步行和骑行比例明显下降。</span>
              </div>
              <div ref="chart1Ref" class="chart-box"></div>
            </div>

            <div v-show="activeMenu === '2'" class="chart-section fade-in">
              <h3 class="section-title">道路类型出行分析</h3>
              <div v-if="!hasRoadData" class="empty-state">
                <div class="empty-icon">🚧</div>
                <p>道路类型数据正在准备中...</p>
              </div>
              <div v-else>
                <div class="insight-alert">
                  <span class="icon">💡</span>
                  <span class="text">洞察：主干道机动车占比占绝对主导，支路与步行道是慢行交通的核心承载区。</span>
                </div>
                <div ref="chart2Ref" class="chart-box"></div>
              </div>
            </div>

            <div v-show="activeMenu === '3'" class="chart-section fade-in">
              <h3 class="section-title">天气影响分析</h3>
              <div class="insight-alert">
                <span class="icon">💡</span>
                <span class="text">洞察：雨天私家车使用比例上升，骑行和步行明显减少。</span>
              </div>
              <div ref="chart3Ref" class="chart-box"></div>
            </div>

            <div v-show="activeMenu === '4'" class="chart-section fade-in">
              <h3 class="section-title">全天拥堵时段演变分析</h3>
              <div class="insight-alert">
                <span class="icon">💡</span>
                <span class="text">洞察：早晚高峰小汽车占比超过40%，与全市拥堵高发时段高度吻合。</span>
              </div>
              <div ref="chart4Ref" class="chart-box"></div>
            </div>

          </div>
        </div>
      </template>
    </el-skeleton>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import * as echarts from 'echarts'

const activeMenu = ref('1')
const loading = ref(true)
const insightData = ref(null)

// ECharts 实例及引用
const chart1Ref = ref(null)
const chart2Ref = ref(null)
const chart3Ref = ref(null)
const chart4Ref = ref(null)
const charts = []

// 懒加载初始化标记（防止 v-show 导致的图表尺寸塌陷）
const initialized = { '1': false, '2': false, '3': false, '4': false }

// 统一配色规范
const modeColors = {
  'Walk': '#00FF88',
  'Bike': '#00FFFF',
  'Bus': '#FFDD00',
  'Car & taxi': '#FF0033',
  'Subway': '#CC33FF',
  'Train': '#FF6600'
}

// 检查道路数据是否为空
const hasRoadData = computed(() => {
  return insightData.value?.road_type?.categories?.length > 0
})

// 图表 1：时段堆叠柱状图
const initChart1 = () => {
  const chart = echarts.init(chart1Ref.value)
  const data = insightData.value.time_period
  
  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff' } },
    legend: { data: data.modes, textStyle: { color: '#cbd5e1' }, bottom: 0 },
    grid: { top: 30, left: '3%', right: '4%', bottom: '12%', containLabel: true },
    xAxis: { type: 'category', name: '时段', nameTextStyle: { color: '#94a3b8' }, data: data.periods, axisLabel: { color: '#cbd5e1' } },
    yAxis: { type: 'value', name: '占比 (%)', nameTextStyle: { color: '#94a3b8' }, axisLabel: { color: '#cbd5e1' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } } },
    series: data.modes.map(mode => ({
      name: mode,
      type: 'bar',
      stack: 'total',
      barWidth: '45%',
      itemStyle: { color: modeColors[mode] },
      data: data.data[mode]
    }))
  })
  charts[0] = chart
}

// 图表 2：道路类型堆叠柱状图
const initChart2 = () => {
  if (!hasRoadData.value) return
  const chart = echarts.init(chart2Ref.value)
  const data = insightData.value.road_type
  
  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff' } },
    legend: { data: data.modes, textStyle: { color: '#cbd5e1' }, bottom: 0 },
    grid: { top: 30, left: '3%', right: '4%', bottom: '12%', containLabel: true },
    xAxis: { type: 'category', data: data.categories, axisLabel: { color: '#cbd5e1' } },
    yAxis: { type: 'value', name: '占比 (%)', nameTextStyle: { color: '#94a3b8' }, axisLabel: { color: '#cbd5e1' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } } },
    series: data.modes.map(mode => ({
      name: mode,
      type: 'bar',
      stack: 'total',
      barWidth: '45%',
      itemStyle: { color: modeColors[mode] },
      data: data.data[mode]
    }))
  })
  charts[1] = chart
}

// 图表 3：天气影响对比分组柱状图
const initChart3 = () => {
  const chart = echarts.init(chart3Ref.value)
  const data = insightData.value.weather_impact
  const modes = Object.keys(data.sunny)
  
  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff' } },
    legend: { data: ['晴天', '雨天'], textStyle: { color: '#cbd5e1' }, bottom: 0 },
    grid: { top: 30, left: '3%', right: '4%', bottom: '12%', containLabel: true },
    xAxis: { type: 'category', data: modes, axisLabel: { color: '#cbd5e1' } },
    yAxis: { type: 'value', name: '占比 (%)', nameTextStyle: { color: '#94a3b8' }, axisLabel: { color: '#cbd5e1' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } } },
    series: [
      {
        name: '晴天',
        type: 'bar',
        barWidth: '30%',
        itemStyle: { color: '#FFDD00', borderRadius: [4, 4, 0, 0] },
        data: modes.map(m => data.sunny[m])
      },
      {
        name: '雨天',
        type: 'bar',
        barWidth: '30%',
        itemStyle: { color: '#00A8FF', borderRadius: [4, 4, 0, 0] },
        data: modes.map(m => data.rainy[m])
      }
    ]
  })
  charts[2] = chart
}

// 图表 4：拥堵时段演变折线图
const initChart4 = () => {
  const chart = echarts.init(chart4Ref.value)
  const data = insightData.value.congestion_timing
  
  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff' } },
    grid: { top: 30, left: '3%', right: '4%', bottom: '8%', containLabel: true },
    xAxis: { 
      type: 'category', 
      boundaryGap: false, 
      data: data.hours.map(h => `${h}时`), 
      axisLabel: { color: '#cbd5e1' } 
    },
    yAxis: { 
      type: 'value', 
      name: '小汽车占比 (%)', 
      nameTextStyle: { color: '#94a3b8' },
      axisLabel: { color: '#cbd5e1' },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } } 
    },
    series: [{
      name: '小汽车占比',
      type: 'line',
      smooth: true,
      symbol: 'none',
      itemStyle: { color: '#FF0033' },
      lineStyle: { width: 3 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(255, 0, 51, 0.6)' },
          { offset: 1, color: 'rgba(255, 0, 51, 0.05)' }
        ])
      },
      data: data.car_ratio,
      // 早晚高峰阴影高亮
      markArea: {
        itemStyle: { color: 'rgba(255, 221, 0, 0.1)' },
        label: { color: '#FFDD00', fontWeight: 'bold' },
        data: [
          [ { name: '早高峰', xAxis: '7时' }, { xAxis: '9时' } ],
          [ { name: '晚高峰', xAxis: '17时' }, { xAxis: '19时' } ]
        ]
      }
    }]
  })
  charts[3] = chart
}

// 动态渲染图表（结合 v-show 确保不缩放塌陷）
const renderActiveChart = async (tabIndex) => {
  await nextTick()
  if (tabIndex === '1' && !initialized['1'] && chart1Ref.value) { initChart1(); initialized['1'] = true }
  if (tabIndex === '2' && !initialized['2'] && chart2Ref.value) { initChart2(); initialized['2'] = true }
  if (tabIndex === '3' && !initialized['3'] && chart3Ref.value) { initChart3(); initialized['3'] = true }
  if (tabIndex === '4' && !initialized['4'] && chart4Ref.value) { initChart4(); initialized['4'] = true }
  
  // 切换后自动自适应一次
  charts.forEach(chart => chart && chart.resize())
}

const handleSelect = (index) => {
  activeMenu.value = index
}

watch(activeMenu, (newVal) => {
  renderActiveChart(newVal)
})

const handleResize = () => {
  charts.forEach(chart => chart && chart.resize())
}

onMounted(async () => {
  try {
    const res = await fetch('/insight_data.json')
    if (res.ok) {
      insightData.value = await res.json()
    } else {
      throw new Error('JSON fetch not ok')
    }
  } catch (err) {
    console.warn('读取本地数据失败，使用后备 Mock 数据', err)
    // 防止开发时缺少文件导致白屏，提供合理的后备数据
    insightData.value = {
      "time_period": { "periods": ["早高峰 (7-9点)", "午间 (11-13点)", "晚高峰 (17-19点)", "夜间 (22-5点)", "其他时段"], "modes": ["Walk", "Bike", "Bus", "Car & taxi", "Subway", "Train"], "data": { "Walk": [8.2, 12.1, 8.3, 5.2, 10.5], "Bike": [6.1, 10.3, 6.2, 2.1, 9.4], "Bus": [25.1, 20.2, 28.5, 10.5, 22.1], "Car & taxi": [42.5, 25.1, 40.2, 15.2, 28.5], "Subway": [15.1, 25.3, 10.5, 50.1, 20.5], "Train": [3.0, 7.0, 6.3, 16.9, 9.0] } },
      "road_type": { "categories": ["主干道", "次干道", "支路", "步行道"], "modes": ["Walk", "Bike", "Bus", "Car & taxi", "Subway", "Train"], "data": { "Walk": [2.2, 15.1, 35.3, 85.2], "Bike": [4.1, 20.3, 40.2, 12.1], "Bus": [45.1, 30.2, 5.5, 0.5], "Car & taxi": [42.5, 30.1, 15.2, 1.5], "Subway": [5.1, 3.3, 2.5, 0.1], "Train": [1.0, 1.0, 1.3, 0.6] } },
      "weather_impact": { "sunny": { "Walk": 15.5, "Bike": 12.2, "Bus": 25.5, "Car & taxi": 30.5, "Subway": 12.5, "Train": 3.8 }, "rainy": { "Walk": 5.3, "Bike": 3.1, "Bus": 35.5, "Car & taxi": 45.2, "Subway": 8.5, "Train": 2.4 } },
      "congestion_timing": { "hours": Array.from({length:24}, (_,i)=>i), "car_ratio": [10,8,7,6,5,8,15,42,45,30,25,22,24,26,25,28,35,46,48,35,25,18,15,12] }
    }
  } finally {
    loading.value = false
    renderActiveChart(activeMenu.value)
  }
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  charts.forEach(chart => chart && chart.dispose())
})
</script>

<style scoped>
/* 全局页面容器 */
.data-insight-container {
  min-height: calc(100vh - 64px);
  background-color: transparent; /* 改为透明，完美融入系统背景 */
  padding: 20px 30px;
  box-sizing: border-box;
  color: #e2e8f0;
}

/* 顶部标题区 */
.page-header {
  margin-bottom: 24px;
}
.page-header .title {
  font-size: 24px;
  font-weight: bold;
  color: #38bdf8;
  margin: 0 0 8px 0;
  letter-spacing: 1px;
}
.page-header .subtitle {
  font-size: 14px;
  color: #94a3b8;
  margin: 0;
}

/* 主体布局 */
.insight-layout {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

/* 侧边栏菜单 */
.sidebar {
  width: 260px;
  flex-shrink: 0;
  padding: 10px 0 !important;
}
.menu-icon {
  margin-right: 12px;
  font-size: 16px;
}
:deep(.el-menu) {
  border-right: none;
}
:deep(.el-menu-item) {
  height: 50px;
  line-height: 50px;
  margin: 8px 16px;
  border-radius: 8px;
  transition: all 0.3s;
  font-size: 15px;
  font-weight: 500;
}
:deep(.el-menu-item.is-active) {
  background: rgba(56, 189, 248, 0.15) !important;
  font-weight: bold;
}
:deep(.el-menu-item:hover) {
  background: rgba(255, 255, 255, 0.05) !important;
}

/* 右侧内容区 */
.content-area {
  flex: 1;
  min-width: 0; /* 防止 ECharts 突破 flex 容器 */
  min-height: 540px;
  display: flex;
  flex-direction: column;
}

/* 玻璃拟态卡片复用 */
.glass-card {
  background: rgba(15, 25, 45, 0.6);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(56, 189, 248, 0.25);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* 单个业务区块 */
.chart-section {
  display: flex;
  flex-direction: column;
  flex: 1;
}
.section-title {
  font-size: 18px;
  color: #fff;
  margin: 0 0 16px 0;
  display: flex;
  align-items: center;
}
.section-title::before {
  content: '';
  display: inline-block;
  width: 4px;
  height: 18px;
  background-color: #38bdf8;
  margin-right: 10px;
  border-radius: 2px;
}

/* 洞察警告框 */
.insight-alert {
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(56, 189, 248, 0.02));
  border-left: 3px solid #38bdf8;
  padding: 12px 16px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}
.insight-alert .icon {
  margin-right: 10px;
  font-size: 18px;
}
.insight-alert .text {
  font-size: 14px;
  color: #cbd5e1;
  letter-spacing: 0.5px;
}

/* ECharts 挂载盒 */
.chart-box {
  width: 100%;
  height: 420px; /* 强制高度防止 v-show 引起塌陷 */
  flex: 1;
}

/* 空状态处理 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 380px;
  color: #94a3b8;
}
.empty-icon {
  font-size: 40px;
  margin-bottom: 16px;
  opacity: 0.6;
}

/* 简单的淡入动画 */
.fade-in {
  animation: fadeIn 0.4s ease forwards;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 1024px) {
  .insight-layout {
    flex-direction: column;
  }
  .sidebar {
    width: 100%;
  }
  :deep(.el-menu) {
    display: flex;
    overflow-x: auto;
  }
  :deep(.el-menu-item) {
    margin: 0 8px;
  }
}
</style>