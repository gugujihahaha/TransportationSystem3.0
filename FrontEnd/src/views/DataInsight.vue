<template>
  <div class="data-insight-container">
    
    <!-- <PageHeader title="城市出行多维数据洞察" subtitle="深度挖掘城市交通脉络，从多时空维度分析出行规律与拥堵成因" /> -->

    <el-skeleton :loading="loading" animated :rows="15" class="skeleton-wrapper">
      <template #default>
        <div class="insight-layout">
          
          <div class="sidebar glass-card">
            <el-menu
              :default-active="activeMenu"
              class="insight-menu"
              @select="handleSelect"
              background-color="transparent"
              text-color="#cbd5e1"
              active-text-color="#38bdf8"
            >
              <el-menu-item index="1">
                <span class="menu-icon"></span> <span>时段出行分析</span>
              </el-menu-item>
              <el-menu-item index="2">
                <span class="menu-icon"></span> <span>道路类型出行分析</span>
              </el-menu-item>
              <el-menu-item index="3">
                <span class="menu-icon"></span> <span>天气影响分析</span>
              </el-menu-item>
              <el-menu-item index="4">
                <span class="menu-icon"></span> <span>拥堵时段分析</span>
              </el-menu-item>
            </el-menu>
          </div>

          <div class="content-area glass-card">
            
            <div v-show="activeMenu === '1'" class="chart-section fade-in">
              <h3 class="section-title">时段出行分析</h3>
              <div class="insight-alert">
                <span class="icon"></span>
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
                  <span class="icon"></span>
                  <span class="text">洞察：主干道机动车占比占绝对主导，支路与步行道是慢行交通的核心承载区。</span>
                </div>
                <div ref="chart2Ref" class="chart-box"></div>
              </div>
            </div>

            <div v-show="activeMenu === '3'" class="chart-section fade-in">
              <h3 class="section-title">天气影响分析</h3>
              <div class="insight-alert">
                <span class="icon"></span>
                <span class="text">洞察：雨天私家车使用比例上升，骑行和步行明显减少。</span>
              </div>
              <div ref="chart3Ref" class="chart-box"></div>
            </div>

            <div v-show="activeMenu === '4'" class="chart-section fade-in">
              <h3 class="section-title">全天拥堵时段演变分析</h3>
              <div class="insight-alert">
                <span class="icon"></span>
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
import PageHeader from '@/components/PageHeader.vue'
const activeMenu = ref('1')
const loading = ref(true)
const insightData = ref(null)

const chart1Ref = ref(null)
const chart2Ref = ref(null)
const chart3Ref = ref(null)
const chart4Ref = ref(null)
const charts = []

const initialized = { '1': false, '2': false, '3': false, '4': false }

const modeColors = {
  'Walk': '#8dd1e1',      
  'Bike': '#6794c7',      
  'Bus': '#a5d6a7',       
  'Car & taxi': '#ef5350',
  'Subway': '#9575cd',    
  'Train': '#f48fb1'      
}

const hasRoadData = computed(() => {
  return insightData.value?.road_type?.categories?.length > 0
})

// 图表1：时段出行
const initChart1 = () => {
  const chart = echarts.init(chart1Ref.value)
  const data = insightData.value.time_period
  
  chart.setOption({
    tooltip: { 
      trigger: 'axis', 
      backgroundColor: 'rgba(15,23,42,0.95)', 
      textStyle: { color: '#fff' },
      borderRadius: 8, padding: 12
    },
    legend: { data: data.modes, textStyle: { color: '#fff', fontSize:13 }, bottom:0 },
    grid: { top: 30, left: '3%', right: '4%', bottom: '12%', containLabel: true },
    xAxis: { 
      type: 'category', data: data.periods, 
      axisLabel: { color: '#fff', fontSize:12 }
    },
    yAxis: { 
      type: 'value', 
      axisLabel: { color: '#fff', fontSize:12 }, 
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
    },
    series: data.modes.map(mode => ({
      name: mode, type: 'bar', stack: 'total', barWidth: '45%',
      itemStyle: { color: modeColors[mode], borderRadius: [4,4,0,0] },
      data: data.data[mode]
    }))
  })
  charts[0] = chart
}

// 图表2：道路类型
const initChart2 = () => {
  if (!hasRoadData.value) return
  const chart = echarts.init(chart2Ref.value)
  const data = insightData.value.road_type
  
  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: '#0f172a', textStyle: { color: '#fff' }, borderRadius:8 },
    legend: { data: data.modes, textStyle: { color: '#fff', fontSize:13 }, bottom:0 },
    grid: { top:30, left:'3%', right:'4%', bottom:'12%' },
    xAxis: { type: 'category', data: data.categories, axisLabel: { color: '#fff' } },
    yAxis: { type: 'value', axisLabel: { color: '#fff' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } },
    series: data.modes.map(mode => ({
      name: mode, type: 'bar', stack: 'total', barWidth: '45%',
      itemStyle: { color: modeColors[mode], borderRadius: [4,4,0,0] },
      data: data.data[mode]
    }))
  })
  charts[1] = chart
}

// 图表3：天气
const initChart3 = () => {
  const chart = echarts.init(chart3Ref.value)
  const data = insightData.value.weather_impact
  const modes = Object.keys(data.sunny)
  
  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: '#0f172a', textStyle: { color: '#fff' }, borderRadius:8 },
    legend: { data: ['晴天','雨天'], textStyle: { color: '#fff' }, bottom:0 },
    grid: { top:30, left:'3%', right:'4%', bottom:'12%' },
    xAxis: { type: 'category', data: modes, axisLabel: { color: '#fff' } },
    yAxis: { type: 'value', axisLabel: { color: '#fff' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } },
    series: [
      { name: '晴天', type: 'bar', barWidth: '30%', itemStyle: { color: '#ffd580', borderRadius:[4,4,0,0] }, data: modes.map(m=>data.sunny[m]) },
      { name: '雨天', type: 'bar', barWidth: '30%', itemStyle: { color: '#8cb6d8', borderRadius:[4,4,0,0] }, data: modes.map(m=>data.rainy[m]) }
    ]
  })
  charts[2] = chart
}

// 图表4：拥堵时段
const initChart4 = () => {
  const chart = echarts.init(chart4Ref.value)
  const data = insightData.value.congestion_timing
  
  chart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: '#0f172a', textStyle: { color: '#fff' }, borderRadius:8 },
    grid: { top:30, left:'3%', right:'4%', bottom:'10%' },
    xAxis: { type: 'category', boundaryGap: false, data: data.hours.map(h=>`${h}时`), axisLabel: { color: '#fff' } },
    yAxis: { type: 'value', axisLabel: { color: '#fff' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } },
    series: [{
      name: '小汽车占比', type: 'line', smooth: true,
      symbol: 'circle', symbolSize: 6,
      itemStyle: { color: '#ef5350' },
      lineStyle: { width: 3 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0,0,0,1, [
          { offset:0, color: 'rgba(239,83,80,0.4)' },
          { offset:1, color: 'rgba(239,83,80,0.05)' }
        ])
      },
      data: data.car_ratio,
      markArea: {
        itemStyle: { color: 'rgba(255,213,128,0.2)' },
        label: { color: '#ffd580', fontWeight:'bold' },
        data: [
          [{ name: '早高峰', xAxis: '7时' }, { xAxis: '9时' }],
          [{ name: '晚高峰', xAxis: '17时' }, { xAxis: '19时' }]
        ]
      }
    }]
  })
  charts[3] = chart
}

const renderActiveChart = async (tabIndex) => {
  await nextTick()
  if (tabIndex === '1' && !initialized['1']) { initChart1(); initialized['1'] = true }
  if (tabIndex === '2' && !initialized['2']) { initChart2(); initialized['2'] = true }
  if (tabIndex === '3' && !initialized['3']) { initChart3(); initialized['3'] = true }
  if (tabIndex === '4' && !initialized['4']) { initChart4(); initialized['4'] = true }
  charts.forEach(c=>c&&c.resize())
}

const handleSelect = (index) => { activeMenu.value = index }
watch(activeMenu, renderActiveChart)

const handleResize = () => { charts.forEach(c=>c&&c.resize()) }

onMounted(async () => {
  try {
    const res = await fetch('/insight_data.json')
    if (res.ok) insightData.value = await res.json()
  } catch (err) {
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
  charts.forEach(c => c && c.dispose())
})
</script>

<style scoped>
.data-insight-container {
  min-height: calc(100vh - 64px);
  background: linear-gradient(135deg, #111827, #1e293b);
  padding: 24px 36px;
  color: #ffffff;
  font-family: 'Inter', 'Microsoft YaHei', sans-serif;
}



.insight-layout {
  display: flex;
  gap: 28px;
  align-items: flex-start;
}

.sidebar {
  width: 270px;
  flex-shrink: 0;
}

.glass-card {
  background: rgba(30, 41, 59, 0.8);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 18px;
  padding: 28px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

:deep(.el-menu-item) {
  height: 54px;
  line-height: 54px;
  margin: 6px 10px;
  border-radius: 12px;
  font-size: 15px;
  font-weight: 500;
}
:deep(.el-menu-item.is-active) {
  background: rgba(14, 165, 233, 0.25) !important;
  color: #ffffff !important;
  font-weight: 600;
}

.content-area {
  flex: 1;
  min-height: 580px;
}

.section-title {
  font-size: 20px;
  color: #ffffff;
  font-weight: 600;
  margin: 0 0 20px 0;
}
.section-title::before {
  content: '';
  display: inline-block;
  width: 5px;
  height: 20px;
  background: #38bdf8;
  margin-right: 10px;
  border-radius: 2px;
}

.insight-alert {
  background: rgba(14, 165, 233, 0.15);
  border-left: 4px solid #38bdf8;
  padding: 14px 18px;
  border-radius: 8px;
  margin-bottom: 24px;
  color: #ffffff;
}
.insight-alert .text {
  font-size: 15px;
  color: #ffffff;
}

.chart-box {
  width: 100%;
  height: 440px;
  background: rgba(15, 23, 42, 0.6);
  border-radius: 12px;
  padding: 12px;
}

.fade-in {
  animation: fadeIn 0.5s ease forwards;
}
@keyframes fadeIn {
  from { opacity:0; transform: translateY(10px); }
  to { opacity:1; transform: translateY(0); }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 400px;
  color: #9ca3af;
}
.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

@media (max-width: 1024px) {
  .insight-layout {
    flex-direction: column;
  }
  .sidebar {
    width: 100%;
  }
}
</style>