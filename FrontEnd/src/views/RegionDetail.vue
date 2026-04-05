<template>
  <div class="region-detail-container">
    <div class="breadcrumb-header">
      <router-link to="/" class="back-link">
        <span class="back-arrow">←</span> 返回首页
      </router-link>
      <span class="separator">/</span>
      <span class="current-page">区域详情</span>
      <span class="separator">/</span>
      <span class="region-name">{{ regionName }}</span>
    </div>

    <div v-if="loading" class="status-container">
      <div class="loading-spinner"></div>
      <p>数据加载中...</p>
    </div>
    <div v-else-if="!regionData" class="status-container">
      <div class="empty-icon">📂</div>
      <p>该区域（{{ regionName }}）数据整理中，敬请期待...</p>
    </div>

    <div v-else class="detail-content fade-up">
      <div class="dashboard-row row-1">
        <div class="glass-card metrics-card">
          <div class="card-title">核心交通指标</div>
          <div class="metrics-grid">
            <div class="metric-item">
              <div class="metric-label">交通拥堵指数</div>
              <div class="metric-value text-red">{{ regionData.index.toFixed(2) }}</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">平均车速 (km/h)</div>
              <div class="metric-value text-green">{{ regionData.speed.toFixed(1) }}</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">全市拥堵排名</div>
              <div class="metric-value text-orange">TOP {{ regionData.rank }}</div>
            </div>
          </div>
        </div>

        <div class="glass-card chart-card">
          <div class="card-title">出行结构识别占比</div>
          <div ref="modeChartRef" class="chart-container"></div>
        </div>
      </div>

      <div class="dashboard-row row-2">
        <div class="glass-card table-card">
          <div class="card-title">主要拥堵路段排行</div>
          <el-table 
            :data="regionData.topRoads" 
            style="width: 100%" 
            class="cyber-table"
            :row-style="{ background: 'transparent' }"
            :cell-style="{ borderBottom: '1px solid rgba(255,255,255,0.08)' }"
            :header-cell-style="{ background: 'rgba(56,189,248,0.1)', color: '#e2e8f0', fontWeight: 'bold', borderBottom: '1px solid rgba(56,189,248,0.3)' }"
          >
            <el-table-column type="index" label="排名" width="80" align="center">
              <template #default="scope">
                <span :class="['rank-badge', `rank-${scope.$index + 1}`]">{{ scope.$index + 1 }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="name" label="路段名称" />
            <el-table-column prop="hours" label="年均拥堵时长(h)" align="right">
              <template #default="scope">
                <span class="text-highlight">{{ scope.row.hours }}</span>
              </template>
            </el-table-column>
          </el-table>
        </div>

        <div class="glass-card chart-card">
          <div class="card-title">近7天拥堵指数趋势</div>
          <div ref="trendChartRef" class="chart-container"></div>
        </div>
      </div>

      <div class="dashboard-row row-3">
        <div class="glass-card suggestion-card">
          <div class="card-title">AI 精细化治理建议</div>
          <div class="suggestion-content">
            <span class="quote-mark">“</span>
            {{ regionData.suggestion }}
            <span class="quote-mark">”</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import * as echarts from 'echarts'

const route = useRoute()
// 从路由参数中获取区域名称，默认为"东城区"防止报错
const regionName = ref(route.params.name || route.query.name || '东城区')

const loading = ref(true)
const regionData = ref(null)

const modeChartRef = ref(null)
const trendChartRef = ref(null)
const charts = []

// 获取数据
const loadData = async () => {
  try {
    const res = await fetch('/region_data.json')
    if (res.ok) {
      const allData = await res.json()
      regionData.value = allData[regionName.value] || null
    }
  } catch (err) {
    console.error('加载区域详情数据失败:', err)
  } finally {
    loading.value = false
    if (regionData.value) {
      await nextTick()
      initCharts()
    }
  }
}

// 初始化图表
const initCharts = () => {
  if (!regionData.value) return
  const data = regionData.value

  // --- 1. 出行结构环形图 ---
  if (modeChartRef.value) {
    const modeChart = echarts.init(modeChartRef.value)
    const modeColors = { '步行': '#00FF88', '骑行': '#00FFFF', '公交': '#FFDD00', '地铁': '#CC33FF', '火车': '#FF6600', '小汽车': '#FF0033' }
    
    modeChart.setOption({
      tooltip: { trigger: 'item', backgroundColor: 'rgba(10,14,23,0.9)', textStyle: { color: '#fff' } },
      legend: { orient: 'vertical', right: '5%', top: 'center', textStyle: { color: '#cbd5e1' } },
      series: [{
        type: 'pie',
        radius: ['55%', '80%'],
        center: ['35%', '50%'],
        itemStyle: { borderColor: 'rgba(10,14,23,0.8)', borderWidth: 2 },
        label: { show: false },
        data: Object.entries(data.modeDistribution).map(([name, value]) => ({
          name, value, itemStyle: { color: modeColors[name] || '#38bdf8' }
        }))
      }]
    })
    charts.push(modeChart)
  }

  // --- 2. 拥堵趋势折线图 ---
  if (trendChartRef.value) {
    const trendChart = echarts.init(trendChartRef.value)
    const days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    
    trendChart.setOption({
      grid: { top: 30, bottom: 20, left: 40, right: 20 },
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(10,14,23,0.9)', textStyle: { color: '#fff' } },
      xAxis: { 
        type: 'category', 
        data: days, 
        axisLabel: { color: '#94a3b8' }, 
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } 
      },
      yAxis: { 
        type: 'value', 
        min: 'dataMin',
        axisLabel: { color: '#94a3b8' }, 
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } 
      },
      series: [{
        data: data.trend, 
        type: 'line', 
        smooth: true, 
        symbolSize: 8,
        itemStyle: { color: '#FF4D00' },
        lineStyle: { width: 3, color: '#FF4D00' },
        areaStyle: { 
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(255, 77, 0, 0.4)' }, 
            { offset: 1, color: 'rgba(255, 77, 0, 0)' }
          ]) 
        }
      }]
    })
    charts.push(trendChart)
  }
}

const handleResize = () => {
  charts.forEach(c => c.resize())
}

onMounted(() => {
  loadData()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  charts.forEach(c => c.dispose())
})
</script>

<style scoped>
/* ================= 全局容器 ================= */
.region-detail-container {
  min-height: calc(100vh - 64px);
  background-color: #0A0E17;
  color: #e2e8f0;
  padding: 20px 30px;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  box-sizing: border-box;
}

/* ================= 面包屑导航 ================= */
.breadcrumb-header {
  display: flex;
  align-items: center;
  font-size: 14px;
  margin-bottom: 24px;
  color: #94a3b8;
}
.back-link {
  color: #38bdf8;
  text-decoration: none;
  display: flex;
  align-items: center;
  transition: opacity 0.3s;
}
.back-link:hover { opacity: 0.8; }
.back-arrow { margin-right: 4px; font-weight: bold; }
.separator { margin: 0 10px; color: #475569; }
.current-page { color: #cbd5e1; }
.region-name { 
  color: #fff; 
  font-weight: 600; 
  margin-left: 10px;
  padding-left: 10px;
  border-left: 2px solid #38bdf8;
}

/* ================= 状态提示 ================= */
.status-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 400px;
  color: #94a3b8;
  font-size: 16px;
}
.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(56, 189, 248, 0.2);
  border-top-color: #38bdf8;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.empty-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.8; }

/* ================= 布局与毛玻璃卡片 ================= */
.detail-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.dashboard-row {
  display: grid;
  gap: 20px;
}
.row-1 { grid-template-columns: 1fr 1fr; }
.row-2 { grid-template-columns: 1fr 1fr; }
.row-3 { grid-template-columns: 1fr; }

@media (max-width: 1024px) {
  .row-1, .row-2 { grid-template-columns: 1fr; }
}

.glass-card {
  background: rgba(15, 25, 45, 0.6);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(56, 189, 248, 0.25);
  border-radius: 16px;
  padding: 20px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
}
.glass-card:hover {
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
  border-color: rgba(56, 189, 248, 0.5);
}
.card-title {
  font-size: 16px;
  color: #e2e8f0;
  font-weight: 600;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
}
.card-title::before {
  content: '';
  display: inline-block;
  width: 4px;
  height: 16px;
  background-color: #38bdf8;
  margin-right: 8px;
  border-radius: 2px;
}

/* ================= 具体组件样式 ================= */
/* 核心指标 */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 15px;
  flex: 1;
  align-items: center;
}
.metric-item {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 12px;
  padding: 20px 10px;
  text-align: center;
}
.metric-label { font-size: 13px; color: #94a3b8; margin-bottom: 10px; }
.metric-value { 
  font-size: 32px; 
  font-weight: bold; 
  font-family: 'Din', 'Arial', sans-serif;
  text-shadow: 0 0 10px currentColor;
}
.text-red { color: #FF0033; }
.text-green { color: #00FF88; }
.text-orange { color: #FF9900; }

/* 图表容器 */
.chart-container { width: 100%; height: 220px; }

/* 表格样式 */
.cyber-table {
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
  background-color: transparent !important;
  font-size: 14px;
}
:deep(.el-table__inner-wrapper::before) { display: none; }
:deep(.el-table tbody tr:hover > td) { background-color: rgba(56, 189, 248, 0.1) !important; }

/* 排名徽章 */
.rank-badge {
  display: inline-block;
  width: 24px;
  height: 24px;
  line-height: 24px;
  border-radius: 4px;
  background: rgba(255,255,255,0.1);
  color: #fff;
  font-weight: bold;
}
.rank-1 { background: #FF0033; box-shadow: 0 0 8px #FF0033; }
.rank-2 { background: #FF6600; box-shadow: 0 0 8px #FF6600; }
.rank-3 { background: #FF9900; box-shadow: 0 0 8px #FF9900; }
.text-highlight { color: #38bdf8; font-weight: bold; font-family: monospace; font-size: 16px; }

/* 治理建议 */
.suggestion-content {
  font-size: 16px;
  line-height: 1.8;
  color: #cbd5e1;
  padding: 10px 20px;
  background: linear-gradient(90deg, rgba(56,189,248,0.1), transparent);
  border-left: 4px solid #38bdf8;
  border-radius: 4px;
  position: relative;
}
.quote-mark {
  color: rgba(56,189,248,0.4);
  font-size: 24px;
  font-family: serif;
  font-weight: bold;
  vertical-align: middle;
}

/* 动画 */
.fade-up { animation: fadeInUp 0.5s ease forwards; }
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>