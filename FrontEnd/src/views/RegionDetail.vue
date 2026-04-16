<template>
  <div class="region-detail-container">
    <div class="breadcrumb-header">
      <router-link to="/" class="back-link">
        <span class="back-arrow">←</span> 返回全局态势
      </router-link>
      <span class="separator">/</span>
      <span class="current-page">区域多模态画像</span>
      <span class="separator">/</span>
      <span class="region-name">{{ regionName }}</span>
    </div>

    <div v-if="loading" class="status-container">
      <div class="loading-spinner"></div>
      <p>正在拉取区域多源融合数据...</p>
    </div>
    <div v-else-if="!regionData" class="status-container">
      <div class="empty-icon">📂</div>
      <p>该区域（{{ regionName }}）数据整理中，敬请期待...</p>
    </div>

    <div v-else class="detail-content fade-up">
      <div class="dashboard-row row-1">
        <div class="glass-card metrics-card">
          <div class="card-title">区域出行核心特征</div>
          <div class="metrics-grid">
            <div class="metric-item">
              <div class="metric-label">区域轨迹覆盖量</div>
              <div class="metric-value text-blue">{{ regionData.pointCount.toLocaleString() }}</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">绿色出行比例</div>
              <div class="metric-value text-green">{{ (regionData.greenRatio * 100).toFixed(1) }}%</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">全市低碳榜排名</div>
              <div class="metric-value text-orange">TOP {{ regionData.rank }}</div>
            </div>
          </div>
        </div>

        <div class="glass-card chart-card">
          <div class="card-title">模型识别出行结构占比</div>
          <div ref="modeChartRef" class="chart-container"></div>
        </div>
      </div>

      <div class="dashboard-row row-2">
        <div class="glass-card table-card">
          <div class="card-title">重点微循环网格 (高频绿色出行商圈)</div>
          <el-table 
            :data="regionData.topGreenGrids" 
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
            <el-table-column prop="name" label="网格/商圈名称" />
            <el-table-column prop="ratio" label="绿色出行占比" align="right">
              <template #default="scope">
                <span class="text-highlight" style="color: #00FF88;">{{ scope.row.ratio }}</span>
              </template>
            </el-table-column>
          </el-table>
        </div>

        <div class="glass-card chart-card">
          <div class="card-title">近7天绿色出行比例趋势分析</div>
          <div ref="trendChartRef" class="chart-container"></div>
        </div>
      </div>

      <div class="dashboard-row row-3">
        <div class="glass-card suggestion-card">
          <div class="card-title">基于多模态融合模型的低碳建设洞察</div>
          <div class="suggestion-content">
            <span class="quote-mark">“</span>
            {{ regionData.insight }}
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
const regionName = ref(route.params.name || route.query.name || '海淀区')

const loading = ref(true)
const regionData = ref(null)

const modeChartRef = ref(null)
const trendChartRef = ref(null)
const charts = []

// 模拟各个区域的新版测试数据
const mockRegionDatabase = {
  '朝阳区': {
    pointCount: 384500,
    greenRatio: 0.38,
    rank: 3,
    modeDistribution: { '步行': 22, '骑行': 16, '公交': 25, '地铁': 20, '火车': 2, '小汽车': 15 },
    topGreenGrids: [
      { name: '三里屯-工人体育场', ratio: '62.4%' },
      { name: '望京 SOHO 核心区', ratio: '58.1%' },
      { name: '国贸 CBD', ratio: '54.3%' },
      { name: '朝阳大悦城周边', ratio: '49.8%' }
    ],
    trend: [0.35, 0.36, 0.38, 0.37, 0.39, 0.41, 0.38],
    insight: "Exp4 模型精准识别出该区域呈现极强的‘潮汐接驳’特征：早晚高峰存在大量‘骑行-地铁’多模态切换行为。建议在国贸及望京地铁站点周边 500 米范围扩容非机动车电子围栏，以承接庞大的共享单车接驳需求。"
  },
  '海淀区': {
    pointCount: 425600,
    greenRatio: 0.46,
    rank: 1,
    modeDistribution: { '步行': 25, '骑行': 21, '公交': 22, '地铁': 18, '火车': 1, '小汽车': 13 },
    topGreenGrids: [
      { name: '中关村软件园', ratio: '71.2%' },
      { name: '五道口-清华科技园', ratio: '68.5%' },
      { name: '人大-知春路沿线', ratio: '64.0%' },
      { name: '西北旺区域', ratio: '58.7%' }
    ],
    trend: [0.42, 0.44, 0.46, 0.45, 0.47, 0.49, 0.46],
    insight: "得益于高校与互联网园区的密集分布，该区常态化绿色出行比例领跑全市。Exp2 空间特征分析表明，园区内部步道及周边专用自行车道利用率极高，建议作为‘慢行系统友好型示范区’向全市推广。"
  }
}

const loadData = async () => {
  try {
    // 真实项目中这里依然可以用 fetch 请求 /region_data.json
    // 这里为了演示直接 fallback 到 mock 数据库
    await new Promise(resolve => setTimeout(resolve, 600)) // 模拟网络延迟
    regionData.value = mockRegionDatabase[regionName.value] || mockRegionDatabase['朝阳区']
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

const initCharts = () => {
  if (!regionData.value) return
  const data = regionData.value

  // --- 1. 出行结构环形图 ---
  if (modeChartRef.value) {
    const modeChart = echarts.init(modeChartRef.value)
    const modeColors = { '步行': '#00FF88', '骑行': '#00FFFF', '公交': '#FFDD00', '地铁': '#CC33FF', '火车': '#FF6600', '小汽车': '#FF0033' }
    
    modeChart.setOption({
      tooltip: { trigger: 'item', backgroundColor: 'rgba(10,14,23,0.9)', textStyle: { color: '#fff', fontWeight: 'bold' }, formatter: '{b}: {c}%' },
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

  // --- 2. 绿色出行趋势折线图 (更换为环保主题色) ---
  if (trendChartRef.value) {
    const trendChart = echarts.init(trendChartRef.value)
    const days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    trendChart.setOption({
      grid: { top: 30, bottom: 20, left: 45, right: 20 },
      tooltip: { 
        trigger: 'axis', backgroundColor: 'rgba(10,14,23,0.9)', 
        textStyle: { color: '#fff', fontWeight:'bold' },
        formatter: (params) => `${params[0].name}<br/>比例: ${(params[0].value * 100).toFixed(1)}%`
      },
      xAxis: { 
        type: 'category', data: days, 
        axisLabel: { color: '#94a3b8' }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } 
      },
      yAxis: { 
        type: 'value', min: 'dataMin',
        axisLabel: { color: '#94a3b8', formatter: (val) => (val * 100).toFixed(0) + '%' }, 
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } 
      },
      series: [{
        data: data.trend, 
        type: 'line', smooth: true, symbolSize: 8,
        itemStyle: { color: '#00FF88' }, // 绿色主题
        lineStyle: { width: 3, color: '#00FF88' },
        areaStyle: { 
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(0, 255, 136, 0.4)' }, 
            { offset: 1, color: 'rgba(0, 255, 136, 0)' }
          ]) 
        }
      }]
    })
    charts.push(trendChart)
  }
}

const handleResize = () => { charts.forEach(c => c.resize()) }

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
.region-detail-container {
  min-height: calc(100vh - 64px);
  background-color: transparent;
  color: #e2e8f0;
  padding: 0 10px;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  box-sizing: border-box;
}

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
.metric-label { font-size: 13px; color: #94a3b8; margin-bottom: 10px; font-weight: bold; }
.metric-value { 
  font-size: 32px; 
  font-weight: bold; 
  font-family: 'Din', 'Arial', sans-serif;
  text-shadow: 0 0 10px currentColor;
}
.text-blue { color: #38bdf8; }
.text-green { color: #00FF88; }
.text-orange { color: #FFD700; }

.chart-container { width: 100%; height: 220px; }

.cyber-table {
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
  background-color: transparent !important;
  font-size: 14px;
}
:deep(.el-table__inner-wrapper::before) { display: none; }
:deep(.el-table tbody tr:hover > td) { background-color: rgba(56, 189, 248, 0.1) !important; }

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
.rank-1 { background: #00FF88; box-shadow: 0 0 8px rgba(0,255,136,0.6); color: #000; }
.rank-2 { background: #00A8FF; box-shadow: 0 0 8px rgba(0,168,255,0.6); }
.rank-3 { background: #FFD700; box-shadow: 0 0 8px rgba(255,215,0,0.6); color: #000; }
.text-highlight { font-weight: bold; font-family: monospace; font-size: 16px; }

.suggestion-content {
  font-size: 15px;
  line-height: 1.8;
  color: #cbd5e1;
  padding: 15px 20px;
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

.fade-up { animation: fadeInUp 0.5s ease forwards; }
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>