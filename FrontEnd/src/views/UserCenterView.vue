<template>
  <div class="fui-user-center">
    <div class="dashboard-grid" :class="{ 'is-loaded': isLoaded }">
      
      <aside class="profile-panel panel-standard">
        <div class="user-profile-header">
          <div class="avatar-box">
            <span class="avatar-text">{{ userInitials }}</span>
          </div>
          <h2 class="username">{{ authStore.username || '系统用户' }}</h2>
          <div class="user-tag" :class="levelColor">{{ levelTitle }}</div>
        </div>

        <div class="info-group">
          <div class="info-row">
            <span class="label">账户状态</span>
            <span class="value text-status">{{ authStore.isAuthenticated() ? '在线' : '离线' }}</span>
          </div>
          <div class="info-row">
            <span class="label">数据版本</span>
            <span class="value">v3.0.4 - Release</span>
          </div>
          <div class="info-row">
            <span class="label">历史记录</span>
            <span class="value">{{ totalRecords }} 条</span>
          </div>
          <div class="info-row">
            <span class="label">绿色出行指数</span>
            <span class="value">{{ greenScore }} / 100</span>
          </div>
        </div>

        <div class="action-group">
          <button class="btn btn-outline" @click="router.push('/history')">进入历史档案</button>
          <button class="btn btn-danger" @click="handleLogout">退出登录</button>
        </div>
      </aside>

      <main class="main-content">
        
        <div class="stats-row">
          <div class="stat-item panel-standard">
            <div class="stat-label">累计碳减排量 (kg)</div>
            <div class="stat-value text-green">{{ totalCarbonReduction }}</div>
          </div>
          <div class="stat-item panel-standard">
            <div class="stat-label">慢行交通占比 (%)</div>
            <div class="stat-value text-blue">{{ slowTrafficRatio }}</div>
          </div>
          <div class="stat-item panel-standard">
            <div class="stat-label">系统运行状态</div>
            <div class="stat-value text-muted" style="font-size: 1.2rem;">
              {{ trajectoryStore.loading ? '正在同步数据...' : '运行正常' }}
            </div>
          </div>
        </div>

        <div class="analysis-row">
          <div class="chart-box panel-standard">
            <h3 class="chart-title">个人出行特征特征偏好</h3>
            <div v-if="totalRecords === 0" class="no-data">暂无数据汇入</div>
            <div v-else ref="radarChartRef" class="chart-instance"></div>
          </div>

          <div class="chart-box panel-standard">
            <h3 class="chart-title">历史出行方式统计</h3>
            <div class="distribution-list" v-if="totalRecords > 0">
              <div class="dist-item" v-for="mode in dynamicTravelModes" :key="mode.name">
                <div class="dist-info">
                  <span class="name">{{ mode.name }}</span>
                  <span class="detail">{{ mode.count }}次 / {{ mode.percent }}%</span>
                </div>
                <div class="progress-container">
                  <div class="progress-bar" :style="{ width: isLoaded ? mode.percent + '%' : '0%', backgroundColor: mode.color }"></div>
                </div>
              </div>
            </div>
            <div v-else class="no-data">无统计记录</div>
          </div>
        </div>

        <div class="footer-banner panel-standard" @click="router.push('/green-travel')">
          <div class="banner-body">
            <div class="banner-main">
              <h3>绿色出行与碳普惠评估</h3>
              <p>系统将基于历史轨迹生成量化的减排报告，支持导出学术级 PDF 报告及数据摘要。</p>
            </div>
            <div class="banner-link">查看详情 →</div>
          </div>
        </div>

      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, shallowRef } from 'vue'
import { useRouter } from 'vue-router'
import * as echarts from 'echarts'
import { useAuthStore } from '@/stores/auth'
import { useTrajectoryStore } from '@/stores/trajectory'

const router = useRouter()
const authStore = useAuthStore()
const trajectoryStore = useTrajectoryStore()

const isLoaded = ref(false)
const radarChartRef = ref<HTMLElement | null>(null)
const chartInstance = shallowRef<echarts.ECharts | null>(null)

// 1. 基础信息
const userInitials = computed(() => (authStore.username ? authStore.username.charAt(0).toUpperCase() : 'U'))
const totalRecords = computed(() => trajectoryStore.history.length)

// 2. 身份标签计算 
const userLevel = computed(() => Math.floor(totalRecords.value / 10) + 1)
const levelTitle = computed(() => {
  if (userLevel.value >= 10) return '荣誉：核心贡献者'
  if (userLevel.value >= 5) return '荣誉：活跃贡献者'
  return '荣誉：普通用户'
})
const levelColor = computed(() => userLevel.value >= 5 ? 'tag-gold' : 'tag-blue')

// 3. 业务逻辑
const getRecordMode = (record: any): string => {
  if (!record) return 'unknown'
  return String(record.predicted_mode || record.pred_label || record.mode || 'unknown').toLowerCase()
}

const modeCounts = computed(() => {
  const counts = { 'Walk': 0, 'Bike': 0, 'Bus': 0, 'Subway': 0, 'Train': 0, 'Car & taxi': 0 }
  trajectoryStore.history.forEach(p => {
    const mode = getRecordMode(p)
    if (mode.includes('walk')) counts['Walk']++
    else if (mode.includes('bike')) counts['Bike']++
    else if (mode.includes('bus')) counts['Bus']++
    else if (mode.includes('subway')) counts['Subway']++
    else if (mode.includes('train')) counts['Train']++
    else if (mode.includes('car') || mode.includes('taxi')) counts['Car & taxi']++
  })
  return counts
})

const slowTrafficRatio = computed(() => {
  if (totalRecords.value === 0) return '0.0'
  const slow = modeCounts.value['Walk'] + modeCounts.value['Bike']
  return ((slow / totalRecords.value) * 100).toFixed(1)
})

const greenScore = computed(() => {
  if (totalRecords.value === 0) return 0
  const green = modeCounts.value['Walk'] + modeCounts.value['Bike'] + modeCounts.value['Bus'] + modeCounts.value['Subway']
  return Math.round((green / totalRecords.value) * 100)
})

const totalCarbonReduction = computed(() => {
  let reduction = 0
  trajectoryStore.history.forEach(record => {
    const mode = getRecordMode(record)
    const distanceKm = (record.distance || 0) / 1000
    if (mode.includes('walk')) reduction += distanceKm * 0.15
    else if (mode.includes('bike')) reduction += distanceKm * 0.10
    else if (mode.includes('bus') || mode.includes('subway')) reduction += distanceKm * 0.05
  })
  return reduction.toFixed(2)
})

// 4. 配色方案
const dynamicTravelModes = computed(() => {
  const total = totalRecords.value || 1 
  return [
    { name: '步行 (Walk)', count: modeCounts.value['Walk'], percent: ((modeCounts.value['Walk'] / total) * 100).toFixed(1), color: '#64748b' },
    { name: '骑行 (Bike)', count: modeCounts.value['Bike'], percent: ((modeCounts.value['Bike'] / total) * 100).toFixed(1), color: '#3b82f6' },
    { name: '公交 (Bus)', count: modeCounts.value['Bus'], percent: ((modeCounts.value['Bus'] / total) * 100).toFixed(1), color: '#10b981' },
    { name: '地铁 (Subway)', count: modeCounts.value['Subway'], percent: ((modeCounts.value['Subway'] / total) * 100).toFixed(1), color: '#8b5cf6' },
    { name: '机动车 (Car)', count: modeCounts.value['Car & taxi'], percent: ((modeCounts.value['Car & taxi'] / total) * 100).toFixed(1), color: '#ef4444' },
  ]
})

const initRadarChart = () => {
  if (!radarChartRef.value || totalRecords.value === 0) return
  if (!chartInstance.value) chartInstance.value = echarts.init(radarChartRef.value)
  
  chartInstance.value.setOption({
    radar: {
      indicator: [
        { name: '步行', max: Math.max(totalRecords.value, 1) },
        { name: '骑行', max: Math.max(totalRecords.value, 1) },
        { name: '公交', max: Math.max(totalRecords.value, 1) },
        { name: '地铁', max: Math.max(totalRecords.value, 1) },
        { name: '火车', max: Math.max(totalRecords.value, 1) },
        { name: '机动车', max: Math.max(totalRecords.value, 1) }
      ],
      axisName: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#334155' } },
      splitArea: { show: false }
    },
    series: [{
      type: 'radar',
      data: [{
        value: [modeCounts.value['Walk'], modeCounts.value['Bike'], modeCounts.value['Bus'], modeCounts.value['Subway'], modeCounts.value['Train'], modeCounts.value['Car & taxi']],
        itemStyle: { color: '#3b82f6' },
        areaStyle: { color: 'rgba(59, 130, 246, 0.2)' }
      }]
    }]
  })
}

watch(() => trajectoryStore.history, () => {
  if (totalRecords.value > 0) setTimeout(initRadarChart, 100)
}, { deep: true })

const handleResize = () => chartInstance.value?.resize()
const handleLogout = () => { authStore.logout(); router.push('/login'); }

onMounted(async () => {  
  await trajectoryStore.fetchHistory() 
  setTimeout(() => {
    isLoaded.value = true
    if (totalRecords.value > 0) initRadarChart()
    window.addEventListener('resize', handleResize)
  }, 100)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance.value?.dispose()
})
</script>

<style scoped>
.fui-user-center {
  background-color: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 40px 20px;
  font-family: 'Inter', -apple-system, sans-serif;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 24px;
  max-width: 1300px;
  margin: 0 auto;
  opacity: 0;
  transition: opacity 0.5s ease;
}
.dashboard-grid.is-loaded { opacity: 1; }

/* 通用面板样式 */
.panel-standard {
  background-color: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  overflow: hidden;
}

/* 侧边栏 */
.profile-panel { padding: 30px 20px; display: flex; flex-direction: column; align-items: center; text-align: center; }
.avatar-box {
  width: 80px; height: 80px; background: #334155; border-radius: 50%;margin: 0 auto;
  display: flex; align-items: center; justify-content: center; margin-bottom: 16px;
  text-align: center;
}
.avatar-text { font-size: 2rem; font-weight: bold; color: #3b82f6; }
.username { font-size: 1.5rem; margin-bottom: 20px; font-weight: 600;  text-align: center; }
.user-tag { font-size: 0.75rem; padding: 4px 12px; border-radius: 20px; margin-bottom: 24px;display: inline-block; }
.tag-blue { background: rgba(59, 130, 246, 0.1); color: #3b82f6; border: 1px solid #3b82f6; }
.tag-gold { background: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid #f59e0b; }

.info-group { width: 100%; margin-bottom: 30px; }
.info-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #334155; font-size: 0.9rem; }
.info-row .label { color: #94a3b8; }
.text-status { color: #10b981; font-weight: bold; }

.action-group { width: 100%; display: flex; flex-direction: column; gap: 12px; }
.btn { padding: 10px; border-radius: 6px; font-weight: 500; cursor: pointer; transition: 0.2s; border: none; }
.btn-outline { background: transparent; border: 1px solid #475569; color: #e2e8f0; }
.btn-outline:hover { background: #334155; }
.btn-danger { background: #ef4444; color: white; }
.btn-danger:hover { background: #dc2626; }

/* 主区域 */
.main-content { display: flex; flex-direction: column; gap: 24px; }
.stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
.stat-item { padding: 20px; }
.stat-label { color: #94a3b8; font-size: 0.85rem; margin-bottom: 10px; }
.stat-value { font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.text-green { color: #10b981; }
.text-blue { color: #3b82f6; }

.analysis-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.chart-box { padding: 20px; min-height: 350px; }
.chart-title { font-size: 1rem; font-weight: 500; margin-bottom: 20px; color: #f8fafc; }
.chart-instance { height: 280px; width: 100%; }
.no-data { text-align: center; color: #64748b; padding-top: 100px; font-style: italic; }

.dist-item { margin-bottom: 16px; }
.dist-info { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px; }
.progress-container { height: 6px; background: #0f172a; border-radius: 3px; overflow: hidden; }
.progress-bar { height: 100%; transition: width 1s ease; }

/* 横幅 */
.footer-banner { padding: 24px; cursor: pointer; transition: 0.2s; }
.footer-banner:hover { border-color: #3b82f6; background: #1e293b; }
.banner-body { display: flex; justify-content: space-between; align-items: center; }
.banner-main h3 { margin: 0 0 8px 0; font-size: 1.1rem; }
.banner-main p { margin: 0; font-size: 0.9rem; color: #94a3b8; }
.banner-link { color: #3b82f6; font-weight: 600; }

@media (max-width: 1024px) {
  .dashboard-grid { grid-template-columns: 1fr; }
  .analysis-row { grid-template-columns: 1fr; }
  .stats-row { grid-template-columns: 1fr; }
}
</style>