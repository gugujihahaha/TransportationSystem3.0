<template>
  <div class="fui-user-center">
    <div class="cyber-bg">
      <div class="scan-line"></div>
    </div>

    <div class="dashboard-grid" :class="{ 'is-loaded': isLoaded }">
      
      <aside class="profile-panel panel-glass">
        <div class="panel-border-top"></div>
        
        <div class="avatar-container">
          <div class="avatar-ring"></div>
          <div class="avatar-img">
            <span class="avatar-text">{{ userInitials }}</span>
          </div>
          <div class="level-badge" :class="levelColor">LV.{{ userLevel }} {{ levelTitle }}</div>
        </div>

        <div class="user-identity">
          <h2 class="username">{{ authStore.username || '未命名研究员' }}</h2>
          <div class="affiliation">安徽师范大学 - 计算机与信息学院</div>
          <div class="id-hash">Token Status: {{ authStore.isAuthenticated() ? 'Active (Secured)' : 'Invalid' }}</div>
        </div>

        <div class="info-list">
          <div class="info-item">
            <span class="label">当前使用模型</span>
            <span class="value text-cyan">Exp4 (Focal Loss)</span>
          </div>
          <div class="info-item">
            <span class="label">真实轨迹解析</span>
            <span class="value text-gold">{{ totalRecords }} 条</span>
          </div>
          <div class="info-item">
            <span class="label">绿色出行指数</span>
            <span class="value" :class="greenScore >= 60 ? 'text-green' : 'text-cyan'">{{ greenScore }} / 100</span>
          </div>
        </div>

        <button class="primary-action-btn history-btn" @click="router.push('/history')">
          <span>📂 查阅完整历史档案</span>
        </button>

        <button class="primary-action-btn edit-btn" @click="handleLogout">
          <span>安全退出登录</span>
        </button>
      </aside>

      <main class="matrix-panel">
        
        <div class="stat-cards">
          <div class="stat-card panel-glass">
            <div class="stat-icon text-green">🌿</div>
            <div class="stat-info">
              <div class="stat-title">预估碳减排量 (基于真实里程)</div>
              <div class="stat-value text-green">{{ totalCarbonReduction }} <span class="unit">kg CO₂</span></div>
            </div>
          </div>
          <div class="stat-card panel-glass">
            <div class="stat-icon text-cyan">🚲</div>
            <div class="stat-info">
              <div class="stat-title">慢行交通占比 (Walk+Bike)</div>
              <div class="stat-value text-cyan">{{ slowTrafficRatio }} <span class="unit">%</span></div>
            </div>
          </div>
          <div class="stat-card panel-glass">
            <div class="stat-icon text-gold">🎯</div>
            <div class="stat-info">
              <div class="stat-title">当前系统状态</div>
              <div class="stat-value text-gold">{{ trajectoryStore.loading ? '模型推断中...' : '推断引擎空闲' }}</div>
            </div>
          </div>
        </div>

        <div class="chart-section">
          <div class="chart-container panel-glass radar-box">
            <h3 class="panel-title">真实个人出行特征画像</h3>
            <div v-if="totalRecords === 0" class="empty-state text-cyan">
              [ 等待数据汇入... 请先前往数据舱进行轨迹预测 ]
            </div>
            <div v-else ref="radarChartRef" class="echarts-wrapper"></div>
          </div>

          <div class="chart-container panel-glass progress-box">
            <h3 class="panel-title">AI 推断出行分布 (全局历史)</h3>
            <div class="progress-list" v-if="totalRecords > 0">
              <div class="progress-item" v-for="mode in dynamicTravelModes" :key="mode.name">
                <div class="progress-info">
                  <span class="mode-name">{{ mode.name }}</span>
                  <span class="mode-percent" :style="{ color: mode.color }">{{ mode.count }} 次 ({{ mode.percent }}%)</span>
                </div>
                <div class="progress-track">
                  <div class="progress-bar" 
                       :style="{ width: isLoaded ? mode.percent + '%' : '0%', backgroundColor: mode.color, boxShadow: `0 0 10px ${mode.color}` }">
                  </div>
                </div>
              </div>
            </div>
            <div v-else class="empty-state text-cyan">No Records Found.</div>
          </div>
        </div>

<!-- 碳普惠入口卡片 -->
<div class="carbon-entry-banner panel-glass" @click="router.push('/green-travel')">
  <div class="banner-content">
    <div class="banner-icon text-green">🌱</div>
    <div class="banner-text">
      <h3 class="banner-title">绿色出行碳普惠</h3>
      <p>上传轨迹，核算碳减排，领取 AI 专属环保表扬信与海报</p>
    </div>
    <div class="banner-arrow">→</div>
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

// --- 真实数据计算逻辑 ---

// 1. 基础信息
const userInitials = computed(() => (authStore.username ? authStore.username.charAt(0).toUpperCase() : 'U'))
const totalRecords = computed(() => trajectoryStore.history.length)

// 2. 动态等级计算
const userLevel = computed(() => Math.floor(totalRecords.value / 10) + 1)
const levelTitle = computed(() => {
  if (userLevel.value >= 10) return '交通架构师'
  if (userLevel.value >= 5) return '环保先锋'
  return '初级探索者'
})
const levelColor = computed(() => userLevel.value >= 5 ? 'bg-gold' : 'bg-green')

// 精准匹配后端字段 predicted_mode，并统一转为小写处理
const getRecordMode = (record: any): string => {
  if (!record) return 'unknown'
  return String(record.predicted_mode || record.pred_label || record.mode || 'unknown').toLowerCase()
}

// 3. 分类统计逻辑 (匹配小写，确保雷达图真实激活)
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

// 4. 业务指标核算
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

// 将计算逻辑改为“基于真实轨迹里程(distance)”的科学计算
const totalCarbonReduction = computed(() => {
  let reduction = 0
  trajectoryStore.history.forEach(record => {
    const mode = getRecordMode(record)
    const distanceKm = (record.distance || 0) / 1000 // 数据库存的是米，转为公里
    
    // 各模态低碳因子：步行 0.15kg/km, 骑行 0.10kg/km, 公交/地铁 0.05kg/km
    if (mode.includes('walk')) {
      reduction += distanceKm * 0.15
    } else if (mode.includes('bike')) {
      reduction += distanceKm * 0.10
    } else if (mode.includes('bus') || mode.includes('subway')) {
      reduction += distanceKm * 0.05
    }
  })
  return reduction.toFixed(2)
})

// 5. 动态进度条数据构建
const dynamicTravelModes = computed(() => {
  const total = totalRecords.value || 1 
  return [
    { name: '步行 (Walk)', count: modeCounts.value['Walk'], percent: ((modeCounts.value['Walk'] / total) * 100).toFixed(1), color: '#00ff88' },
    { name: '骑行 (Bike)', count: modeCounts.value['Bike'], percent: ((modeCounts.value['Bike'] / total) * 100).toFixed(1), color: '#00f0ff' },
    { name: '公交 (Bus)', count: modeCounts.value['Bus'], percent: ((modeCounts.value['Bus'] / total) * 100).toFixed(1), color: '#4dabf7' },
    { name: '地铁 (Subway)', count: modeCounts.value['Subway'], percent: ((modeCounts.value['Subway'] / total) * 100).toFixed(1), color: '#b197fc' },
    { name: '机动车 (Car&Taxi)', count: modeCounts.value['Car & taxi'], percent: ((modeCounts.value['Car & taxi'] / total) * 100).toFixed(1), color: '#ff6b6b' },
  ]
})

// --- ECharts 图表渲染 ---
const initRadarChart = () => {
  if (!radarChartRef.value || totalRecords.value === 0) return
  if (!chartInstance.value) {
    chartInstance.value = echarts.init(radarChartRef.value)
  }
  
const option = {
    backgroundColor: 'transparent',
    radar: {
      indicator: [
        { name: '步行 (Walk)', max: Math.max(totalRecords.value, 1) },
        { name: '骑行 (Bike)', max: Math.max(totalRecords.value, 1) },
        { name: '公交 (Bus)', max: Math.max(totalRecords.value, 1) },
        { name: '地铁 (Subway)', max: Math.max(totalRecords.value, 1) },
        { name: '火车 (Train)', max: Math.max(totalRecords.value, 1) },
        { name: '机动车 (Car)', max: Math.max(totalRecords.value, 1) }
      ],
      shape: 'polygon',
      radius: '65%',
      splitNumber: 4,
      // 坐标轴文字颜色（改亮一点）
      axisName: { color: '#e5eaf3', fontSize: 13, fontWeight: 'bold' },
      // 雷达图内部的网格线颜色（发光青色透明）
      splitLine: { 
        lineStyle: { 
          color: ['rgba(0, 240, 255, 0.1)', 'rgba(0, 240, 255, 0.2)', 'rgba(0, 240, 255, 0.4)', 'rgba(0, 240, 255, 0.6)'] 
        } 
      },
      splitArea: { show: false },
      // 从中心射出的轴线颜色
      axisLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.5)' } }
    },
    series: [{
      name: '真实数据画像',
      type: 'radar',
      data: [{
        value: [
          modeCounts.value['Walk'], 
          modeCounts.value['Bike'], 
          modeCounts.value['Bus'], 
          modeCounts.value['Subway'], 
          modeCounts.value['Train'], 
          modeCounts.value['Car & taxi']
        ],
        name: '当前真实记录',
        // 数据点颜色（高亮青色）
        itemStyle: { color: '#00f0ff' },
        lineStyle: { width: 2, color: '#00f0ff', shadowBlur: 15, shadowColor: '#00f0ff' },
        areaStyle: { color: 'rgba(0, 240, 255, 0.3)' }
      }]
    }]
  }
  chartInstance.value.setOption(option)
}

watch(() => trajectoryStore.history, () => {
  if (totalRecords.value > 0) {
    setTimeout(initRadarChart, 100)
  }
}, { deep: true })

const handleResize = () => {
  chartInstance.value?.resize()
}

const handleLogout = () => {
  authStore.logout()
  router.push('/login')
}

onMounted(async () => {  
  await trajectoryStore.fetchHistory() 

  setTimeout(() => {
    isLoaded.value = true
    if (totalRecords.value > 0) {
      initRadarChart()
    }
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
  --theme-dark: #070b19;
  --theme-cyan: #00f0ff;
  --theme-green: #00ff88;
  --theme-gold: #ffb800;
  --theme-blue: #4dabf7;
  --glass-bg: rgba(13, 20, 40, 0.5);
  --glass-border: rgba(0, 240, 255, 0.15);
  
  min-height: 100vh;
  background-color: var(--theme-dark);
  color: #fff;
  padding: 40px 5%;
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif;
  position: relative;
  overflow: hidden;
}

.cyber-bg {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(circle at 50% 0%, rgba(0, 87, 255, 0.05) 0%, transparent 70%);
  z-index: 0; pointer-events: none;
}
.scan-line {
  width: 100%; height: 2px;
  background: linear-gradient(90deg, transparent, var(--theme-cyan), transparent);
  position: absolute; opacity: 0.3; animation: scan 8s linear infinite;
}

.dashboard-grid {
  position: relative; z-index: 1; display: grid; grid-template-columns: 320px 1fr;
  gap: 30px; max-width: 1600px; margin: 0 auto;
  opacity: 0; transform: translateY(20px); transition: all 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.dashboard-grid.is-loaded { opacity: 1; transform: translateY(0); }

.panel-glass {
  background: var(--glass-bg); border: 1px solid var(--glass-border);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border-radius: 16px; position: relative; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}
.panel-title {
  font-size: 1.1rem; color: #cbd5e1; border-bottom: 1px solid rgba(255,255,255,0.05);
  padding-bottom: 15px; margin-bottom: 20px; font-weight: 600; letter-spacing: 1px;
}
.history-btn {
  width: 100%; padding: 12px; background: rgba(0, 240, 255, 0.1); 
  border: 1px solid var(--theme-cyan); color: var(--theme-cyan);
  border-radius: 8px; font-weight: bold; cursor: pointer; 
  transition: all 0.3s; margin-bottom: 15px; 
}
.history-btn:hover {
  background: var(--theme-cyan); color: #000; box-shadow: 0 0 15px var(--theme-cyan);
}
.text-cyan { color: var(--theme-cyan); text-shadow: 0 0 8px rgba(0,240,255,0.4); }
.text-green { color: var(--theme-green); text-shadow: 0 0 8px rgba(0,255,136,0.4); }
.text-gold { color: var(--theme-gold); text-shadow: 0 0 8px rgba(255,184,0,0.4); }
.bg-green { background: var(--theme-green); color: #000; box-shadow: 0 0 10px var(--theme-green); }
.bg-gold { background: var(--theme-gold); color: #000; box-shadow: 0 0 10px var(--theme-gold); }

/* 左侧面板 */
.profile-panel { padding: 40px 30px; display: flex; flex-direction: column; align-items: center; }
.panel-border-top {
  position: absolute; top: 0; left: 10%; width: 80%; height: 2px;
  background: linear-gradient(90deg, transparent, var(--theme-cyan), transparent);
}
.avatar-container { position: relative; width: 120px; height: 120px; margin-bottom: 25px; }
.avatar-ring {
  position: absolute; inset: -5px; border: 2px dashed var(--theme-cyan);
  border-radius: 50%; animation: spin 15s linear infinite;
}
.avatar-img {
  width: 100%; height: 100%; background: linear-gradient(135deg, #0f172a, #1e293b);
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  border: 2px solid rgba(0, 240, 255, 0.5); box-shadow: inset 0 0 20px rgba(0,240,255,0.2);
}
.avatar-text { font-size: 2.5rem; font-weight: 900; color: var(--theme-cyan); }
.level-badge {
  position: absolute; bottom: -10px; left: 50%; transform: translateX(-50%);
  font-weight: bold; font-size: 0.8rem; padding: 4px 12px; border-radius: 12px; white-space: nowrap;
}
.user-identity { text-align: center; margin-bottom: 30px; width: 100%; }
.username { font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; letter-spacing: 1px; }
.affiliation { font-size: 0.9rem; color: #94a3b8; margin-bottom: 5px; }
.id-hash { font-family: 'Courier New', Courier, monospace; font-size: 0.8rem; color: rgba(255,255,255,0.4); }
.info-list { width: 100%; margin-bottom: 40px; }
.info-item { display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px dashed rgba(255,255,255,0.1); }
.info-item .label { color: #cbd5e1; font-size: 0.95rem; }
.info-item .value { font-weight: bold; font-family: 'Courier New', Courier, monospace; }

.edit-btn {
  width: 100%; padding: 12px; background: transparent; border: 1px solid #ff6b6b; color: #ff6b6b;
  border-radius: 8px; font-weight: bold; cursor: pointer; transition: all 0.3s;
}
.edit-btn:hover { background: rgba(255, 107, 107, 0.1); box-shadow: 0 0 15px rgba(255,107,107,0.3); }

/* 右侧面板 */
.matrix-panel { display: flex; flex-direction: column; gap: 25px; }
.stat-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 25px; }
.stat-card { padding: 25px; display: flex; align-items: center; gap: 20px; transition: transform 0.3s; }
.stat-card:hover { transform: translateY(-5px); border-color: rgba(255,255,255,0.3); }
.stat-icon { font-size: 2.5rem; }
.stat-title { font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px; }
.stat-value { font-size: 2rem; font-weight: 800; font-family: 'Courier New', Courier, monospace; }
.stat-value .unit { font-size: 1rem; color: #64748b; font-weight: normal; text-shadow: none; }

.chart-section { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; height: 350px; }
.chart-container { padding: 25px; display: flex; flex-direction: column; }
.echarts-wrapper { flex: 1; width: 100%; min-height: 250px; }
.empty-state { text-align: center; color: rgba(255,255,255,0.3); margin-top: 50px; font-style: italic; }

.progress-list { display: flex; flex-direction: column; justify-content: space-around; height: 100%; }
.progress-item { margin-bottom: 15px; }
.progress-info { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9rem; }
.mode-name { color: #e2e8f0; }
.mode-percent { font-weight: bold; font-family: 'Courier New', Courier, monospace; }
.progress-track { width: 100%; height: 8px; background: rgba(0,0,0,0.5); border-radius: 4px; overflow: hidden; }
.progress-bar { height: 100%; border-radius: 4px; transition: width 1.5s cubic-bezier(0.2, 0.8, 0.2, 1); }

@keyframes spin { 100% { transform: rotate(360deg); } }
@keyframes scan { 0% { top: -10%; } 100% { top: 110%; } }

@media (max-width: 1024px) {
  .dashboard-grid { grid-template-columns: 1fr; }
  .chart-section { grid-template-columns: 1fr; height: auto; }
}

.archive-banner { margin-top: 25px; padding: 25px; cursor: pointer; transition: all 0.4s; overflow: hidden; position: relative; }
.archive-banner:hover { border-color: var(--theme-cyan); transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0, 240, 255, 0.15); }
.archive-banner::after { content: ''; position: absolute; right: -50px; bottom: -50px; width: 150px; height: 150px; background: radial-gradient(circle, rgba(0, 240, 255, 0.1) 0%, transparent 70%); border-radius: 50%; }
.banner-content { display: flex; align-items: center; justify-content: space-between; position: relative; z-index: 1;}
.banner-icon { font-size: 3rem; margin-right: 25px; }
.banner-text { flex: 1; }
.banner-title { font-size: 1.4rem; color: #fff; margin-bottom: 8px; font-weight: bold; letter-spacing: 1px; }
.banner-text p { color: #94a3b8; font-size: 0.95rem; margin: 0; }
.banner-arrow { font-size: 2rem; color: var(--theme-cyan); font-weight: bold; margin-left: 20px; transition: transform 0.3s; }
.archive-banner:hover .banner-arrow { transform: translateX(10px); }
</style>