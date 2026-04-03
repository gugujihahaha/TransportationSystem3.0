<template>
  <div class="fui-home-container">
    <div class="tech-bg">
      <div class="grid-overlay"></div>
      <div class="glow-sphere-1"></div>
      <div class="glow-sphere-2"></div>
    </div>

    <div class="content-wrapper" :class="{ 'is-loaded': isLoaded }">
      
      <header class="hero-section">
        <div class="tagline">National Award Level Project</div>
        <h1 class="main-title">
          <span class="text-glow">TrafficRec</span> 
          <span class="sub-text">多模态出行大模型</span>
        </h1>
        <p class="description">
          基于四阶递进式 PyTorch 引擎与星火大模型，精准溯源城市拥堵，构建碳普惠智能评估体系。
        </p>
        
        <div class="action-group">
          <button class="cyber-btn primary" @click="navigateTo('/dashboard')">
            <span class="btn-text">进入数据舱</span>
            <span class="btn-glitch"></span>
          </button>
          <button class="cyber-btn secondary" @click="navigateTo('/user-center')">
            <span class="btn-text">驾驶员档案</span>
          </button>
        </div>
      </header>

      <section class="dashboard-section">
        <div class="dash-panel glass-panel stats-col">
          <h3 class="panel-title"><span class="icon">📊</span> 平台实时运行指标</h3>
          <div class="stat-content">
            <div class="stat-box">
              <div class="stat-label">全网累计轨迹解析量</div>
              <div class="stat-value text-cyan">{{ Math.floor(animatedPredictions) }} <span class="unit">条</span></div>
            </div>
            <div class="stat-box">
              <div class="stat-label">核算碳资产总量 (CO₂)</div>
              <div class="stat-value text-green">{{ animatedCarbon.toFixed(2) }} <span class="unit">kg</span></div>
            </div>
            <div class="stat-box">
              <div class="stat-label">系统运行健康度</div>
              <div class="stat-value text-gold">99.9 <span class="unit">%</span></div>
            </div>
          </div>
        </div>

        <div class="dash-panel glass-panel chart-col">
          <h3 class="panel-title"><span class="icon">🎯</span> 交通流模态全景拓扑</h3>
          
          <div v-if="trajectoryStore.history.length < 3" class="radar-standby">
            <div class="radar-scope">
              <div class="radar-sweep"></div>
              <div class="radar-grid"></div>
              <div class="radar-dot dot-1"></div>
              <div class="radar-dot dot-2"></div>
            </div>
            <p class="standby-text">全球时空引擎已就绪<br><span class="standby-sub">等待数据流注入...</span></p>
          </div>
          
          <div v-else ref="roseChartRef" class="echarts-container"></div>
        </div>

        <div class="dash-panel glass-panel feed-col">
          <h3 class="panel-title"><span class="icon">📡</span> 实时态势感知链路</h3>
          <div class="feed-wrapper">
            <div v-if="recentLogs.length === 0" class="listening-state">
              <div class="pulse-ring"></div>
              <span>监听节点接入中...</span>
            </div>
            
            <div v-else class="feed-list" :class="{ 'scroll-anim': recentLogs.length > 5 }">
              <div v-for="(log, index) in recentLogs" :key="index" class="feed-item">
                <span class="feed-time">{{ formatTime(log.created_at) }}</span>
                <span class="feed-engine">[{{ log.model_id.toUpperCase() }}]</span>
                <span class="feed-mode" :class="getModeColorClass(log.predicted_mode)">
                  识别为 {{ log.predicted_mode.toUpperCase() }}
                </span>
                <span class="feed-conf">{{ (log.confidence * 100).toFixed(1) }}%</span>
              </div>
              <div v-for="(log, index) in recentLogs" :key="'dup-'+index" class="feed-item" v-if="recentLogs.length > 5">
                <span class="feed-time">{{ formatTime(log.created_at) }}</span>
                <span class="feed-engine">[{{ log.model_id.toUpperCase() }}]</span>
                <span class="feed-mode" :class="getModeColorClass(log.predicted_mode)">
                  识别为 {{ log.predicted_mode.toUpperCase() }}
                </span>
                <span class="feed-conf">{{ (log.confidence * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="engine-matrix">
        <div v-for="(engine, index) in engines" :key="index" class="engine-card">
          <div class="engine-icon">{{ engine.icon }}</div>
          <h3 class="engine-name">{{ engine.name }}</h3>
          <p class="engine-desc">{{ engine.desc }}</p>
          <div class="engine-metric">
            <span class="metric-label">Global Acc:</span>
            <span class="metric-value" :style="engine.id === 'exp4' ? 'color: #ffb800;' : ''">
              {{ engine.accuracy }}%
            </span>
          </div>
        </div>
      </section>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, shallowRef, nextTick } from 'vue'
import type { Ref } from 'vue'
import { useRouter } from 'vue-router'
import * as echarts from 'echarts'
import { useTrajectoryStore } from '@/stores/trajectory'

const router = useRouter()
const trajectoryStore = useTrajectoryStore()
const isLoaded = ref(false)

const engines = ref([
  { id: 'exp1', name: 'Exp1 基线引擎', desc: '纯轨迹运动学特征提取，支持基础分类。', accuracy: 83.30, icon: '🚀' },
  { id: 'exp2', name: 'Exp2 拓扑增强', desc: '融合 OSM 真实路网，车道级空间映射。', accuracy: 83.88, icon: '🗺️' },
  { id: 'exp3', name: 'Exp3 气象解耦', desc: '剥离恶劣天气对时空特征的噪声干扰。', accuracy: 84.50, icon: '⛈️' },
  { id: 'exp4', name: 'Exp4 终极推断', desc: 'Focal Loss 优化，牺牲全局准度攻克长尾难题。', accuracy: 81.57, icon: '🧠' }
])

const navigateTo = (path: string) => { router.push(path) }
const history = computed(() => trajectoryStore.history)

const totalPredictions = computed(() => history.value.length)
const animatedPredictions = ref(0)
const animatedCarbon = ref(0)

const getRecordMode = (record: any): string => {
  return String(record.predicted_mode || record.pred_label || record.mode || 'unknown').toLowerCase()
}

const totalCarbon = computed(() => {
  let reduction = 0
  history.value.forEach(record => {
    const mode = getRecordMode(record)
    const distanceKm = (record.distance || 0) / 1000
    if (mode.includes('walk')) reduction += distanceKm * 0.15
    else if (mode.includes('bike')) reduction += distanceKm * 0.10
    else if (mode.includes('bus') || mode.includes('subway')) reduction += distanceKm * 0.05
  })
  return Number(reduction.toFixed(2))
})

const animateValue = (targetRef: Ref<number>, finalValue: number, duration = 2000) => {
  const startTime = performance.now()
  const startValue = targetRef.value
  const update = (currentTime: number) => {
    const elapsed = currentTime - startTime
    const progress = Math.min(elapsed / duration, 1)
    const ease = 1 - Math.pow(1 - progress, 4)
    targetRef.value = startValue + (finalValue - startValue) * ease
    if (progress < 1) requestAnimationFrame(update)
    else targetRef.value = finalValue
  }
  requestAnimationFrame(update)
}

watch(totalPredictions, (newVal) => animateValue(animatedPredictions, newVal))
watch(totalCarbon, (newVal) => animateValue(animatedCarbon, newVal))

const recentLogs = computed(() => history.value.slice(0, 10))
const formatTime = (isoString?: string) => {
  if (!isoString) return '--:--:--'
  return new Date(isoString).toLocaleTimeString('zh-CN', { hour12: false })
}
const getModeColorClass = (mode: string) => {
  const m = mode.toLowerCase()
  if (m.includes('walk') || m.includes('bike')) return 'text-green'
  if (m.includes('bus') || m.includes('subway')) return 'text-cyan'
  if (m.includes('car') || m.includes('taxi')) return 'text-red'
  return 'text-gold'
}

const roseChartRef = ref<HTMLElement | null>(null)
const chartInstance = shallowRef<echarts.ECharts | null>(null)

const initRoseChart = () => {
  if (!roseChartRef.value || history.value.length < 3) return
  if (!chartInstance.value) chartInstance.value = echarts.init(roseChartRef.value)
  
  const counts = { '步行': 0, '骑行': 0, '公交': 0, '地铁': 0, '火车': 0, '机动车': 0 }
  
  history.value.forEach(p => {
    const m = getRecordMode(p)
    if (m.includes('walk')) counts['步行'] += 1
    else if (m.includes('bike')) counts['骑行'] += 1
    else if (m.includes('bus')) counts['公交'] += 1
    else if (m.includes('subway')) counts['地铁'] += 1
    else if (m.includes('train')) counts['火车'] += 1
    else if (m.includes('car') || m.includes('taxi')) counts['机动车'] += 1
  })

  const data = (Object.keys(counts) as Array<keyof typeof counts>)
    .map(key => ({ name: key, value: counts[key] }))
    .filter(d => d.value > 0)

  const option = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item', formatter: '{b} : {c}次 ({d}%)', backgroundColor: 'rgba(0,0,0,0.8)', textStyle: { color: '#fff' } },
    series: [
      {
        name: '模态分布',
        type: 'pie',
        radius: ['20%', '80%'],
        center: ['50%', '50%'],
        roseType: 'area',
        itemStyle: { borderRadius: 5 },
        label: { color: '#e5eaf3', fontSize: 11, formatter: '{b}\n{c}' }, // 缩小字体适应紧凑布局
        color: ['#00ff88', '#00f0ff', '#4dabf7', '#b197fc', '#ffb800', '#ff6b6b'],
        data: data.sort((a, b) => a.value - b.value)
      }
    ]
  }
  chartInstance.value.setOption(option)
}

watch(() => history.value, () => { nextTick(initRoseChart) }, { deep: true })

const handleResize = () => { chartInstance.value?.resize() }

onMounted(async () => {
  isLoaded.value = true
  await trajectoryStore.fetchHistory()
  
  animatedPredictions.value = 0
  animatedCarbon.value = 0
  animateValue(animatedPredictions, totalPredictions.value)
  animateValue(animatedCarbon, totalCarbon.value)
  
  nextTick(() => {
    initRoseChart()
    window.addEventListener('resize', handleResize)
  })
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance.value?.dispose()
})
</script>

<style scoped>

.fui-home-container { 
  min-height: 100vh; 
  background-color: #070b19; 
  color: #fff; 
  position: relative; 
  overflow-x: hidden; 
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif; 
  padding-bottom: 20px; /* 极小化底部留白 */
}

.tech-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
.grid-overlay { position: absolute; inset: 0; background-image: linear-gradient(rgba(0, 240, 255, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 240, 255, 0.05) 1px, transparent 1px); background-size: 50px 50px; transform: perspective(500px) rotateX(60deg); transform-origin: top; opacity: 0.3; }
.glow-sphere-1 { position: absolute; top: -10%; left: -10%; width: 50vw; height: 50vw; background: radial-gradient(circle, rgba(0, 87, 255, 0.15) 0%, transparent 70%); border-radius: 50%; filter: blur(50px); }
.glow-sphere-2 { position: absolute; bottom: -20%; right: -10%; width: 60vw; height: 60vw; background: radial-gradient(circle, rgba(0, 240, 255, 0.1) 0%, transparent 70%); border-radius: 50%; filter: blur(50px); }

.content-wrapper { 
  position: relative; 
  z-index: 1; 
  max-width: 1300px; /* 稍微缩小总宽度，让排版更聚拢 */
  margin: 0 auto; 
  padding: 10px 20px; 
  display: flex;
  flex-direction: column;
  gap: 20px; 
  opacity: 0; 
  transform: translateY(20px); 
  transition: all 1s ease-out; 
}
.content-wrapper.is-loaded { opacity: 1; transform: translateY(0); }

/* --- 1. 头部区域 --- */
.hero-section { 
  text-align: center; 
  padding-top: 5px; 
}
.tagline { color: #00f0ff; font-weight: bold; letter-spacing: 3px; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 5px; }
.main-title { font-size: 3.5rem; font-weight: 900; margin-bottom: 10px; line-height: 1.1; display: flex; flex-direction: column; align-items: center; } 
.text-glow { color: #fff; text-shadow: 0 0 20px rgba(0, 240, 255, 0.8), 0 0 40px rgba(0, 240, 255, 0.4); }
.sub-text { font-size: 1.6rem; color: #cbd5e1; font-weight: 600; margin-top: 5px; }
.description { max-width: 700px; margin: 0 auto 20px; color: #94a3b8; font-size: 1.05rem; line-height: 1.5; } 

.action-group { display: flex; gap: 15px; justify-content: center; }
.cyber-btn { position: relative; padding: 10px 30px; font-size: 0.95rem; font-weight: bold; cursor: pointer; transition: all 0.3s; text-transform: uppercase; letter-spacing: 1px; border: none; outline: none; overflow: hidden; }
.cyber-btn.primary { background: linear-gradient(45deg, #0057ff, #00f0ff); color: #fff; box-shadow: 0 0 20px rgba(0, 240, 255, 0.4); border-radius: 4px; }
.cyber-btn.primary:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(0, 240, 255, 0.6); }
.cyber-btn.secondary { background: transparent; color: #00f0ff; border: 1px solid #00f0ff; border-radius: 4px; }
.cyber-btn.secondary:hover { background: rgba(0, 240, 255, 0.1); }

/* --- 2. 宏观大屏区域 --- */
.dashboard-section { 
  display: grid; 
  grid-template-columns: 1fr 1fr 1fr; 
  gap: 15px; 
  height: 260px; 
}
.glass-panel { 
  background: rgba(13, 20, 40, 0.5); 
  border: 1px solid rgba(0, 240, 255, 0.15); 
  backdrop-filter: blur(12px); 
  border-radius: 12px; 
  padding: 15px 20px; 
  box-shadow: inset 0 0 20px rgba(0,240,255,0.05); 
  display: flex; 
  flex-direction: column;
  overflow: hidden; 
}
.panel-title { 
  font-size: 0.95rem; color: #fff; margin-bottom: 10px; 
  border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; 
  display: flex; align-items: center; 
}
.panel-title .icon { margin-right: 6px; }

/* 左侧指标 */
.stat-content { display: flex; flex-direction: column; justify-content: space-between; flex: 1; padding: 5px 0;} 
.stat-box { background: rgba(0,0,0,0.3); padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00f0ff; }
.stat-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 2px; }
.stat-value { font-size: 1.6rem; font-weight: 900; font-family: monospace; line-height: 1; }
.stat-value .unit { font-size: 0.85rem; font-weight: normal; margin-left: 3px; opacity: 0.7;}

/* 中间玫瑰图与雷达待命态 */
.echarts-container { flex: 1; width: 100%; height: 100%; }

/* 雷达扫描动画 */
.radar-standby { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; }
.radar-scope { width: 100px; height: 100px; border-radius: 50%; border: 2px solid rgba(0, 240, 255, 0.3); position: relative; overflow: hidden; background: rgba(0, 240, 255, 0.05); box-shadow: 0 0 20px rgba(0, 240, 255, 0.1); margin-bottom: 10px;}
.radar-grid { position: absolute; inset: 0; background-image: radial-gradient(circle, rgba(0, 240, 255, 0.3) 1px, transparent 1px); background-size: 15px 15px; border-radius: 50%; }
.radar-sweep { position: absolute; top: 50%; left: 50%; width: 50%; height: 50%; transform-origin: top left; background: linear-gradient(45deg, rgba(0, 240, 255, 0.8) 0%, transparent 60%); animation: sweep 3s linear infinite; }
.radar-dot { position: absolute; width: 5px; height: 5px; background: #00ff88; border-radius: 50%; box-shadow: 0 0 6px #00ff88; opacity: 0; }
.dot-1 { top: 30%; left: 60%; animation: ping 3s infinite 0.5s; }
.dot-2 { top: 70%; left: 40%; animation: ping 3s infinite 2s; }
.standby-text { text-align: center; color: #00f0ff; font-weight: bold; letter-spacing: 1px; line-height: 1.3; font-size: 0.85rem;}
.standby-sub { font-size: 0.7rem; color: #64748b; font-weight: normal; }

@keyframes sweep { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes ping { 0%, 100% { opacity: 0; transform: scale(0.5); } 50% { opacity: 1; transform: scale(1.5); } }

/* 右侧播报流与监听态 */
.feed-wrapper { flex: 1; overflow: hidden; position: relative; mask-image: linear-gradient(to bottom, transparent, black 10%, black 90%, transparent); -webkit-mask-image: linear-gradient(to bottom, transparent, black 10%, black 90%, transparent); }
.listening-state { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #4dabf7; font-style: italic; gap: 8px;}
.pulse-ring { width: 20px; height: 20px; border-radius: 50%; border: 2px solid #4dabf7; animation: pulse 2s infinite; }
@keyframes pulse { 0% { transform: scale(0.8); opacity: 1; } 100% { transform: scale(2); opacity: 0; } }

.feed-list { display: flex; flex-direction: column; gap: 8px; }
.scroll-anim { animation: scrollFeed 15s linear infinite; }
.scroll-anim:hover { animation-play-state: paused; }
.feed-item { font-size: 0.8rem; background: rgba(255,255,255,0.03); padding: 6px 8px; border-radius: 4px; display: flex; align-items: center; gap: 6px; border: 1px solid rgba(255,255,255,0.02); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.feed-time { color: #64748b; font-family: monospace; }
.feed-engine { color: #a855f7; font-weight: bold; }
.feed-mode { flex: 1; font-weight: bold; }
.feed-conf { color: #e2e8f0; background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 3px; font-size: 0.7rem;}

@keyframes scrollFeed { 0% { transform: translateY(0); } 100% { transform: translateY(calc(-50% - 4px)); } }

.text-cyan { color: #00f0ff; text-shadow: 0 0 10px rgba(0,240,255,0.5); }
.text-green { color: #00ff88; text-shadow: 0 0 10px rgba(0,255,136,0.5); }
.text-gold { color: #ffb800; text-shadow: 0 0 10px rgba(255,184,0,0.5); }
.text-red { color: #ff6b6b; text-shadow: 0 0 10px rgba(255,107,107,0.5); }

/* --- 3. 底部引擎矩阵 --- */
.engine-matrix { 
  display: grid; 
  grid-template-columns: repeat(4, 1fr); 
  gap: 15px; 
}
.engine-card { 
  background: rgba(13, 20, 40, 0.4); 
  border: 1px solid rgba(255,255,255,0.1); 
  border-radius: 12px; 
  padding: 20px 15px; 
  text-align: center; 
  transition: all 0.3s; 
  position: relative; 
  overflow: hidden; 
}
.engine-card:hover { 
  transform: translateY(-5px); 
  border-color: #00f0ff; 
  box-shadow: 0 10px 20px rgba(0, 240, 255, 0.15), inset 0 0 15px rgba(0, 240, 255, 0.05); 
}
.engine-icon { font-size: 2rem; margin-bottom: 10px; filter: drop-shadow(0 0 8px rgba(255,255,255,0.3)); }
.engine-name { font-size: 1.05rem; font-weight: 600; color: #fff; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; }
.engine-desc { font-size: 0.8rem; color: #cbd5e1; line-height: 1.4; margin-bottom: 15px; min-height: 35px; }
.engine-metric { display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 6px; }
.metric-label { font-size: 0.8rem; color: #94a3b8; font-weight: bold; }
.metric-value { font-size: 1.05rem; color: #00f0ff; font-weight: bold; font-family: monospace; }

@media (max-width: 1024px) {
  .dashboard-section { grid-template-columns: 1fr; height: auto; }
  .engine-matrix { grid-template-columns: repeat(2, 1fr); }
}
</style>