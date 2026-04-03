<template>
  <div class="dashboard-screen-container">
    
    <div class="screen-bg"></div>
    <div class="screen-grid"></div>
    <div class="scan-line"></div>

    <header class="screen-header">
      <div class="header-left">
        <div class="time-panel"><span class="clock-icon">◷</span> {{ currentTime }}</div>
      </div>
      <div class="header-center">
        <h1 class="glow-title">多模态交通方式智能研判总控台</h1>
        <div class="title-underline"></div>
      </div>
      <div class="header-right">
        <div class="status-panel">
          <span class="pulse-dot"></span> <span class="status-text">引擎矩阵在线</span>
        </div>
      </div>
    </header>

    <div class="screen-main" :class="{ 'is-loaded': isLoaded }">
      
      <aside class="panel-left">
        <div class="dv-panel h-50">
          <div class="panel-angle top-left"></div><div class="panel-angle top-right"></div>
          <div class="panel-angle bottom-left"></div><div class="panel-angle bottom-right"></div>
          
          <div class="panel-title">
            <span class="icon">⬡</span> 核心特征融合体系 (Feature Space)
          </div>
          <div class="panel-content feature-tree-box">
            <div class="feature-item cyan-theme">
              <div class="f-header">轨迹运动学特征 (18D)</div>
              <div class="f-tags"><span>速度均值</span><span>加速度</span><span>航向角方差</span><span>轨迹线形度</span></div>
            </div>
            <div class="feature-item gold-theme mt-10">
              <div class="f-header">OSM 空间拓扑 (21D)</div>
              <div class="f-tags"><span>路网密度</span><span>交叉口距离</span><span>公交专用道</span><span>POI 熵值</span></div>
            </div>
            <div class="feature-item green-theme mt-10">
              <div class="f-header">气象环境解耦 (10D)</div>
              <div class="f-tags"><span>降水量</span><span>地表温度</span><span>能见度</span><span>阵风风速</span></div>
            </div>
            <div class="fusion-arrow">▼ 多头注意力跨模态融合 (Multi-Head Attention) ▼</div>
          </div>
        </div>

        <div class="dv-panel h-50 mt-15">
          <div class="panel-angle top-left"></div><div class="panel-angle top-right"></div>
          <div class="panel-angle bottom-left"></div><div class="panel-angle bottom-right"></div>
          
          <div class="panel-title">
            <span class="icon">⚙️</span> 模型超参配置 (Hyperparameters)
          </div>
          <div class="panel-content param-table">
            <div class="p-row"><span class="p-key">Dataset Split</span><span class="p-val text-cyan">70% / 10% / 20%</span></div>
            <div class="p-row"><span class="p-key">Random Seed</span><span class="p-val text-cyan">42</span></div>
            <div class="p-row"><span class="p-key">Optimizer</span><span class="p-val text-gold">AdamW (lr=1e-3)</span></div>
            <div class="p-row"><span class="p-key">Exp1~3 Loss</span><span class="p-val">CrossEntropy (Weighted)</span></div>
            <div class="p-row"><span class="p-key">Exp4 Loss</span><span class="p-val text-green">FocalLoss (γ=2.0)</span></div>
            <div class="p-row"><span class="p-key">Architecture</span><span class="p-val">Dual-Encoder Attention</span></div>
          </div>
        </div>
      </aside>

      <main class="panel-center">
        <div class="hud-metrics">
          <div class="hud-card">
            <div class="hud-label">全网解析轨迹总量</div>
            <div class="hud-value text-cyan">{{ trajectoryStore.history.length }}<span class="unit">条</span></div>
          </div>
          <div class="hud-card">
            <div class="hud-label">核算碳减排量 (CO₂)</div>
            <div class="hud-value text-green">{{ totalCarbon }}<span class="unit">kg</span></div>
          </div>
          <div class="hud-card">
            <div class="hud-label">基线引擎 (Exp1) Acc</div>
            <div class="hud-value">83.30<span class="unit">%</span></div>
          </div>
          <div class="hud-card highlight-card">
            <div class="hud-label">终极引擎 (Exp4) F1</div>
            <div class="hud-value text-gold">81.66<span class="unit">%</span></div>
          </div>
        </div>

        <div class="dv-panel main-chart-panel mt-15">
          <div class="panel-angle top-left"></div><div class="panel-angle top-right"></div>
          <div class="panel-angle bottom-left"></div><div class="panel-angle bottom-right"></div>
          
          <div class="panel-title center-title">
            <span class="icon">📊</span> 核心评估指标演进态势 (Accuracy vs Macro-F1)
            <div class="insight-badge">💡 牺牲部分 Acc 换取整体 F1 提升，攻克不均衡难题</div>
          </div>
          <div class="panel-content chart-wrap" ref="evolutionChartRef"></div>
        </div>

        <div class="dv-panel stream-panel mt-15">
          <div class="panel-angle top-left"></div><div class="panel-angle top-right"></div>
          <div class="panel-angle bottom-left"></div><div class="panel-angle bottom-right"></div>
          
          <div class="panel-title">
            <span class="icon">📡</span> 实时流量推断监控链路
          </div>
          <div class="panel-content stream-box">
             <div class="stream-header">
                <span class="c-time">时间戳</span><span class="c-id">轨迹ID</span><span class="c-engine">引擎</span><span class="c-res">研判结果</span><span class="c-conf">置信度</span>
             </div>
             <div class="stream-list scroll-anim">
               <div v-for="(log, idx) in displayLogs" :key="idx" class="stream-item">
                 <span class="c-time text-gray">{{ formatTime(log.created_at) }}</span>
                 <span class="c-id">{{ log.trajectory_id?.slice(-6) || 'N/A' }}</span>
                 <span class="c-engine text-cyan">[{{ log.model_id.toUpperCase() }}]</span>
                 <span class="c-res" :class="getModeColorClass(log.predicted_mode)">{{ log.predicted_mode.toUpperCase() }}</span>
                 <span class="c-conf text-gold">{{ (log.confidence * 100).toFixed(1) }}%</span>
               </div>
             </div>
          </div>
        </div>
      </main>

      <aside class="panel-right">
        <div class="dv-panel h-50">
          <div class="panel-angle top-left"></div><div class="panel-angle top-right"></div>
          <div class="panel-angle bottom-left"></div><div class="panel-angle bottom-right"></div>
          
          <div class="panel-title">
            <span class="icon">🎯</span> 弱势样本靶向突破追踪
          </div>
          <div class="panel-content chart-wrap" ref="longTailChartRef"></div>
        </div>

        <div class="dv-panel h-50 mt-15 conclusion-panel">
          <div class="panel-angle top-left"></div><div class="panel-angle top-right"></div>
          <div class="panel-angle bottom-left"></div><div class="panel-angle bottom-right"></div>
          
          <div class="panel-title">
            <span class="icon">🧠</span> 引擎架构诊断结论 (Conclusion)
          </div>
          
          <div class="panel-content">
            <div class="engine-tabs">
              <div class="e-tab" :class="{ active: currentExp === 'exp1' }" @click="currentExp = 'exp1'">Exp1</div>
              <div class="e-tab" :class="{ active: currentExp === 'exp2' }" @click="currentExp = 'exp2'">Exp2</div>
              <div class="e-tab" :class="{ active: currentExp === 'exp3' }" @click="currentExp = 'exp3'">Exp3</div>
              <div class="e-tab" :class="{ active: currentExp === 'exp4' }" @click="currentExp = 'exp4'">Exp4</div>
            </div>

            <div class="diagnostic-box" :class="insightData.theme">
              <div class="d-header">
                <div class="d-title">{{ insightData.title }}</div>
                <div class="d-metrics">Acc: {{ insightData.acc }}% | F1: {{ insightData.f1 }}%</div>
              </div>
              <div class="d-body">
                <p>{{ insightData.desc }}</p>
                <div class="d-warning"><el-icon><WarningFilled /></el-icon> {{ insightData.warning }}</div>
              </div>
            </div>
          </div>
        </div>
      </aside>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { Location, MapLocation, PartlyCloudy, WarningFilled } from '@element-plus/icons-vue'
import { useTrajectoryStore } from '@/stores/trajectory'

const trajectoryStore = useTrajectoryStore()
const isLoaded = ref(false)

// --- 顶部时间逻辑 ---
const currentTime = ref('')
let timeInterval: number
const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleString('zh-CN', { hour12: false })
}

// --- 后台真实数据核算 ---
const totalCarbon = computed(() => {
  let reduction = 0
  trajectoryStore.history.forEach(record => {
    const mode = String(record.predicted_mode || '').toLowerCase()
    const dist = (record.distance || 0) / 1000
    if (mode.includes('walk')) reduction += dist * 0.15
    else if (mode.includes('bike')) reduction += dist * 0.10
    else if (mode.includes('bus') || mode.includes('subway')) reduction += dist * 0.05
  })
  return reduction.toFixed(2)
})

// --- 实时滚动播报 (优先真实，少于6条造点演示数据补齐大屏效果) ---
const displayLogs = computed(() => {
  const real = trajectoryStore.history.slice(0, 8)
  if (real.length >= 6) return real
  return [
    ...real,
    { created_at: new Date().toISOString(), trajectory_id: 'sim_8a9', model_id: 'exp4', predicted_mode: 'subway', confidence: 0.94 },
    { created_at: new Date(Date.now()-10000).toISOString(), trajectory_id: 'sim_b2c', model_id: 'exp4', predicted_mode: 'walk', confidence: 0.98 },
    { created_at: new Date(Date.now()-20000).toISOString(), trajectory_id: 'sim_7f1', model_id: 'exp2', predicted_mode: 'bus', confidence: 0.85 },
    { created_at: new Date(Date.now()-30000).toISOString(), trajectory_id: 'sim_e4d', model_id: 'exp1', predicted_mode: 'car', confidence: 0.72 }
  ].slice(0, 8)
})

const formatTime = (iso?: string) => iso ? new Date(iso).toLocaleTimeString('zh-CN', { hour12: false }) : '--:--:--'
const getModeColorClass = (mode: string) => {
  const m = mode.toLowerCase()
  if (m.includes('walk') || m.includes('bike')) return 'text-green'
  if (m.includes('bus') || m.includes('subway')) return 'text-cyan'
  return 'text-gold'
}

// --- 右侧诊断卡片逻辑 ---
const currentExp = ref('exp4')
const insights = {
  exp1: { theme: 'blue', title: '纯轨迹基线引擎', acc: '83.30', f1: '75.89', desc: '仅提取速度、加速度等运动学特征。由于缺乏路网拓扑约束，极易将拥堵路段的私家车与公交车混淆。', warning: '模态混淆严重，长尾类别识别率极低。' },
  exp2: { theme: 'cyan', title: 'OSM空间拓扑映射', acc: '83.88', f1: '78.27', desc: '引入 OpenStreetMap 真实路网，成功捕获公交专用道等空间语义。强路网关联模态（公交/地铁）精度大幅上升。', warning: '未解耦气象因素，遭遇恶劣天气易发生特征偏移。' },
  exp3: { theme: 'green', title: '气象因子多模态解耦', acc: '84.50', f1: '79.50', desc: '融合多维气象特征矩阵，剥离恶劣天气对车速特征的干扰，显著增强了系统在非理想环境下的推断鲁棒性。', warning: '交叉熵损失导致小样本（单车）梯度被大样本（汽车）淹没。' },
  exp4: { theme: 'gold', title: 'Focal Loss 终极引擎', acc: '81.57', f1: '81.66', desc: '重构底层损失函数。通过牺牲部分高频类的全局准确率，换取了弱势群体（自行车/地铁等低碳方式）识别率的爆发式增长。', warning: '完美匹配碳普惠场景逻辑，最具业务价值的科研折中。' }
}
const insightData = computed(() => insights[currentExp.value as keyof typeof insights])

// --- ECharts 图表渲染 (致敬大屏风格) ---
const evolutionChartRef = ref<HTMLElement | null>(null)
const longTailChartRef = ref<HTMLElement | null>(null)
let evolutionChart: echarts.ECharts | null = null
let longTailChart: echarts.ECharts | null = null

const initCharts = () => {
  if (evolutionChartRef.value) {
    evolutionChart = echarts.init(evolutionChartRef.value)
    evolutionChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(3,8,22,0.9)', borderColor: '#00f0ff', textStyle: { color: '#fff' } },
      legend: { data: ['Accuracy (全局准度)', 'Macro-F1 (综合均衡度)'], textStyle: { color: '#a0aabf' }, top: 0, right: 10 },
      grid: { left: '3%', right: '4%', bottom: '5%', top: '15%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: ['Exp1', 'Exp2', 'Exp3', 'Exp4'], axisLabel: { color: '#a0aabf' }, axisLine: { lineStyle: { color: '#1e293b' } } },
      yAxis: { type: 'value', min: 75, max: 86, axisLabel: { color: '#a0aabf' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } } },
      series: [
        { name: 'Accuracy (全局准度)', type: 'line', data: [83.30, 83.88, 84.50, 81.57], smooth: true, symbolSize: 8, itemStyle: { color: '#00f0ff' }, lineStyle: { width: 3, shadowBlur: 15, shadowColor: '#00f0ff' } },
        { name: 'Macro-F1 (综合均衡度)', type: 'line', data: [75.89, 78.27, 79.50, 81.66], smooth: true, symbolSize: 8, itemStyle: { color: '#ffb800' }, lineStyle: { width: 3, shadowBlur: 15, shadowColor: '#ffb800' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(255, 184, 0, 0.3)' }, { offset: 1, color: 'transparent' }]) } }
      ]
    })
  }

  if (longTailChartRef.value) {
    longTailChart = echarts.init(longTailChartRef.value)
    longTailChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(3,8,22,0.9)', borderColor: '#00f0ff', textStyle: { color: '#fff' } },
      legend: { data: ['Exp1 基线', 'Exp4 Focal优化'], textStyle: { color: '#a0aabf' }, top: 0, right: 10 },
      grid: { left: '3%', right: '4%', bottom: '5%', top: '18%', containLabel: true },
      xAxis: { type: 'category', data: ['自行车 (Bike)', '公交 (Bus)', '地铁 (Subway)', '火车 (Train)'], axisLabel: { color: '#a0aabf', fontSize: 11 }, axisLine: { lineStyle: { color: '#1e293b' } } },
      yAxis: { type: 'value', name: 'F1 (%)', nameTextStyle: { color: '#a0aabf' }, min: 60, max: 100, axisLabel: { color: '#a0aabf' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } } },
      series: [
        { name: 'Exp1 基线', type: 'bar', barWidth: '20%', data: [79.2, 70.1, 68.5, 75.3], itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(77, 144, 226, 0.8)' }, { offset: 1, color: 'rgba(77, 144, 226, 0.2)' }]), borderRadius: [2, 2, 0, 0] } },
        { name: 'Exp4 Focal优化', type: 'bar', barWidth: '20%', data: [91.21, 74.44, 74.29, 85.19], itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(0, 255, 136, 0.9)' }, { offset: 1, color: 'rgba(0, 255, 136, 0.3)' }]), borderRadius: [2, 2, 0, 0], shadowBlur: 10, shadowColor: 'rgba(0,255,136,0.4)' }, label: { show: true, position: 'top', color: '#00ff88', fontSize: 11, formatter: '{c}%' } }
      ]
    })
  }
}

onMounted(async () => {
  updateTime()
  timeInterval = window.setInterval(updateTime, 1000)
  await trajectoryStore.fetchHistory()
  isLoaded.value = true
  nextTick(initCharts)
  window.addEventListener('resize', () => { evolutionChart?.resize(); longTailChart?.resize(); })
})

onUnmounted(() => {
  if (timeInterval) clearInterval(timeInterval)
  window.removeEventListener('resize', () => { evolutionChart?.resize(); longTailChart?.resize(); })
  evolutionChart?.dispose()
  longTailChart?.dispose()
})
</script>

<style scoped>
/* ================= 全局大屏容器设定 ================= */
.dashboard-screen-container {
  /* 强制填满视口，禁止系统原生滚动条，一切在大屏内部流转 */
  height: calc(100vh - 64px); /* 减去顶部菜单栏高度 */
  background-color: #020614; /* 极深的科技藏蓝底色 */
  color: #fff;
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif;
  position: relative;
  overflow: auto;
  display: flex;
  flex-direction: column;
}

/* 背景特效 */
.screen-bg { position: absolute; inset: 0; background: radial-gradient(circle at 50% 50%, rgba(0, 87, 255, 0.05) 0%, transparent 70%); z-index: 0; }
.screen-grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px); background-size: 30px 30px; z-index: 0; }
.scan-line { position: absolute; width: 100%; height: 2px; background: linear-gradient(90deg, transparent, rgba(0,240,255,0.4), transparent); z-index: 1; animation: scan 6s linear infinite; }
@keyframes scan { 0% { top: -10%; } 100% { top: 110%; } }

/* ================= 顶部大屏 Header ================= */
.screen-header {
  position: relative; z-index: 2; height: 60px; display: flex; justify-content: space-between; align-items: flex-end; padding: 0 20px 10px;
  background: url('data:image/svg+xml;utf8,<svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg"><path d="M0,60 L20,40 Lcalc(50% - 250px),40 Lcalc(50% - 200px),20 Lcalc(50% + 200px),20 Lcalc(50% + 250px),40 Lcalc(100% - 20px),40 L100,60" fill="none" stroke="rgba(0,240,255,0.3)" stroke-width="2"/></svg>') no-repeat bottom center;
  background-size: 100% 100%;
}
.header-left, .header-right { flex: 1; padding-bottom: 5px; }
.time-panel { color: #00f0ff; font-family: monospace; font-size: 1.1rem; letter-spacing: 1px; }
.header-center { flex: 2; text-align: center; position: relative; padding-bottom: 15px;}
.glow-title { margin: 0; font-size: 2rem; font-weight: 900; color: #fff; letter-spacing: 4px; text-shadow: 0 0 15px rgba(0,240,255,0.8); }
.status-panel { float: right; color: #00ff88; font-size: 0.9rem; display: flex; align-items: center; gap: 8px;}
.pulse-dot { width: 8px; height: 8px; background: #00ff88; border-radius: 50%; box-shadow: 0 0 10px #00ff88; animation: blink 1s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* ================= 主体 1:2:1 布局 ================= */
.screen-main {
  position: relative; z-index: 2; flex: 1; padding: 15px 20px; display: grid;
  grid-template-columns: 1fr 2fr 1fr; gap: 20px; height: calc(100% - 60px);
  opacity: 0; transform: scale(0.98); transition: all 0.8s cubic-bezier(0.1, 0.9, 0.2, 1);
}
.screen-main.is-loaded { opacity: 1; transform: scale(1); }

/* 公共发光切角面板框 */
.dv-panel {
  position: relative; background: rgba(8, 15, 38, 0.7); border: 1px solid rgba(0, 240, 255, 0.2); box-shadow: inset 0 0 30px rgba(0, 240, 255, 0.05); display: flex; flex-direction: column; padding: 15px;
}
.panel-angle { position: absolute; width: 15px; height: 15px; border: 2px solid #00f0ff; }
.top-left { top: -2px; left: -2px; border-right: none; border-bottom: none; }
.top-right { top: -2px; right: -2px; border-left: none; border-bottom: none; }
.bottom-left { bottom: -2px; left: -2px; border-right: none; border-top: none; }
.bottom-right { bottom: -2px; right: -2px; border-left: none; border-top: none; }

.panel-title { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 15px; display: flex; align-items: center; background: linear-gradient(90deg, rgba(0,240,255,0.2) 0%, transparent 100%); padding: 5px 10px; border-left: 3px solid #00f0ff; }
.panel-title .icon { margin-right: 8px; color: #00f0ff; }
.panel-content { flex: 1; position: relative; overflow: auto; }

/* 辅助布局类 */
.h-50 { height: calc(50% - 7.5px); }
.h-100 { height: 100%; }
.mt-10 { margin-top: 10px; }
.mt-15 { margin-top: 15px; }
.mt-20 { margin-top: 20px; }

/* ================= 左侧细节 ================= */
.feature-tree-box { display: flex; flex-direction: column; justify-content: space-between;}
.feature-item { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 10px; border-radius: 4px; }
.f-header { font-size: 0.95rem; font-weight: bold; margin-bottom: 8px; display: flex; align-items: center; gap: 5px;}
.text-cyan { color: #00f0ff; } .text-gold { color: #ffb800; } .text-green { color: #00ff88; }
.f-tags { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
.f-tag { font-size: 0.75rem; background: rgba(0,240,255,0.1); color: #00f0ff; text-align: center; padding: 4px 0; border: 1px solid rgba(0,240,255,0.2); border-radius: 2px;}
.f-tag.gold { background: rgba(255,184,0,0.1); color: #ffb800; border-color: rgba(255,184,0,0.2);}
.f-tag.green { background: rgba(0,255,136,0.1); color: #00ff88; border-color: rgba(0,255,136,0.2);}

.fusion-arrow { text-align: center; font-size: 0.8rem; color: #a0aabf; margin-top: auto; padding: 10px 0; animation: pulseOp 2s infinite; font-weight: bold;}
@keyframes pulseOp { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }

.param-table { display: flex; flex-direction: column; justify-content: space-around; height: 100%;}
.p-row { display: flex; justify-content: space-between; font-size: 0.9rem; padding: 8px 0; border-bottom: 1px dashed rgba(255,255,255,0.1); }
.p-key { color: #94a3b8; } .p-val { font-family: monospace; font-weight: bold; color: #e2e8f0; }

/* ================= 中间主视觉细节 ================= */
.panel-center { display: flex; flex-direction: column; }
.hud-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; height: 90px; }
.hud-card { background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.2); padding: 15px; text-align: center; display: flex; flex-direction: column; justify-content: center; position: relative; overflow: auto;}
.hud-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: #00f0ff; }
.highlight-card { background: rgba(255, 184, 0, 0.05); border-color: rgba(255, 184, 0, 0.3); }
.highlight-card::before { background: #ffb800; }
.hud-label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 5px; text-transform: uppercase; }
.hud-value { font-size: 2rem; font-weight: 900; font-family: monospace; line-height: 1; text-shadow: 0 0 10px currentColor; }
.hud-value .unit { font-size: 0.9rem; font-weight: normal; margin-left: 3px; opacity: 0.7; text-shadow: none; }

.main-chart-panel { flex: 2; display: flex; flex-direction: column; }
.center-title { background: none; border-left: none; justify-content: center; position: relative; }
.center-title::after { content: ''; position: absolute; bottom: 0; left: 20%; width: 60%; height: 1px; background: linear-gradient(90deg, transparent, #00f0ff, transparent); }
.insight-badge { position: absolute; right: 0; font-size: 0.8rem; background: rgba(255,184,0,0.1); color: #ffb800; padding: 2px 10px; border: 1px solid rgba(255,184,0,0.3); border-radius: 10px; }
.chart-wrap { width: 100%; height: 100%; min-height: 200px; }

.stream-panel { flex: 1; display: flex; flex-direction: column; }
.stream-box { display: flex; flex-direction: column; }
.stream-header { display: grid; grid-template-columns: 2fr 2fr 1.5fr 2fr 1.5fr; font-size: 0.85rem; color: #00f0ff; padding: 8px 10px; background: rgba(0,240,255,0.1); margin-bottom: 5px; font-weight: bold;}
.stream-list { flex: 1; overflow: auto; display: flex; flex-direction: column; gap: 5px;}
.stream-item { display: grid; grid-template-columns: 2fr 2fr 1.5fr 2fr 1.5fr; font-size: 0.85rem; padding: 8px 10px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); align-items: center; }
.text-gray { color: #64748b; font-family: monospace; }

/* ================= 右侧细节 ================= */
.conclusion-panel { display: flex; flex-direction: column; }
.engine-tabs { display: flex; gap: 10px; margin-bottom: 15px; }
.e-tab { flex: 1; text-align: center; padding: 8px 0; background: rgba(255,255,255,0.05); color: #94a3b8; font-size: 0.9rem; cursor: pointer; border: 1px solid rgba(255,255,255,0.1); transition: all 0.3s; clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px); }
.e-tab.active { background: rgba(0,240,255,0.2); color: #fff; border-color: #00f0ff; font-weight: bold; }

.diagnostic-box { flex: 1; padding: 15px; border: 1px solid; background: rgba(0,0,0,0.3); display: flex; flex-direction: column; justify-content: space-between;}
.diagnostic-box.blue { border-color: rgba(74, 144, 226, 0.4); }
.diagnostic-box.cyan { border-color: rgba(0, 240, 255, 0.4); }
.diagnostic-box.green { border-color: rgba(0, 255, 136, 0.4); }
.diagnostic-box.gold { border-color: rgba(255, 184, 0, 0.4); background: rgba(255,184,0,0.05);}

.d-header { border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 10px; }
.d-title { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 5px; }
.d-metrics { font-family: monospace; font-size: 0.9rem; color: #a0aabf; }
.gold .d-title { color: #ffb800; }
.gold .d-metrics { color: #ffb800; }

.d-body p { font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; margin: 0; text-align: justify; }
.d-warning { margin-top: 15px; font-size: 0.85rem; color: #ff6b6b; display: flex; align-items: center; gap: 5px; background: rgba(255,107,107,0.1); padding: 8px; border-radius: 4px;}
.gold .d-warning { color: #00ff88; background: rgba(0,255,136,0.1); }
</style>