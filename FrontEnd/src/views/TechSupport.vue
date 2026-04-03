<template>
  <div class="grand-screen-container">
    <div class="grand-bg"></div>
    <div class="grand-grid"></div>
    <div class="grand-light-beam"></div>

    <header class="grand-header">
      <div class="header-flank left-flank">
        <div class="h-dec-line"></div>
        <div class="h-time"><span class="icon">◷</span> ENG-OPS // {{ currentTime }}</div>
      </div>
      
      <div class="header-core">
        <div class="core-bg">
          <h1 class="core-title">核心工程架构与系统鲁棒性压测</h1>
        </div>
        <div class="core-glow"></div>
      </div>

      <div class="header-flank right-flank">
        <div class="h-status"><span class="pulse-dot"></span> Telemetry Online</div>
        <div class="h-dec-line"></div>
      </div>
    </header>

    <div class="screen-main" :class="{ 'is-loaded': isLoaded }">
      
      <div class="hud-matrix">
        <div class="hud-block">
          <div class="hud-title">FastAPI 并发吞吐 (QPS)</div>
          <div class="hud-num text-cyan">1,250<span class="unit">req/s</span></div>
        </div>
        <div class="hud-block">
          <div class="hud-title">PyTorch 显存占用</div>
          <div class="hud-num text-green">4.2<span class="unit">GB</span></div>
        </div>
        <div class="hud-block">
          <div class="hud-title">平均推断耗时</div>
          <div class="hud-num text-blue">18.5<span class="unit">ms</span></div>
        </div>
        <div class="hud-block highlight-hud">
          <div class="hud-title">系统高可用 SLA</div>
          <div class="hud-num text-gold">99.99<span class="unit">%</span></div>
        </div>
      </div>

      <div class="main-grid mt-20">
        <aside class="panel-column left-column">
          <div class="heavy-panel h-60">
            <div class="corner tl"></div><div class="corner tr"></div>
            <div class="corner bl"></div><div class="corner br"></div>
            
            <div class="panel-header">
              <span class="p-icon">🔄</span> 系统端到端数据流拓扑
            </div>
            <div class="panel-body pipeline-visual">
               <div class="p-stage">
                 <div class="stage-name text-gray">L1: Data Ingestion</div>
                 <div class="stage-box box-gray">GPS/OSM/气象数据对齐</div>
               </div>
               <div class="p-arrow text-gray">▼</div>
               <div class="p-stage">
                 <div class="stage-name text-cyan">L2: PyTorch Engine</div>
                 <div class="stage-box box-cyan">多模态特征融合与推断</div>
               </div>
               <div class="p-arrow text-cyan">▼</div>
               <div class="p-stage">
                 <div class="stage-name text-gold">L3: FastAPI Service</div>
                 <div class="stage-box box-gold">异步调度 & 星火 LLM</div>
               </div>
               <div class="p-arrow text-gold">▼</div>
               <div class="p-stage">
                 <div class="stage-name text-green">L4: Vue3 Dashboard</div>
                 <div class="stage-box box-green">数字孪生与碳普惠展现</div>
               </div>
            </div>
          </div>

          <div class="heavy-panel h-40 mt-20">
            <div class="corner tl"></div><div class="corner tr"></div>
            <div class="corner bl"></div><div class="corner br"></div>
            <div class="panel-header">
              <span class="p-icon">🛠️</span> 核心引擎驱动栈
            </div>
            <div class="panel-body tech-grid">
              <div class="t-badge"><span style="color:#ee4c2c">🔥</span> PyTorch</div>
              <div class="t-badge"><span style="color:#009688">⚡</span> FastAPI</div>
              <div class="t-badge"><span style="color:#4fc08d">🟢</span> Vue 3</div>
              <div class="t-badge"><span style="color:#e44d26">📊</span> ECharts</div>
              <div class="t-badge"><span style="color:#2196f3">🗺️</span> Leaflet</div>
              <div class="t-badge"><span style="color:#a855f7">💬</span> Spark LLM</div>
            </div>
          </div>
        </aside>

        <main class="panel-column center-column">
          <div class="heavy-panel h-100">
            <div class="corner tl"></div><div class="corner tr"></div>
            <div class="corner bl"></div><div class="corner br"></div>
            
            <div class="panel-header justify-center">
              <span class="p-icon text-warning">🛡️</span> 极端环境系统鲁棒性压力测试 (Robustness Stress Test)
            </div>
            <div class="insight-bar text-warning">
              💡 <strong>压测结论：</strong> 面对特大暴雨与跨城迁移，Exp4 架构凭借气象解耦与动态 Focal 权重，展现出碾压基线模型的抗干扰能力！
            </div>
            <div class="panel-body chart-wrap mt-10" ref="robustnessChartRef"></div>
          </div>
        </main>

        <aside class="panel-column right-column">
          <div class="heavy-panel h-100">
            <div class="corner tl"></div><div class="corner tr"></div>
            <div class="corner bl"></div><div class="corner br"></div>
            
            <div class="panel-header">
              <span class="p-icon">📡</span> API 实时推断延迟遥测 (Live Telemetry)
            </div>
            <div class="telemetry-status">
               <span>当前节点: Beijing-Zone-A</span>
               <span class="text-green blink-text">● CONNECTION SECURE</span>
            </div>
            <div class="panel-body chart-wrap" ref="latencyChartRef"></div>
          </div>
        </aside>

      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import * as echarts from 'echarts'

const isLoaded = ref(false)

// --- 时间逻辑 ---
const currentTime = ref('')
let timeInterval: number
const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleString('zh-CN', { hour12: false })
}

// --- ECharts 图表渲染 ---
const robustnessChartRef = ref<HTMLElement | null>(null)
const latencyChartRef = ref<HTMLElement | null>(null)
let robustnessChart: echarts.ECharts | null = null
let latencyChart: echarts.ECharts | null = null

// 生成动态延迟数据
let latencyData: number[] = Array.from({length: 20}, () => 15 + Math.random() * 5)
let timeAxis: string[] = Array.from({length: 20}, (_, i) => `T-${20-i}s`)

const initCharts = () => {
  // 1. 极端环境鲁棒性压测图 (代替文字 Q&A！)
  if (robustnessChartRef.value) {
    robustnessChart = echarts.init(robustnessChartRef.value)
    robustnessChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(5, 12, 30, 0.9)', borderColor: '#E6A23C', textStyle: { color: '#fff' } },
      legend: { data: ['Exp1 (基线模型)', 'Exp4 (终极架构)'], textStyle: { color: '#94a3b8', fontSize: 13 }, bottom: 0 },
      grid: { left: '3%', right: '4%', bottom: '12%', top: '10%', containLabel: true },
      xAxis: { 
        type: 'category', 
        data: ['常规标准环境\n(Standard)', '极端暴雨干扰\n(Heavy Rain)', '跨城空间迁移\n(Cross-City)'], 
        axisLabel: { color: '#94a3b8', fontSize: 13, fontWeight: 'bold', margin: 15 }, 
        axisLine: { lineStyle: { color: '#1e293b', width: 2 } } 
      },
      yAxis: { type: 'value', name: 'Macro-F1 (%)', nameTextStyle: { color: '#94a3b8' }, min: 40, max: 90, axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } } },
      series: [
        { 
          name: 'Exp1 (基线模型)', type: 'bar', barWidth: '25%', barGap: '15%',
          data: [75.8, 45.2, 50.1], // 基线在极端环境下彻底崩溃
          itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(56, 189, 248, 0.8)' }, { offset: 1, color: 'rgba(56, 189, 248, 0.2)' }]), borderRadius: [4, 4, 0, 0] },
          label: { show: true, position: 'top', color: '#38bdf8', formatter: '{c}%' }
        },
        { 
          name: 'Exp4 (终极架构)', type: 'bar', barWidth: '25%',
          data: [81.6, 78.5, 75.2], // Exp4 非常坚挺
          itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(230, 162, 60, 0.9)' }, { offset: 1, color: 'rgba(230, 162, 60, 0.3)' }]), borderRadius: [4, 4, 0, 0], shadowBlur: 10, shadowColor: 'rgba(230,162,60,0.5)' }, 
          label: { show: true, position: 'top', color: '#E6A23C', fontSize: 13, fontWeight: 'bold', formatter: '{c}%' } 
        }
      ]
    })
  }

  // 2. API 延迟实时监控图
  if (latencyChartRef.value) {
    latencyChart = echarts.init(latencyChartRef.value)
    latencyChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(5, 12, 30, 0.9)', borderColor: '#00ff88', textStyle: { color: '#fff' } },
      grid: { left: '3%', right: '5%', bottom: '5%', top: '10%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: timeAxis, axisLabel: { show: false }, axisLine: { lineStyle: { color: '#1e293b' } }, splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.02)' } } },
      yAxis: { type: 'value', name: 'Latency (ms)', nameTextStyle: { color: '#94a3b8' }, min: 10, max: 30, axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } } },
      series: [
        { 
          name: '推断延迟', type: 'line', data: latencyData, smooth: true, symbol: 'none',
          itemStyle: { color: '#00ff88' }, lineStyle: { width: 2, shadowBlur: 8, shadowColor: 'rgba(0,255,136,0.8)' }, 
          areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(0, 255, 136, 0.3)' }, { offset: 1, color: 'transparent' }]) } 
        }
      ]
    })
  }
}

// 动态推入数据模拟监控
const updateTelemetry = () => {
  if (!latencyChart) return
  latencyData.shift()
  // 模拟偶尔的延迟波动
  const jump = Math.random() > 0.9 ? Math.random() * 8 : 0;
  latencyData.push(15 + Math.random() * 4 + jump)
  
  latencyChart.setOption({ series: [{ data: latencyData }] })
}

onMounted(() => {
  updateTime()
  timeInterval = window.setInterval(() => {
    updateTime();
    updateTelemetry();
  }, 1000)
  
  isLoaded.value = true
  nextTick(initCharts)
  
  window.addEventListener('resize', () => { robustnessChart?.resize(); latencyChart?.resize(); })
})

onUnmounted(() => {
  if (timeInterval) clearInterval(timeInterval)
  window.removeEventListener('resize', () => { robustnessChart?.resize(); latencyChart?.resize(); })
  robustnessChart?.dispose(); latencyChart?.dispose();
})
</script>

<style scoped>
/* ================= 全局重工业大屏基调 ================= */
.grand-screen-container {
  height: calc(100vh - 64px); 
  background-color: #010510;
  color: #fff;
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif;
  position: relative;
  overflow: auto;
  display: flex;
  flex-direction: column;
}

.grand-bg { position: absolute; inset: 0; background: radial-gradient(circle at 50% 40%, rgba(0, 102, 255, 0.1) 0%, transparent 60%); z-index: 0; }
.grand-grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px); background-size: 60px 60px; z-index: 0; perspective: 1000px; transform-style: preserve-3d; opacity: 0.6;}
.grand-light-beam { position: absolute; top: 0; left: 0; right: 0; height: 150px; background: linear-gradient(180deg, rgba(0, 240, 255, 0.05) 0%, transparent 100%); z-index: 0;}

/* ================= 1. 重装甲顶部主梁 ================= */
.grand-header {
  position: relative; z-index: 2; height: 80px; display: flex; justify-content: space-between; align-items: flex-start; padding: 0 20px;
}
.header-flank { flex: 1; display: flex; flex-direction: column; justify-content: center; height: 60px; padding-top: 15px;}
.h-time { color: #00f0ff; font-family: monospace; font-size: 1.1rem; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0,240,255,0.6); margin-top: 5px;}
.h-dec-line { height: 2px; background: linear-gradient(90deg, rgba(0,240,255,0.8), transparent); width: 100%; position: relative;}
.right-flank .h-dec-line { background: linear-gradient(270deg, rgba(0,240,255,0.8), transparent); }
.h-status { text-align: right; color: #00ff88; font-size: 1rem; margin-top: 5px; text-transform: uppercase; font-family: monospace; letter-spacing: 1px;}
.pulse-dot { display: inline-block; width: 10px; height: 10px; background: #00ff88; border-radius: 50%; box-shadow: 0 0 12px #00ff88; margin-right: 8px;}

.header-core { flex: 2.5; display: flex; justify-content: center; position: relative;}
.core-bg {
  background: rgba(4, 18, 48, 0.8);
  border: 2px solid rgba(0, 240, 255, 0.4);
  border-top: none;
  padding: 10px 80px 25px;
  clip-path: polygon(0 0, 100% 0, 92% 100%, 8% 100%);
  box-shadow: inset 0 -20px 40px rgba(0, 240, 255, 0.2);
  display: flex; align-items: center; justify-content: center;
  position: relative; z-index: 2;
}
.core-title { margin: 0; font-size: 2.2rem; font-weight: 900; color: #fff; letter-spacing: 5px; text-shadow: 0 0 20px rgba(0,240,255,0.8); }
.core-glow { position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 60%; height: 15px; background: #00f0ff; filter: blur(25px); z-index: 1; opacity: 0.5;}

/* ================= 2. 主体网格布局 ================= */
.screen-main {
  position: relative; z-index: 2; flex: 1; padding: 20px; display: flex; flex-direction: column;
  height: calc(100% - 80px); opacity: 0; transform: scale(0.98); transition: all 0.8s cubic-bezier(0.1, 0.9, 0.2, 1);
}
.screen-main.is-loaded { opacity: 1; transform: scale(1); }

/* HUD 指标阵列 */
.hud-matrix { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; height: 100px; flex-shrink: 0;}
.hud-block { background: rgba(4, 14, 38, 0.8); border: 1px solid rgba(0, 240, 255, 0.2); padding: 15px 25px; display: flex; flex-direction: column; justify-content: center; position: relative; box-shadow: 0 5px 15px rgba(0,0,0,0.5);}
.hud-block::before { content:''; position: absolute; left: 0; bottom: 0; width: 100%; height: 3px; background: linear-gradient(90deg, #00f0ff, transparent); }
.highlight-hud { background: rgba(255,184,0,0.05); border-color: rgba(255,184,0,0.3); }
.highlight-hud::before { background: linear-gradient(90deg, #ffb800, transparent); }
.hud-title { font-size: 0.95rem; color: #94a3b8; margin-bottom: 5px; text-transform: uppercase; font-weight: bold; letter-spacing: 1px;}
.hud-num { font-size: 2.8rem; font-weight: 900; font-family: 'Impact', monospace; line-height: 1; letter-spacing: 2px; text-shadow: 0 0 20px currentColor; }
.hud-num .unit { font-size: 1rem; font-weight: normal; margin-left: 6px; opacity: 0.6; text-shadow: none; font-family: 'Rajdhani'; letter-spacing: 0;}

/* 下方 1:2:1 主区域 */
.main-grid { display: grid; grid-template-columns: 1fr 2.2fr 1fr; gap: 25px; flex: 1; overflow: auto;}
.panel-column { display: flex; flex-direction: column; height: 100%;}

/* 军工级面板边框 */
.heavy-panel {
  position: relative; 
  background: rgba(6, 14, 33, 0.75); 
  border: 1px solid rgba(0, 240, 255, 0.15); 
  box-shadow: inset 0 0 30px rgba(0, 240, 255, 0.05), 0 5px 15px rgba(0,0,0,0.5);
  display: flex; flex-direction: column; padding: 15px 20px;
}
.corner { position: absolute; width: 15px; height: 15px; border: 2px solid #00f0ff; z-index: 5;}
.corner.tl { top: -2px; left: -2px; border-right: none; border-bottom: none; }
.corner.tr { top: -2px; right: -2px; border-left: none; border-bottom: none; }
.corner.bl { bottom: -2px; left: -2px; border-right: none; border-top: none; }
.corner.br { bottom: -2px; right: -2px; border-left: none; border-top: none; }

.panel-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 15px; display: flex; align-items: center; border-bottom: 1px solid rgba(0,240,255,0.2); padding-bottom: 8px; text-shadow: 0 0 10px rgba(255,255,255,0.3);}
.panel-header.justify-center { justify-content: center; border-bottom: none; position: relative;}
.panel-header.justify-center::after { content:''; position: absolute; bottom: 0; width: 60%; height: 1px; background: linear-gradient(90deg, transparent, #00f0ff, transparent);}
.p-icon { margin-right: 8px; color: #00f0ff; font-size: 1.2rem;}
.panel-body { flex: 1; position: relative; overflow: auto; width: 100%; height: 100%;}

.h-40 { height: calc(40% - 10px); } .h-60 { height: calc(60% - 10px); } .h-100 { height: 100%; }
.mt-10 { margin-top: 10px; } .mt-20 { margin-top: 20px; }

/* 左侧：架构视觉流 */
.pipeline-visual { display: flex; flex-direction: column; justify-content: space-around; padding: 10px 0;}
.p-stage { text-align: center; }
.stage-name { font-size: 0.85rem; font-weight: bold; font-family: monospace; margin-bottom: 4px; }
.stage-box { padding: 8px 15px; border: 1px solid; border-radius: 4px; font-weight: bold; font-size: 0.95rem; background: rgba(0,0,0,0.4); box-shadow: inset 0 0 10px currentColor;}
.box-gray { border-color: #64748b; color: #cbd5e1; }
.box-cyan { border-color: #00f0ff; color: #00f0ff; }
.box-gold { border-color: #ffb800; color: #ffb800; }
.box-green { border-color: #00ff88; color: #00ff88; }
.p-arrow { text-align: center; font-size: 1.2rem; animation: flow 1s infinite; }
@keyframes flow { 0%, 100% { opacity: 0.3; transform: translateY(-2px);} 50% { opacity: 1; transform: translateY(2px);} }

.tech-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; align-content: center;}
.t-badge { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 8px; text-align: center; border-radius: 4px; font-size: 0.95rem; font-weight: bold;}

/* 中间：压测高光区 */
.insight-bar { background: rgba(230, 162, 60, 0.1); border: 1px solid rgba(230, 162, 60, 0.3); padding: 12px 20px; text-align: center; font-size: 1rem; border-radius: 4px; margin-top: 15px;}
.text-warning { color: #E6A23C; }
.chart-wrap { width: 100%; height: 100%; min-height: 200px; }

/* 右侧：遥测区 */
.telemetry-status { display: flex; justify-content: space-between; font-size: 0.85rem; font-family: monospace; color: #94a3b8; border-bottom: 1px dashed rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 10px;}
.blink-text { animation: blink 1.5s infinite; }

/* 色彩工具 */
.text-cyan { color: #00f0ff; } .text-gold { color: #ffb800; } .text-green { color: #00ff88; } .text-blue { color: #38bdf8;} .text-gray { color: #64748b;} .font-bold { font-weight: bold;}
</style>