<template>
  <div class="grand-screen-container">
    <div class="grand-bg"></div>
    <div class="grand-grid"></div>
    <div class="grand-light-beam"></div>

    <header class="grand-header">
      <div class="header-flank left-flank">
        <div class="h-dec-line"></div>
        <div class="h-time"><span class="icon">◷</span> DATA CABIN // {{ currentTime }}</div>
      </div>
      
      <div class="header-core">
        <div class="core-bg">
          <h1 class="core-title">多模态特征融合与消融实验控制台</h1>
        </div>
        <div class="core-glow"></div>
      </div>

      <div class="header-flank right-flank">
        <div class="h-status"><span class="pulse-dot"></span> Pytorch Engine Online</div>
        <div class="h-dec-line"></div>
      </div>
    </header>

    <div class="screen-main" :class="{ 'is-loaded': isLoaded }">
      
      <aside class="panel-column left-column">
        <div class="heavy-panel h-50">
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
          
          <div class="panel-header">
            <span class="p-icon">⬡</span> 多模态特征空间分布 (49D)
          </div>
          <div class="panel-body" ref="featurePieChartRef"></div>
        </div>

        <div class="heavy-panel h-50 mt-20">
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
          
          <div class="panel-header">
            <span class="p-icon">⚙️</span> 模型超参空间 (Hyperparameters)
          </div>
          <div class="panel-body param-list">
            <div class="param-row"><span class="p-name">Dataset Split</span><span class="p-value text-cyan">70% / 10% / 20%</span></div>
            <div class="param-row"><span class="p-name">Random Seed</span><span class="p-value text-cyan">42</span></div>
            <div class="param-row"><span class="p-name">Optimizer</span><span class="p-value text-gold">AdamW (lr=1e-3)</span></div>
            <div class="param-row"><span class="p-name">Exp 1~3 Loss</span><span class="p-value">CrossEntropy</span></div>
            <div class="param-row"><span class="p-name">Exp 4 Loss</span><span class="p-value text-green font-bold">FocalLoss (γ=2.0)</span></div>
            <div class="param-row"><span class="p-name">Architecture</span><span class="p-value">Dual-Encoder Attention</span></div>
          </div>
        </div>
      </aside>

      <main class="panel-column center-column">
        <div class="hud-matrix">
          <div class="hud-block">
            <div class="hud-title">训练样本规模</div>
            <div class="hud-num text-cyan">52,480<span class="unit">条</span></div>
          </div>
          <div class="hud-block">
            <div class="hud-title">特征融合维度</div>
            <div class="hud-num text-green">49<span class="unit">D</span></div>
          </div>
          <div class="hud-block">
            <div class="hud-title">单步推理延迟</div>
            <div class="hud-num text-blue">18.5<span class="unit">ms</span></div>
          </div>
          <div class="hud-block highlight-hud">
            <div class="hud-title">最优均衡度 (Macro-F1)</div>
            <div class="hud-num text-gold">81.66<span class="unit">%</span></div>
          </div>
        </div>

        <div class="heavy-panel flex-2 mt-20">
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
          
          <div class="panel-header justify-center">
            <span class="p-icon">📈</span> 核心评估指标演进态势 (Accuracy vs Macro-F1)
          </div>
          <div class="panel-body" ref="evolutionChartRef"></div>
        </div>

        <div class="heavy-panel flex-1 mt-20">
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
          
          <div class="panel-header">
            <span class="p-icon">🧩</span> 递进式特征消融矩阵 (Ablation Matrix)
          </div>
          <div class="panel-body ablation-grid">
            <div class="a-row a-header">
               <span>阶段模型</span><span>轨迹运动学</span><span>OSM空间拓扑</span><span>气象特征解耦</span><span>Loss 优化</span>
            </div>
            <div class="a-row">
               <span class="text-gray">Exp1 (基线)</span>
               <span class="a-dot active"></span><span class="a-dot"></span><span class="a-dot"></span><span>CrossEntropy</span>
            </div>
            <div class="a-row">
               <span class="text-cyan">Exp2 (+OSM)</span>
               <span class="a-dot active"></span><span class="a-dot active"></span><span class="a-dot"></span><span>CrossEntropy</span>
            </div>
            <div class="a-row">
               <span class="text-green">Exp3 (+气象)</span>
               <span class="a-dot active"></span><span class="a-dot active"></span><span class="a-dot active"></span><span>CrossEntropy</span>
            </div>
            <div class="a-row a-highlight">
               <span class="text-gold font-bold">Exp4 (终极)</span>
               <span class="a-dot active"></span><span class="a-dot active"></span><span class="a-dot active"></span><span class="text-gold font-bold">FocalLoss</span>
            </div>
          </div>
        </div>
      </main>

      <aside class="panel-column right-column">
        <div class="heavy-panel h-50">
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
          
          <div class="panel-header">
            <span class="p-icon">🎯</span> 弱势类簇四阶跟踪 (Tail Samples F1)
          </div>
          <div class="panel-body" ref="longTailChartRef"></div>
        </div>

        <div class="heavy-panel h-50 mt-20">
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
          
          <div class="panel-header">
            <span class="p-icon">🧠</span> 引擎综合能力研判 (Radar Diagnosis)
          </div>
          <div class="panel-body flex-col">
            <div class="tech-tabs">
              <div class="t-btn" :class="{ active: currentExp === 'exp1' }" @click="currentExp = 'exp1'">Exp1</div>
              <div class="t-btn" :class="{ active: currentExp === 'exp2' }" @click="currentExp = 'exp2'">Exp2</div>
              <div class="t-btn" :class="{ active: currentExp === 'exp3' }" @click="currentExp = 'exp3'">Exp3</div>
              <div class="t-btn" :class="{ active: currentExp === 'exp4' }" @click="currentExp = 'exp4'">Exp4</div>
            </div>
            <div class="radar-wrap" ref="radarChartRef"></div>
            <div class="radar-conclusion" :class="radarTheme">
              {{ currentConclusion }}
            </div>
          </div>
        </div>
      </aside>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import * as echarts from 'echarts'

const isLoaded = ref(false)

// --- 顶部时间逻辑 ---
const currentTime = ref('')
let timeInterval: number
const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleString('zh-CN', { hour12: false })
}

// --- ECharts Refs ---
const featurePieChartRef = ref<HTMLElement | null>(null)
const evolutionChartRef = ref<HTMLElement | null>(null)
const longTailChartRef = ref<HTMLElement | null>(null)
const radarChartRef = ref<HTMLElement | null>(null)

let featurePieChart: echarts.ECharts | null = null
let evolutionChart: echarts.ECharts | null = null
let longTailChart: echarts.ECharts | null = null
let radarChart: echarts.ECharts | null = null

// --- 动态雷达图逻辑 ---
const currentExp = ref('exp4')

const radarDataMap = {
  exp1: { theme: 'blue', value: [83, 50, 40, 50, 40], conclusion: '作为纯轨迹基线，全局准度尚可，但长尾类识别率极低，空间感知极弱。' },
  exp2: { theme: 'cyan', value: [84, 65, 90, 50, 60], conclusion: '引入 OSM 后空间感知力拉满，公交车等强路网模态识别率飙升。' },
  exp3: { theme: 'green', value: [85, 70, 90, 95, 80], conclusion: '引入气象解耦，抗噪鲁棒性登顶，全局 Acc 达到巅峰 (84.50%)。' },
  exp4: { theme: 'gold', value: [82, 95, 90, 95, 90], conclusion: '采用 Focal Loss，虽舍弃微小 Acc，却使得长尾类簇召回率实现降维打击。' }
}

const currentConclusion = computed(() => radarDataMap[currentExp.value as keyof typeof radarDataMap].conclusion)
const radarTheme = computed(() => radarDataMap[currentExp.value as keyof typeof radarDataMap].theme)

// --- 初始化图表引擎 ---
const initCharts = () => {
  // 1. 左侧环形特征分布图
  if (featurePieChartRef.value) {
    featurePieChart = echarts.init(featurePieChartRef.value)
    featurePieChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'item', backgroundColor: 'rgba(5, 12, 30, 0.9)', borderColor: '#00f0ff', textStyle: { color: '#fff' } },
      legend: { bottom: '0%', left: 'center', textStyle: { color: '#94a3b8' } },
      color: ['#00f0ff', '#ffb800', '#00ff88'],
      series: [
        {
          name: '特征占比', type: 'pie', radius: ['45%', '70%'], center: ['50%', '42%'],
          avoidLabelOverlap: false,
          itemStyle: { borderRadius: 10, borderColor: '#010510', borderWidth: 2 },
          label: { show: false, position: 'center' },
          emphasis: { label: { show: true, fontSize: 16, fontWeight: 'bold', formatter: '{b}\n{c}D' } },
          labelLine: { show: false },
          data: [
            { value: 18, name: '轨迹运动学' },
            { value: 21, name: 'OSM空间拓扑' },
            { value: 10, name: '气象因子解耦' }
          ]
        }
      ]
    })
  }

  // 2. 中间演进图
  if (evolutionChartRef.value) {
    evolutionChart = echarts.init(evolutionChartRef.value)
    evolutionChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(5, 12, 30, 0.9)', borderColor: '#00f0ff', textStyle: { color: '#fff' } },
      legend: { data: ['Accuracy (全局准度)', 'Macro-F1 (均衡度)'], textStyle: { color: '#94a3b8', fontSize: 13 }, top: 0, right: 10 },
      grid: { left: '3%', right: '4%', bottom: '5%', top: '15%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: ['Exp1', 'Exp2', 'Exp3', 'Exp4'], axisLabel: { color: '#94a3b8', fontSize: 13, fontWeight: 'bold' }, axisLine: { lineStyle: { color: '#1e293b', width: 2 } } },
      yAxis: { type: 'value', min: 75, max: 86, axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } } },
      series: [
        { name: 'Accuracy (全局准度)', type: 'line', data: [83.30, 83.88, 84.50, 81.57], smooth: true, symbolSize: 10, itemStyle: { color: '#00f0ff' }, lineStyle: { width: 3, shadowBlur: 10, shadowColor: 'rgba(0,240,255,0.8)' } },
        { name: 'Macro-F1 (均衡度)', type: 'line', data: [75.89, 78.27, 79.50, 81.66], smooth: true, symbolSize: 10, itemStyle: { color: '#ffb800' }, lineStyle: { width: 3, shadowBlur: 10, shadowColor: 'rgba(255,184,0,0.8)' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(255, 184, 0, 0.4)' }, { offset: 1, color: 'transparent' }]) } }
      ]
    })
  }

  // 3. 右上：彻底重构的长尾靶向柱状图 (四阶并列对比！)
  if (longTailChartRef.value) {
    longTailChart = echarts.init(longTailChartRef.value)
    longTailChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(5, 12, 30, 0.9)', borderColor: '#00f0ff', textStyle: { color: '#fff' } },
      legend: { data: ['Exp1', 'Exp2', 'Exp3', 'Exp4'], textStyle: { color: '#94a3b8' }, top: 0, right: 0 },
      grid: { left: '3%', right: '4%', bottom: '5%', top: '20%', containLabel: true },
      xAxis: { type: 'category', data: ['自行车', '公交车', '地铁', '火车'], axisLabel: { color: '#94a3b8', fontSize: 12, fontWeight: 'bold' }, axisLine: { lineStyle: { color: '#1e293b' } } },
      yAxis: { type: 'value', min: 60, max: 95, axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } } },
      // 包含四个实验阶段的真实阶梯演进数据
      series: [
        { name: 'Exp1', type: 'bar', barWidth: '15%', data: [79.2, 70.1, 68.5, 75.3], itemStyle: { color: '#334155' } },
        { name: 'Exp2', type: 'bar', barWidth: '15%', data: [80.5, 79.8, 72.2, 76.5], itemStyle: { color: '#00f0ff' } },
        { name: 'Exp3', type: 'bar', barWidth: '15%', data: [82.1, 81.2, 75.1, 78.0], itemStyle: { color: '#00ff88' } },
        { name: 'Exp4', type: 'bar', barWidth: '15%', data: [91.2, 74.4, 74.2, 85.1], itemStyle: { color: '#ffb800' }, label: { show: true, position: 'top', color: '#ffb800', formatter: '{c}%', fontSize: 10 } }
      ]
    })
  }

  // 4. 右下：能力雷达图初始化
  if (radarChartRef.value) {
    radarChart = echarts.init(radarChartRef.value)
    updateRadarChart()
  }
}

// 动态更新雷达图
const updateRadarChart = () => {
  if (!radarChart) return
  const data = radarDataMap[currentExp.value as keyof typeof radarDataMap]
  let lineColor = '#ffb800'
  if(data.theme === 'blue') lineColor = '#38bdf8'
  if(data.theme === 'cyan') lineColor = '#00f0ff'
  if(data.theme === 'green') lineColor = '#00ff88'

  radarChart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item' },
    radar: {
      indicator: [
        { name: '全局准确率 (Acc)', max: 100 },
        { name: '长尾召回力', max: 100 },
        { name: '空间感知力', max: 100 },
        { name: '抗噪鲁棒性', max: 100 },
        { name: '泛化稳定性', max: 100 }
      ],
      radius: '65%',
      center: ['50%', '50%'],
      splitNumber: 4,
      axisName: { color: '#cbd5e1', fontSize: 10 },
      splitArea: { areaStyle: { color: ['rgba(0, 240, 255, 0.05)', 'rgba(0, 240, 255, 0.02)'] } },
      axisLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.2)' } },
      splitLine: { lineStyle: { color: 'rgba(0, 240, 255, 0.2)' } }
    },
    series: [
      {
        name: '模型能力',
        type: 'radar',
        data: [ { value: data.value, name: currentExp.value.toUpperCase() } ],
        itemStyle: { color: lineColor },
        lineStyle: { width: 2, color: lineColor },
        areaStyle: { color: lineColor, opacity: 0.3 }
      }
    ]
  })
}

// 监听按钮切换，重绘雷达
watch(currentExp, () => {
  updateRadarChart()
})

onMounted(() => {
  updateTime()
  timeInterval = window.setInterval(updateTime, 1000)
  isLoaded.value = true
  nextTick(initCharts)
  window.addEventListener('resize', () => { featurePieChart?.resize(); evolutionChart?.resize(); longTailChart?.resize(); radarChart?.resize(); })
})

onUnmounted(() => {
  if (timeInterval) clearInterval(timeInterval)
  window.removeEventListener('resize', () => { featurePieChart?.resize(); evolutionChart?.resize(); longTailChart?.resize(); radarChart?.resize(); })
  featurePieChart?.dispose(); evolutionChart?.dispose(); longTailChart?.dispose(); radarChart?.dispose();
})
</script>

<style scoped>
/* 极具压迫感的军工大屏基调 */
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
.grand-grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px); background-size: 50px 50px; z-index: 0; perspective: 1000px; transform-style: preserve-3d; opacity: 0.5;}
.grand-light-beam { position: absolute; top: 0; left: 0; right: 0; height: 150px; background: linear-gradient(180deg, rgba(0, 240, 255, 0.05) 0%, transparent 100%); z-index: 0;}

/* 1. 顶部主梁 */
.grand-header {
  position: relative; z-index: 2; height: 75px; display: flex; justify-content: space-between; align-items: flex-start; padding: 0 20px;
}
.header-flank { flex: 1; display: flex; flex-direction: column; justify-content: center; height: 50px; padding-top: 10px;}
.h-time { color: #00f0ff; font-family: monospace; font-size: 1rem; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0,240,255,0.6); margin-top: 5px;}
.h-dec-line { height: 2px; background: linear-gradient(90deg, rgba(0,240,255,0.8), transparent); width: 100%; position: relative;}
.right-flank .h-dec-line { background: linear-gradient(270deg, rgba(0,240,255,0.8), transparent); }
.h-status { text-align: right; color: #00ff88; font-size: 0.95rem; margin-top: 5px; text-transform: uppercase; font-family: monospace; letter-spacing: 1px;}
.pulse-dot { display: inline-block; width: 8px; height: 8px; background: #00ff88; border-radius: 50%; box-shadow: 0 0 10px #00ff88; animation: blink 1.5s infinite; margin-right: 8px;}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

.header-core { flex: 2.5; display: flex; justify-content: center; position: relative;}
.core-bg {
  background: rgba(4, 18, 48, 0.8);
  border: 2px solid rgba(0, 240, 255, 0.4);
  border-top: none;
  padding: 10px 80px 20px;
  clip-path: polygon(0 0, 100% 0, 92% 100%, 8% 100%);
  box-shadow: inset 0 -15px 30px rgba(0, 240, 255, 0.2);
  display: flex; align-items: center; justify-content: center;
  position: relative; z-index: 2;
}
.core-title { margin: 0; font-size: 2.1rem; font-weight: 900; color: #fff; letter-spacing: 5px; text-shadow: 0 0 20px rgba(0,240,255,0.8); }
.core-glow { position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 60%; height: 15px; background: #00f0ff; filter: blur(25px); z-index: 1; opacity: 0.5;}

/* 2. 主体 1:2:1 阵列布局 */
.screen-main {
  position: relative; z-index: 2; flex: 1; padding: 15px 20px; display: grid;
  grid-template-columns: 1fr 2.2fr 1fr; gap: 20px; height: calc(100% - 75px);
  opacity: 0; transform: scale(0.98); transition: all 0.8s cubic-bezier(0.1, 0.9, 0.2, 1);
}
.screen-main.is-loaded { opacity: 1; transform: scale(1); }
.panel-column { display: flex; flex-direction: column; }

/* 3. 重型军工面板 */
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
.panel-body { flex: 1; position: relative; overflow: auto; width: 100%; height: 100%; min-height: 150px;}

.h-50 { height: calc(50% - 10px); }
.flex-1 { flex: 1; } .flex-2 { flex: 2; }
.flex-col { display: flex; flex-direction: column; height: 100%;}
.mt-20 { margin-top: 20px; }

/* 左侧：超参表 */
.param-list { display: flex; flex-direction: column; justify-content: space-around;}
.param-row { display: flex; justify-content: space-between; font-size: 0.95rem; padding: 10px; background: rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.03);}
.p-name { color: #cbd5e1; } .p-value { font-family: monospace; font-size: 1rem;}

/* 中间：数字列阵 */
.hud-matrix { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; height: 100px; }
.hud-block { background: rgba(4, 14, 38, 0.8); border: 1px solid rgba(0, 240, 255, 0.2); padding: 15px 20px; display: flex; flex-direction: column; justify-content: center; position: relative; box-shadow: 0 5px 15px rgba(0,0,0,0.5);}
.hud-block::before { content:''; position: absolute; left: 0; bottom: 0; width: 100%; height: 3px; background: linear-gradient(90deg, #00f0ff, transparent); }
.highlight-hud { background: rgba(255,184,0,0.05); border-color: rgba(255,184,0,0.3); }
.highlight-hud::before { background: linear-gradient(90deg, #ffb800, transparent); }
.hud-title { font-size: 0.85rem; color: #94a3b8; margin-bottom: 5px; text-transform: uppercase; font-weight: bold; letter-spacing: 1px;}
.hud-num { font-size: 2.5rem; font-weight: 900; font-family: 'Impact', monospace; line-height: 1; letter-spacing: 2px; text-shadow: 0 0 15px currentColor; }
.hud-num .unit { font-size: 0.9rem; font-weight: normal; margin-left: 6px; opacity: 0.6; text-shadow: none; font-family: 'Rajdhani'; letter-spacing: 0;}

/* 中下：消融矩阵 */
.ablation-grid { display: flex; flex-direction: column; justify-content: space-around; padding-top: 5px;}
.a-row { display: grid; grid-template-columns: 1.5fr 1.2fr 1.2fr 1.2fr 1.5fr; font-size: 0.95rem; padding: 10px; background: rgba(0,0,0,0.3); align-items: center; text-align: center; border: 1px solid transparent;}
.a-header { color: #00f0ff; font-weight: bold; background: rgba(0,240,255,0.1); border-bottom: 1px solid #00f0ff; margin-bottom: 5px;}
.a-highlight { border: 1px solid rgba(255,184,0,0.5); background: rgba(255,184,0,0.1); box-shadow: 0 0 15px rgba(255,184,0,0.1) inset;}
.a-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin: 0 auto;}
.a-dot.active { background: #00f0ff; box-shadow: 0 0 10px #00f0ff;}
.a-highlight .a-dot.active { background: #ffb800; box-shadow: 0 0 10px #ffb800;}
.a-dot:not(.active) { background: #1e293b; border: 1px solid #334155;}

/* 右翼：雷达选项卡与总结 */
.tech-tabs { display: flex; gap: 10px; margin-bottom: 10px; }
.t-btn { flex: 1; text-align: center; padding: 8px 0; background: rgba(0,0,0,0.5); color: #64748b; font-size: 1rem; font-weight: bold; cursor: pointer; border: 1px solid rgba(255,255,255,0.1); transition: all 0.3s; transform: skewX(-15deg); }
.t-btn > span { display: inline-block; transform: skewX(15deg); }
.t-btn.active { background: rgba(0,240,255,0.15); color: #fff; border-color: #00f0ff; box-shadow: 0 0 10px rgba(0,240,255,0.2);}

.radar-wrap { flex: 1; min-height: 180px; width: 100%;}
.radar-conclusion { font-size: 0.95rem; line-height: 1.5; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 4px; text-align: justify; font-weight: bold; border-left: 4px solid;}
.theme-blue { border-color: #38bdf8; color: #38bdf8; background: rgba(56, 189, 248, 0.1); }
.theme-cyan { border-color: #00f0ff; color: #00f0ff; background: rgba(0, 240, 255, 0.1); }
.theme-green { border-color: #00ff88; color: #00ff88; background: rgba(0, 255, 136, 0.1); }
.theme-gold { border-color: #ffb800; color: #ffb800; background: rgba(255, 184, 0, 0.1); }

/* 色彩工具 */
.text-cyan { color: #00f0ff; } .text-gold { color: #ffb800; } .text-green { color: #00ff88; } .text-blue { color: #38bdf8;} .text-gray { color: #64748b;} .font-bold { font-weight: bold;}
</style>