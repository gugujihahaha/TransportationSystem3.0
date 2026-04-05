<template>
  <div class="homepage-container">
    <el-skeleton :loading="loading" animated :rows="15">
      <template #default>
        <div class="dashboard-grid">
          
          <div class="glass-card fade-up card-delay-1">
            <div class="card-title">交通方式识别构成（Exp4模型）</div>
            <div class="chart-wrapper relative-box">
              <div class="ring-center">
                <div class="center-label">总样本</div>
                <div class="center-value">521</div>
              </div>
              <div ref="chart1Ref" class="chart-container"></div>
            </div>
          </div>

          <div class="glass-card map-card fade-up card-delay-2">
            <div class="card-title">北京各区拥堵热力分布</div>
            <div class="map-wrapper relative-box">
              <div ref="chart2Ref" class="chart-container map-canvas"></div>
              
              <div class="custom-legend">
                <div class="legend-note">
                  <span class="highlight-text">数据说明：</span><br>
                  拥堵指数经归一化处理（0-1）。<br>
                  <span class="sub-note" style="color: #FFD700;">👆 点击地图各行政区可查看详细交通档案</span>
                </div>
              </div>
            </div>
          </div>

          <div class="glass-card flex-col-card fade-up card-delay-3">
            <div class="card-title">模型拥堵识别准确率演进</div>
            <div ref="chart3Ref" class="chart-container-line"></div>
            
            <div class="carbon-panel">
              <div class="carbon-text">累计减碳 <span class="num-glow">{{ animatedCarbonKg.toFixed(1) }}</span> kg</div>
              <div class="carbon-divider">≈</div>
              <div class="carbon-text">种植 <span class="num-glow">{{ animatedCarbonTrees.toFixed(1) }}</span> 棵树</div>
            </div>
          </div>

          <div class="glass-card fade-up card-delay-4">
            <div class="card-title">常发拥堵路段年均拥堵时长排行</div>
            <div ref="chart4Ref" class="chart-container"></div>
          </div>

          <div class="empty-placeholder"></div>

          <div class="glass-card fade-up card-delay-5">
            <div class="card-title">拥堵成因贡献度分析</div>
            <div ref="chart5Ref" class="chart-container"></div>
          </div>

        </div>

        <div class="glass-card table-card fade-up card-delay-6">
          <div class="card-title">重点拥堵路段精细化治理建议</div>
          <el-table 
            :data="dashboardData.governance_table" 
            style="width: 100%" 
            class="cyber-table"
            :row-style="{ background: 'transparent' }"
            :cell-style="{ borderBottom: '1px solid rgba(255,255,255,0.08)' }"
            :header-cell-style="{ background: 'rgba(56,189,248,0.1)', color: '#e2e8f0', fontWeight: 'bold', borderBottom: '1px solid rgba(56,189,248,0.3)' }"
          >
            <el-table-column prop="road" label="路段名称" width="220" />
            <el-table-column prop="car_ratio" label="私家车出行占比" width="160">
              <template #default="scope">
                <span class="highlight-ratio">{{ scope.row.car_ratio }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="suggestion" label="治理决策建议" />
          </el-table>
        </div>
      </template>
    </el-skeleton>

    <div class="data-source-trigger" v-if="!loading">
      <div class="trigger-dot">数据说明</div>
      <div class="source-tooltip">
        拥堵指数基于 GeoLife (2008-2012) 与自研 Exp4 模型；出行结构引用北京交通发展研究院 2024 报告；拥堵趋势引用高德报告。模型可迁移至新数据。
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import * as echarts from 'echarts'

const router = useRouter()
const loading = ref(true)
const dashboardData = ref(null)

// 远郊区与部分缺失城区的估算数据
const estimatedMapData = {
  "通州区": 0.45, "大兴区": 0.42, "昌平区": 0.40, "顺义区": 0.35,
  "房山区": 0.32, "门头沟区": 0.28, "怀柔区": 0.25, "密云区": 0.22,
  "延庆区": 0.20, "平谷区": 0.20
}

// 滚动数字动画
const animatedCarbonKg = ref(0)
const animatedCarbonTrees = ref(0)

const animateValue = (targetRef, endValue, duration = 2000) => {
  let startTimestamp = null
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp
    const progress = Math.min((timestamp - startTimestamp) / duration, 1)
    const easeProgress = 1 - Math.pow(1 - progress, 4)
    targetRef.value = easeProgress * endValue
    if (progress < 1) window.requestAnimationFrame(step)
    else targetRef.value = endValue
  }
  window.requestAnimationFrame(step)
}

const chart1Ref = ref(null)
const chart2Ref = ref(null)
const chart3Ref = ref(null)
const chart4Ref = ref(null)
const chart5Ref = ref(null)
const charts = []

// 数据请求
const loadData = async () => {
  try {
    const res = await fetch('/homepage_data.json')
    if (res.ok) {
      const data = await res.json()
      // 合并补充后的全区数据
      data.map_heatmap = { ...data.map_heatmap, ...estimatedMapData }
      dashboardData.value = data
    }
  } catch (err) {
    console.error('加载面板数据失败:', err)
  } finally {
    loading.value = false
    await nextTick()
    
    if (dashboardData.value?.carbon_reduction) {
      animateValue(animatedCarbonKg, dashboardData.value.carbon_reduction.kg, 2500)
      animateValue(animatedCarbonTrees, dashboardData.value.carbon_reduction.trees, 2500)
    }
    initCharts()
  }
}

// 图表初始化
const initCharts = async () => {
  if (!dashboardData.value) return
  const data = dashboardData.value

  // --- 卡片1：环形图 ---
  const chart1 = echarts.init(chart1Ref.value)
  const modeColors = { '步行': '#00FF88', '骑行': '#00FFFF', '公交': '#FFDD00', '地铁': '#CC33FF', '火车': '#FF6600', '小汽车': '#FF0033' }
  chart1.setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)', backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff', fontWeight: 'bold' } },
    series: [{
      type: 'pie', radius: ['55%', '75%'], center: ['50%', '50%'],
      itemStyle: { borderColor: 'rgba(15, 25, 45, 0.8)', borderWidth: 2 },
      label: { show: true, formatter: '{b} {d}%', color: '#e2e8f0', fontSize: 12, fontWeight: 'bold' },
      labelLine: { length: 10, length2: 15, lineStyle: { color: 'rgba(255,255,255,0.4)' } },
      data: Object.entries(data.mode_distribution).map(([name, value]) => ({
        name, value, itemStyle: { color: modeColors[name] }
      }))
    }]
  })
  charts.push(chart1)

// --- 卡片2：高亮发光版全区热力地图 + GPS真实热力点透视 + 路由跳转 ---
  const chart2 = echarts.init(chart2Ref.value)
  try {
    const geoRes = await fetch('https://geo.datav.aliyun.com/areas_v3/bound/110000_full.json')
    const geoJson = await geoRes.json()
    echarts.registerMap('BJ', geoJson)
    
    const estimatedKeys = Object.keys(estimatedMapData)

    // 1. 加载底层 GPS 小汽车网格热力图数据
    let heatmapData = []
    try {
      const res = await fetch('/heatmap_grid.json')
      if (res.ok) heatmapData = await res.json()
    } catch (e) { 
      console.warn('GPS热力图数据加载失败，继续渲染基础地图', e) 
    }

    // 2. 动态构建 series 数组
    let seriesList = []

    // 如果热力图数据存在，则先画底层热力图 (zlevel 0, 数组 index 0)
    if (heatmapData.length > 0) {
      seriesList.push({
        type: 'heatmap',
        coordinateSystem: 'geo',
        data: heatmapData.map(p => [p.lng, p.lat, p.value]),
        pointSize: 10,
        blurSize: 15,
        label: { show: false },
        emphasis: { disabled: true },
        tooltip: { show: false },
        silent: true, // 鼠标事件穿透，交给上层的行政区地图处理
        zlevel: 0
      })
    }

    // 将行政区地图追加在上层 (数组 index 将变为 1 或 0)
    seriesList.push({
      type: 'map', 
      map: 'BJ', 
      roam: true, 
      zoom: 1.1, 
      center: [116.405285, 40.003989], 
      label: {
        show: true,
        fontSize: 11,
        fontWeight: 'bold',
        color: '#fff',
        textShadow: '0 0 4px #000',
        fontFamily: '"PingFang SC", "Microsoft YaHei", sans-serif'
      },
      emphasis: { 
        label: { color: '#fff', fontSize: 12, fontWeight: 'bold' }, 
        itemStyle: { areaColor: '#38bdf8', borderColor: '#fff', borderWidth: 2, shadowColor: 'rgba(56,189,248,0.8)', shadowBlur: 10 } 
      },
      itemStyle: { 
        borderColor: 'rgba(255,255,255,0.3)',
        borderWidth: 1
      },
      data: Object.entries(data.map_heatmap).map(([name, value]) => ({ name, value }))
    })

    // 3. 动态构建 visualMap 数组，防止热力图和行政区颜色互相污染
    let visualMapConfig = [
      {
        show: true,
        type: 'piecewise',
        pieces: [
          { min: 0, max: 0.2, color: '#FFE4B5', label: '畅通' },
          { min: 0.2, max: 0.35, color: '#FFD700', label: '基本畅通' },
          { min: 0.35, max: 0.5, color: '#FFA500', label: '轻度拥堵' },
          { min: 0.5, max: 0.65, color: '#FF4500', label: '中度拥堵' },
          { min: 0.65, max: 1, color: '#B22222', label: '严重拥堵' }
        ],
        left: 15,
        bottom: 15,
        text: ['拥堵', '畅通'],
        textStyle: { color: '#e2e8f0', fontSize: 10, fontWeight: 'bold' },
        seriesIndex: seriesList.length - 1 // 始终只作用于最后一个系列（即行政区地图）
      }
    ]

    // 如果启用了热力图，为它追加连续色带
    if (heatmapData.length > 0) {
      visualMapConfig.push({
        show: false, // 隐藏热力图的刻度条
        type: 'continuous',
        min: 0,
        max: 1,
        seriesIndex: 0, // 仅作用于底层的热力图系列
        inRange: {
          // 经典的深冷到高热色带，增加视觉冲击
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        }
      })
    }

    // 4. 渲染图表
    chart2.setOption({
      backgroundColor: '#0A0E17', 
      // 必须为热力图预留一个隐形的坐标系基座
      geo: {
        map: 'BJ',
        roam: true,
        zoom: 1.1,
        center: [116.405285, 40.003989],
        silent: true, 
        itemStyle: { opacity: 0 } // 完全透明隐藏
      },
      tooltip: { 
        trigger: 'item', 
        backgroundColor: 'rgba(10,14,23,0.95)', 
        borderColor: '#FFD700',
        borderWidth: 1,
        textStyle: { color: '#fff' },
        formatter: (params) => {
          // 仅当划过行政区地图时，展示 Tooltip 详细信息
          if (params.seriesType === 'map') {
            const val = params.value || 0
            const isEst = estimatedKeys.includes(params.name)
            const estTag = isEst ? '<span style="color:#FFDD00; font-size:10px; margin-left:4px; padding:1px 4px; border:1px solid rgba(255,221,0,0.5); border-radius:2px;">估算值</span>' : ''
            return `<div style="font-size:15px; font-weight:bold; margin-bottom:4px;">${params.name} ${estTag}</div>
                    <div style="font-weight:bold; color:#FFD700;">拥堵指数: ${val.toFixed(3)}</div>
                    <div style="font-size:11px; color:#cbd5e1; margin-top:4px;">(点击查看区域详情)</div>`
          }
          return ''
        }
      },
      visualMap: visualMapConfig,
      series: seriesList
    })

    // ===== 地图点击事件：路由跳转 =====
    chart2.on('click', (params) => {
      // 保险起见，再次判断点击的是行政区域块
      if (params.seriesType === 'map' && params.name) {
        router.push(`/region/${params.name}`)
      }
    })
    // ===================================

  } catch (err) { console.warn('地图加载失败', err) }
  charts.push(chart2)

  // --- 卡片3：准确率折线 ---
  const chart3 = echarts.init(chart3Ref.value)
  chart3.setOption({
    grid: { top: 20, bottom: 20, left: 40, right: 10 },
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff', fontWeight: 'bold' } },
    xAxis: { type: 'category', data: data.accuracy_trend.exp_names, axisLabel: { color: '#cbd5e1', fontSize: 11, fontWeight: 'bold' }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.15)' } } },
    yAxis: { type: 'value', min: 0.8, axisLabel: { color: '#cbd5e1', fontSize: 11, formatter: val => (val*100).toFixed(0)+'%' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } } },
    series: [{
      data: data.accuracy_trend.accuracy, type: 'line', smooth: true, color: '#00A8FF',
      symbol: 'circle', symbolSize: 6,
      itemStyle: { color: '#00A8FF' },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(0,168,255,0.4)' }, { offset: 1, color: 'rgba(0,168,255,0)' }]) }
    }]
  })
  charts.push(chart3)

  // --- 卡片4：排行横向柱状图 ---
  const chart4 = echarts.init(chart4Ref.value)
  const rankData = [...data.congestion_ranking].reverse()
  chart4.setOption({
    grid: { top: 10, bottom: 10, left: 10, right: 40, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff', fontWeight: 'bold' } },
    xAxis: { type: 'value', show: false },
    yAxis: { type: 'category', data: rankData.map(d => d.road), axisLabel: { color: '#e2e8f0', fontSize: 12, fontWeight: '600' }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{
      type: 'bar', data: rankData.map(d => d.hours), barWidth: 14,
      label: { show: true, position: 'right', color: '#e2e8f0', fontWeight: 'bold', formatter: '{c}h', fontSize: 12 },
      itemStyle: {
        borderRadius: 8,
        color: (params) => {
          if (params.dataIndex === rankData.length - 1) { 
            return new echarts.graphic.LinearGradient(1, 0, 0, 0, [{ offset: 0, color: '#B22222' }, { offset: 1, color: '#FF4500' }])
          }
          return new echarts.graphic.LinearGradient(1, 0, 0, 0, [{ offset: 0, color: '#FFA500' }, { offset: 1, color: '#FFD700' }])
        }
      }
    }]
  })
  charts.push(chart4)

  // --- 卡片5：成因饼图 ---
  const chart5 = echarts.init(chart5Ref.value)
  const causeColors = { '私家车出行占比': '#FF4500', '路网属性限制': '#00A8FF', '天气因素影响': '#36CFC9', '其他': '#AAAAAA' }
  chart5.setOption({
    tooltip: { trigger: 'item', backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff', fontWeight: 'bold' } },
    legend: { orient: 'vertical', right: 0, top: 'center', textStyle: { color: '#e2e8f0', fontSize: 11, fontWeight: '600' }, itemWidth: 12, itemHeight: 12 },
    series: [{
      type: 'pie', radius: '65%', center: ['40%', '50%'],
      itemStyle: { borderColor: 'rgba(15,25,45,0.8)', borderWidth: 1 },
      label: { show: true, formatter: '{d}%', color: '#FFFFFF', fontWeight: 'bold', position: 'inside', fontSize: 11, textBorderColor: '#000', textBorderWidth: 1 },
      data: Object.entries(data.cause_contribution).map(([name, value]) => ({
        name, value, itemStyle: { color: causeColors[name] }
      }))
    }]
  })
  charts.push(chart5)
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
.homepage-container { 
  width: 100%; 
  height: 100%; 
  box-sizing: border-box; 
  background: transparent; 
  /* 更清晰稳重的系统黑体 */
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif; 
}

/* 基础布局 */
.dashboard-grid {
  display: grid;
  grid-template-columns: 1.2fr 2.5fr 1.2fr;
  gap: 20px;
  margin-bottom: 20px;
}

/* ================= 玻璃卡片 ================= */
.glass-card {
  background: rgba(15, 25, 45, 0.6);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 16px;
  padding: 16px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
}
.glass-card:hover {
  box-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
  border-color: rgba(56, 189, 248, 0.6);
}

.card-title {
  font-size: 16px;
  color: #e2e8f0; /* 调亮，保证清晰度 */
  margin-bottom: 12px;
  font-weight: 600; /* 加重字重使其稳重 */
  letter-spacing: 0.5px;
}

.chart-container { flex: 1; min-height: 180px; width: 100%; }
.relative-box { position: relative; flex: 1; display: flex; flex-direction: column; }

/* ================= 地图高度及特效调整 ================= */
.map-card {
  height: 380px; 
  grid-row: span 2; 
  padding: 10px; 
  animation: borderGlow 4s infinite alternate; 
}
.map-wrapper {
  border-radius: 8px;
  overflow: hidden;
}
.map-canvas { 
  height: 100%; 
  width: 100%; 
  cursor: pointer; /* 提示可点击 */
}

@keyframes borderGlow {
  0% { box-shadow: 0 0 5px rgba(56,189,248,0.2); }
  100% { box-shadow: 0 0 15px rgba(56,189,248,0.6); border-color: rgba(56,189,248,0.6); }
}

/* 环形图中心文本 */
.ring-center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  pointer-events: none;
  z-index: 10;
}
.center-label { font-size: 12px; color: #cbd5e1; font-weight: 500;}
.center-value { 
  font-size: 26px; 
  color: #38bdf8; 
  font-weight: bold; 
  text-shadow: 0 0 8px rgba(56, 189, 248, 0.5); /* 阴影调弱变锐利 */
  font-family: 'Din', 'Arial', sans-serif; 
}

/* ================= 图注文本说明 ================= */
.custom-legend {
  position: absolute;
  bottom: 10px;
  right: 15px;
  background: rgba(10, 14, 23, 0.85);
  border: 1px solid rgba(56, 189, 248, 0.4);
  border-radius: 6px;
  padding: 8px 12px;
  z-index: 10;
  backdrop-filter: blur(4px);
  pointer-events: none; 
}
.legend-note {
  font-size: 11px;
  color: #cbd5e1; 
  line-height: 1.6;
  font-weight: 500;
}
.highlight-text {
  color: #fff;
  font-weight: bold;
}
.sub-note {
  font-size: 11px;
  display: inline-block;
  margin-top: 4px;
}

/* ================= 折线图 + 碳减排 ================= */
.flex-col-card { display: flex; flex-direction: column; justify-content: space-between; }
.chart-container-line { height: 160px; width: 100%; }
.carbon-panel {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  padding: 10px 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  margin-top: 5px;
}
.carbon-text { font-size: 13px; color: #e2e8f0; font-weight: 500;}
.num-glow {
  font-size: 26px;
  font-weight: bold;
  color: #00FF88;
  text-shadow: 0 0 8px rgba(0, 255, 136, 0.4);
  font-family: 'Din', 'Arial', sans-serif;
  margin: 0 4px;
}
.carbon-divider { font-size: 20px; color: #94a3b8; font-weight: bold; }

/* 占位 */
.empty-placeholder { display: none; }

/* ================= 表格 ================= */
.table-card { margin-top: 20px; }
.cyber-table {
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
  background-color: transparent !important;
  font-size: 13px;
  color: #e2e8f0;
}
:deep(.el-table__inner-wrapper::before) { display: none; }
:deep(.el-table tbody tr:hover > td) { background-color: rgba(56, 189, 248, 0.1) !important; }
.highlight-ratio { color: #FF4500; font-weight: bold; font-size: 14px;}

/* ================= 悬浮说明 ================= */
.data-source-trigger {
  position: fixed;
  bottom: 25px;
  right: 25px;
  z-index: 100;
}
.trigger-dot {
  background: rgba(15, 25, 45, 0.8);
  border: 1px solid rgba(56, 189, 248, 0.5);
  color: #38bdf8;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  cursor: help;
  backdrop-filter: blur(4px);
  transition: all 0.3s;
}
.source-tooltip {
  position: absolute;
  bottom: 35px;
  right: 0;
  width: 260px;
  background: rgba(0, 0, 0, 0.85);
  color: #fff;
  padding: 12px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 500;
  line-height: 1.6;
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(8px);
  opacity: 0;
  visibility: hidden;
  transform: translateY(10px);
  transition: all 0.3s;
}
.data-source-trigger:hover .source-tooltip {
  opacity: 1;
  visibility: visible;
  transform: translateY(0);
}
.data-source-trigger:hover .trigger-dot {
  background: #38bdf8;
  color: #0f192d;
}

/* ================= 动画特效 ================= */
.fade-up {
  opacity: 0;
  animation: fadeInUp 0.5s ease forwards;
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
.card-delay-1 { animation-delay: 0.1s; }
.card-delay-2 { animation-delay: 0.2s; }
.card-delay-3 { animation-delay: 0.3s; }
.card-delay-4 { animation-delay: 0.4s; }
.card-delay-5 { animation-delay: 0.5s; }
.card-delay-6 { animation-delay: 0.6s; }

@media (max-width: 1024px) {
  .dashboard-grid { grid-template-columns: 1fr; }
  .map-card { grid-row: span 1; height: 350px; }
}
</style>