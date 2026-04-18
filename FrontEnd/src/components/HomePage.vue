<template>
  <div class="homepage-container">
    <el-skeleton :loading="loading" animated :rows="15">
      <template #default>
        <div class="dashboard-grid">
          
          <div class="left-col">
            <div class="glass-card fade-up card-delay-1">
              <div class="card-title">交通方式识别构成</div>
              <div class="chart-wrapper relative-box">
                <div class="ring-center">
                  <div class="center-label">总样本</div>
                  <div class="center-value">{{ totalSamples }}</div>
                </div>
                <div ref="chart1Ref" class="chart-container"></div>
              </div>
            </div>

            <div class="glass-card flex-col-card fade-up card-delay-3" style="margin-top: 20px; flex: 1;">
              <div class="card-title">递进实验模型优化亮点</div>
              <div class="highlights-list">
                <div class="highlight-item"><span class="highlight-ratio">Exp2</span> 公交召回率提升 +10%</div>
                <div class="highlight-item"><span class="highlight-ratio">Exp3</span> 雨天识别准确率提升 +9%</div>
                <div class="highlight-item"><span class="highlight-ratio">Exp4</span> 小汽车长尾 F1 值 +0.04</div>
                <div class="highlight-item" style="color: #38bdf8;"> 多模态融合显著提升复杂场景鲁棒性</div>
              </div>
              
              <div class="carbon-panel">
                <div class="carbon-text">评估减碳收益<span class="num-glow">{{ animatedCarbonKg.toFixed(1) }}</span> kg</div>
              </div>
            </div>
          </div>

          <div class="glass-card map-card fade-up card-delay-2">
            <div class="card-title">北京各区轨迹数据覆盖热力 (真实采样分布)</div>
            <div class="map-wrapper relative-box">
              <div ref="chart2Ref" class="chart-container map-canvas"></div>
              
              <div class="custom-legend">
                <div class="legend-note">
                  <span class="highlight-text">数据映射说明：</span><br>
                  基于 GeoLife 真实网格点聚合。<br>
                  采用<span style="color:#00FFFF;">动态色彩截断</span>强化长尾区域显示。<br>
                  <span class="sub-note" style="color: #FFD700;">点击各区可查看真实覆盖量</span>
                </div>
              </div>
            </div>
          </div>

          <div class="right-col">
            <div class="glass-card fade-up card-delay-4">
              <div class="card-title">🌿 绿色出行比例 TOP5 区域</div>
              <div ref="chart4Ref" class="chart-container" style="height: 220px;"></div>
            </div>

            <div class="glass-card fade-up card-delay-5" style="margin-top: 20px; flex: 1;">
              <div class="card-title">四个递进模型核心特点</div>
              <div class="model-features">
                <div class="feature-item">
                  <span class="model-name" style="color: #38bdf8;">Exp1 纯轨迹基线</span>
                  <span class="model-desc">基于速度、加速度等运动学特征</span>
                </div>
                <div class="feature-item">
                  <span class="model-name" style="color: #00FF88;">Exp2 +OSM空间特征</span>
                  <span class="model-desc">引入道路类型、POI等地理语义</span>
                </div>
                <div class="feature-item">
                  <span class="model-name" style="color: #FFD700;">Exp3 +天气特征</span>
                  <span class="model-desc">融合气温、降水、风速等环境变量</span>
                </div>
                <div class="feature-item">
                  <span class="model-name" style="color: #FF4500;">Exp4 +Focal Loss</span>
                  <span class="model-desc">优化类别不平衡，提升长尾类别</span>
                </div>
              </div>
            </div>
          </div>

        </div>

        <div class="glass-card table-card fade-up card-delay-6">
          <div class="card-title" style="margin-bottom: 20px;">多源数据融合规模与模型实验统计</div>
          <div class="stats-row">
            <div class="stat-box">
              <div class="stat-value">13 <span class="unit">万+</span></div>
              <div class="stat-label">处理轨迹点</div>
            </div>
            <div class="stat-box">
              <div class="stat-value">49 <span class="unit">维</span></div>
              <div class="stat-label">多模态特征维度</div>
            </div>
            <div class="stat-box">
              <div class="stat-value">4 <span class="unit">个</span></div>
              <div class="stat-label">递进实验模型</div>
            </div>
            <div class="stat-box">
              <div class="stat-value">6 <span class="unit">类</span></div>
              <div class="stat-label">覆盖交通方式</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" style="font-size: 20px; margin-top: 8px;">2007-2012</div>
              <div class="stat-label">数据时间跨度</div>
            </div>
          </div>
        </div>
      </template>
    </el-skeleton>

    <div class="data-source-trigger" v-if="!loading">
      <div class="trigger-dot">数据说明</div>
      <div class="source-tooltip">
        数据基于 GeoLife (2008-2012) 真实轨迹点聚合。<br>
        绿色出行 = 步行 + 骑行。<br>
        模型评测指标来自独立测试集结果。无模拟数据。
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import * as echarts from 'echarts'
import * as turf from '@turf/turf' 

const router = useRouter()
const loading = ref(true)

const totalSamples = ref(0)
const animatedCarbonKg = ref(0)

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
const chart4Ref = ref(null)
const charts = []

const districtStats = {}

const loadData = async () => {
  try {
    const [resMode, resRegion, geoRes] = await Promise.all([
      fetch('/homepage_data.json').then(r => r.json()),
      fetch('/region_data.json').then(r => r.json()),
      fetch('https://geo.datav.aliyun.com/areas_v3/bound/110000_full.json').then(r => r.json())
    ])

    echarts.registerMap('BJ', geoRes)
    loading.value = false
    await nextTick()
    
    animateValue(animatedCarbonKg, 452.8, 2500)
    initCharts(resMode.mode_distribution, resRegion)

  } catch (err) {
    console.error('加载面板数据失败:', err)
    loading.value = false
  }
}

const initCharts = (modeDistribution, regionData) => {
  // --- 卡片1：环形图 ---
  if (chart1Ref.value && modeDistribution) {
    const chart1 = echarts.init(chart1Ref.value)
    const modeColors = { '步行': '#00FF88', '骑行': '#00FFFF', '公交': '#FFDD00', '地铁': '#CC33FF', '火车': '#FF6600', '小汽车': '#FF0033' }
    const dataList = Object.entries(modeDistribution).map(([name, value]) => ({ name, value, itemStyle: { color: modeColors[name] } }))
    
    chart1.setOption({
      tooltip: { trigger: 'item', backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff', fontWeight: 'bold' } },
      series: [{
        type: 'pie', radius: ['55%', '75%'], center: ['50%', '50%'],
        itemStyle: { borderColor: 'rgba(15, 25, 45, 0.8)', borderWidth: 2 },
        label: { show: true, formatter: '{b}\n{d}%', color: '#e2e8f0', fontSize: 11, fontWeight: 'bold' },
        labelLine: { length: 8, length2: 10, lineStyle: { color: 'rgba(255,255,255,0.4)' } },
        data: dataList
      }]
    })
    totalSamples.value = dataList.reduce((a, b) => a + b.value, 0)
    charts.push(chart1)
  }

  // --- 卡片2：轨迹覆盖行政区划热力图 ---
  if (chart2Ref.value && regionData) {
    const chart2 = echarts.init(chart2Ref.value)
    
const allDistricts = [
  '东城区', '西城区', '朝阳区', '海淀区', '丰台区', '石景山区',
  '通州区', '大兴区', '昌平区', '顺义区', '房山区', '门头沟区',
  '怀柔区', '密云区', '平谷区', '延庆区'
];
const mapData = allDistricts.map(name => ({
  name: name,
  value: regionData[name] ? regionData[name].pointCount : 0
}));
    
    const validCounts = mapData.map(d => d.value).filter(v => v > 0).sort((a, b) => a - b)
    const p80Index = Math.floor(validCounts.length * 0.80)
    const optimalMax = validCounts[p80Index] || 100 

    chart2.setOption({
      backgroundColor: 'transparent', 
      tooltip: { 
        trigger: 'item', 
        backgroundColor: 'rgba(10,14,23,0.95)', 
        borderColor: '#38bdf8', borderWidth: 1,
        textStyle: { color: '#fff' },
        formatter: (params) => {
          if (params.seriesType === 'map' && params.name) {
            const val = params.value || 0
            return `<div style="font-size:15px; font-weight:bold; margin-bottom:6px;">${params.name}</div>
                    <div style="font-weight:bold; color:#00FFFF;">真实轨迹点: ${val.toLocaleString()}</div>`
          }
          return ''
        }
      },
visualMap: {
  show: true,
  type: 'continuous',
  min: 0,
  max: optimalMax,
  left: 15,
  bottom: 15,
  text: ['极密集', '低覆盖'],
  textStyle: { color: '#e2e8f0', fontSize: 10, fontWeight: 'bold' },
inRange: {
color: ['#A8E6CF', '#FFE082', '#FFC107', '#FF9800', '#F44336', '#D32F2F']
},
  outOfRange: {
    color: '#0A0E17'  // 无数据区域也显示科技蓝
  }
},
      series: [{
        type: 'map', map: 'BJ', roam: true, zoom: 1.1, center: [116.405285, 40.003989],
        label: { show: true, fontSize: 11, fontWeight: 'bold', color: '#fff', textShadow: '0 0 4px #000' },
        emphasis: { 
          label: { color: '#fff' }, 
          itemStyle: { areaColor: '#00FFFF', borderColor: '#fff', borderWidth: 2, shadowColor: 'rgba(0,255,255,0.8)', shadowBlur: 10 } 
        },
        itemStyle: { borderColor: 'rgba(56,189,248,0.3)', borderWidth: 1 },
        data: mapData
      }]
    })
    
    chart2.on('click', (params) => {
      if (params.seriesType === 'map' && params.name) {
        router.push(`/region/${params.name}`);
      }
    });
    charts.push(chart2)
  }

  // --- 卡片4：绿色出行 TOP5 区域  ---
  if (chart4Ref.value && regionData) {
    const chart4 = echarts.init(chart4Ref.value)
    
    const top5Data = Object.entries(regionData)
      .sort((a, b) => b[1].greenRatio - a[1].greenRatio)
      .slice(0, 5)
      .reverse()

    chart4.setOption({
      grid: { top: 10, bottom: 20, left: 10, right: 40, containLabel: true },
      tooltip: { 
        trigger: 'axis', axisPointer: { type: 'shadow' }, 
        backgroundColor: 'rgba(15,25,45,0.9)', textStyle: { color: '#fff', fontWeight: 'bold' },
        formatter: (params) => {
           const district = params[0].name
           return `${district}<br/>真实绿色出行比例: ${(params[0].value * 100).toFixed(1)}%`
        }
      },
      xAxis: { type: 'value', max: 1, axisLabel: { show: false }, splitLine: { show: false } },
      yAxis: { 
        type: 'category', data: top5Data.map(d => d[0]), 
        axisLabel: { color: '#e2e8f0', fontSize: 12, fontWeight: '600' }, 
        axisLine: { show: false }, axisTick: { show: false } 
      },
      series: [{
        type: 'bar', data: top5Data.map(d => d[1].greenRatio), barWidth: 14,
        label: { show: true, position: 'right', color: '#00FF88', fontWeight: 'bold', formatter: (p) => (p.value * 100).toFixed(1)+'%', fontSize: 12 },
        itemStyle: {
          borderRadius: [0, 8, 8, 0],
          color: new echarts.graphic.LinearGradient(1, 0, 0, 0, [{ offset: 0, color: '#00FF88' }, { offset: 1, color: '#00A8FF' }])
        }
      }]
    })
    charts.push(chart4)
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
.homepage-container { 
  width: 100%; 
  height: 100%; 
  box-sizing: border-box; 
  background: transparent; 
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif; 
}

.dashboard-grid {
  display: grid;
  grid-template-columns: 1.2fr 2.5fr 1.2fr;
  gap: 20px;
  margin-bottom: 20px;
  align-items: stretch;
}
.left-col, .right-col { display: flex; flex-direction: column; }

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
.glass-card:hover { box-shadow: 0 0 8px rgba(56, 189, 248, 0.6); border-color: rgba(56, 189, 248, 0.6); }

.card-title { font-size: 16px; color: #e2e8f0; margin-bottom: 12px; font-weight: 600; letter-spacing: 0.5px; }
.chart-container { flex: 1; min-height: 200px; width: 100%; }
.relative-box { position: relative; flex: 1; display: flex; flex-direction: column; }

.map-card { padding: 10px; animation: borderGlow 4s infinite alternate; }
.map-wrapper { border-radius: 8px; overflow: hidden; height: 100%; min-height: 480px; display: flex; flex-direction: column; }
.map-canvas { flex: 1; width: 100%; cursor: pointer; }

@keyframes borderGlow {
  0% { box-shadow: 0 0 5px rgba(56,189,248,0.2); }
  100% { box-shadow: 0 0 15px rgba(56,189,248,0.6); border-color: rgba(56,189,248,0.6); }
}

.ring-center { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; pointer-events: none; z-index: 10; }
.center-label { font-size: 12px; color: #cbd5e1; font-weight: 500;}
.center-value { font-size: 26px; color: #38bdf8; font-weight: bold; text-shadow: 0 0 8px rgba(56, 189, 248, 0.5); font-family: 'Din', 'Arial', sans-serif; }

.custom-legend { position: absolute; bottom: 10px; right: 15px; background: rgba(10, 14, 23, 0.85); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 6px; padding: 8px 12px; z-index: 10; backdrop-filter: blur(4px); pointer-events: none; }
.legend-note { font-size: 11px; color: #cbd5e1; line-height: 1.6; font-weight: 500; }
.highlight-text { color: #fff; font-weight: bold; }
.sub-note { font-size: 11px; display: inline-block; margin-top: 4px; }

.flex-col-card { display: flex; flex-direction: column; justify-content: space-between; }
.highlights-list { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 14px; }
.highlight-item { font-size: 13px; color: #e2e8f0; font-weight: 500; }
.highlight-ratio { color: #FF4500; font-weight: bold; font-size: 14px; margin-right: 6px; background: rgba(255,69,0,0.1); padding: 2px 6px; border-radius: 4px;}

.model-features { display: flex; flex-direction: column; gap: 14px; flex: 1; justify-content: center;}
.feature-item { padding: 8px 12px; border-left: 3px solid rgba(56, 189, 248, 0.6); background: rgba(0,0,0,0.2); border-radius: 4px;}
.model-name { display: block; font-weight: bold; font-size: 13px; margin-bottom: 4px; }
.model-desc { color: #94a3b8; font-size: 11px; }

.carbon-panel { background: rgba(0, 0, 0, 0.3); border-radius: 8px; padding: 10px 8px; display: flex; align-items: center; justify-content: center; margin-top: 15px; border: 1px dashed rgba(0, 255, 136, 0.3);}
.carbon-text { font-size: 13px; color: #e2e8f0; font-weight: 500;}
.num-glow { font-size: 22px; font-weight: bold; color: #00FF88; text-shadow: 0 0 8px rgba(0, 255, 136, 0.4); font-family: 'Din', 'Arial', sans-serif; margin: 0 6px; }

.table-card { margin-top: 20px; }
.stats-row { display: flex; justify-content: space-around; align-items: center; padding: 10px 0; }
.stat-box { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 15px; background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; flex: 1; margin: 0 10px; transition: transform 0.3s;}
.stat-box:hover { transform: translateY(-3px); background: rgba(56, 189, 248, 0.1); }
.stat-value { font-size: 28px; font-weight: bold; color: #38bdf8; font-family: 'Din', 'Arial', sans-serif; text-shadow: 0 0 10px rgba(56,189,248,0.4); }
.stat-value .unit { font-size: 14px; color: #94a3b8; font-weight: normal; }
.stat-label { font-size: 13px; color: #cbd5e1; font-weight: bold; }

.data-source-trigger { position: fixed; bottom: 25px; right: 25px; z-index: 100; }
.trigger-dot { background: rgba(15, 25, 45, 0.8); border: 1px solid rgba(56, 189, 248, 0.5); color: #38bdf8; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; cursor: help; backdrop-filter: blur(4px); transition: all 0.3s; }
.source-tooltip { position: absolute; bottom: 35px; right: 0; width: 260px; background: rgba(0, 0, 0, 0.85); color: #fff; padding: 12px; border-radius: 8px; font-size: 12px; font-weight: 500; line-height: 1.6; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); opacity: 0; visibility: hidden; transform: translateY(10px); transition: all 0.3s; }
.data-source-trigger:hover .source-tooltip { opacity: 1; visibility: visible; transform: translateY(0); }
.data-source-trigger:hover .trigger-dot { background: #38bdf8; color: #0f192d; }

.fade-up { opacity: 0; animation: fadeInUp 0.5s ease forwards; }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.card-delay-1 { animation-delay: 0.1s; } .card-delay-2 { animation-delay: 0.2s; } .card-delay-3 { animation-delay: 0.3s; }
.card-delay-4 { animation-delay: 0.4s; } .card-delay-5 { animation-delay: 0.5s; } .card-delay-6 { animation-delay: 0.6s; }

@media (max-width: 1024px) {
  .dashboard-grid { grid-template-columns: 1fr; }
  .stats-row { flex-wrap: wrap; gap: 15px; }
  .stat-box { min-width: 40%; margin: 0; }
}
</style>