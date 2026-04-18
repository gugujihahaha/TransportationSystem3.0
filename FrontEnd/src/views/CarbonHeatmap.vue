<template>
  <div class="cyber-battle-station">
    
    <div class="side-panel left-panel">
      <div class="cyber-card header-card">
        <div class="cyber-title-box">
          <h2 class="neon-text-green">城市碳普惠热力态势</h2>
        </div>
        <div class="sys-status blink-status-green"></div>
      </div>

      <div class="cyber-card content-card flex-col">
        <div class="module-box">
          <div class="module-header">
            <span class="bracket">[</span> <span class="neon-text-green-sub">01</span> <span class="bracket">]</span> 数据源挂载状态
          </div>
          <div class="status-panel">
            <div class="status-item">
              <span class="label">底层网格数据引擎</span>
              <span class="val text-green blink-text">Online</span>
            </div>
            <div class="status-item">
              <span class="label">当前渲染网格总数</span>
              <span class="val">{{ totalGrids }} 区块</span>
            </div>
            <div class="status-item">
              <span class="label">最高绿色出行比例</span>
              <span class="val text-blue">{{ maxRatio }}%</span>
            </div>
          </div>
        </div>

        <div class="module-box flex-1">
          <div class="module-header">
            <span class="bracket">[</span> <span class="neon-text-green-sub">02</span> <span class="bracket">]</span> 热力图例解析
          </div>
          <div class="legend-box">
            <div class="gradient-bar"></div>
            <div class="labels">
              <span>高碳区 (机动车为主)</span>
              <span>低碳区 (步骑公为主)</span>
            </div>
            <div class="desc-text mt-3">
              * 颜色越接近荧光绿，代表该 500m×500m 网格内的慢行与公共交通出行轨迹点占比越高。
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="center-panel">
      <div class="map-frame green-frame" v-loading="isLoading" element-loading-text="正在挂载碳普惠网格矩阵..." element-loading-background="rgba(2, 11, 24, 0.85)">
        <div ref="mapContainer" class="leaflet-map"></div>
        <div class="frame-corner top-left green-line"></div>
        <div class="frame-corner top-right green-line"></div>
        <div class="frame-corner bottom-left green-line"></div>
        <div class="frame-corner bottom-right green-line"></div>
      </div>
    </div>

    <div class="side-panel right-panel">
      <div class="cyber-card ai-card green-border">
        <div class="info-card glass-card">
          <div class="info-icon">🌍</div>
          <div class="info-content">
            <div class="info-title">基于轨迹数据流的宏观洞察</div>
            <div class="info-desc">本模块汇聚海量脱敏出行轨迹，利用网格化算法映射城市低碳出行活跃地带，为碳减排核算与城市慢行系统规划提供决策支持。</div>
          </div>
        </div>
        
        <div class="ai-header green-bg-soft">
          <div class="ai-title">
            <el-icon class="pulse-icon-green"><Location /></el-icon> 空间热力数据研判
          </div>
        </div>
        
        <div class="ai-body">
          <div v-if="isLoading" class="ai-processing green-text">
            <div class="spinner green-spinner"></div>
            正在解析 JSON 网格拓扑...
          </div>
          <div v-else class="ai-stream-text">
            <h3 class="ai-h3"><span class="highlight-bar-green"></span> 核心低碳聚集区</h3>
            <p>从左侧热力图可明显观测到，城市核心商圈、地铁站点周边及滨河绿道附近呈现显著的<strong class="ai-strong-green">高亮绿色斑块</strong>，表明这些区域是步行与骑行轨迹的高频聚集地。</p>
            
            <h3 class="ai-h3"><span class="highlight-bar-green"></span> 规划与决策建议</h3>
            <p>建议相关部门针对深绿色高频网格，进一步优化共享单车投放点与慢行步道建设；对于红色低谷区（高碳排区），可考虑引入“碳普惠积分”激励机制，引导居民向公共交通转移。</p>
          </div>
        </div>
      </div>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { Location } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const mapContainer = ref<HTMLElement | null>(null)
let map: L.Map | null = null
const isLoading = ref(true)
const totalGrids = ref(0)
const maxRatio = ref('0.0')

const initMap = async () => {
  if (!mapContainer.value) return
  map = L.map(mapContainer.value, { zoomControl: false }).setView([39.9042, 116.4074], 11)
  
  L.tileLayer('http://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}', {
    attribution: '© AutoNavi', maxZoom: 19 
  }).addTo(map)

  await loadHeatmapData()
}

const loadHeatmapData = async () => {
  try {
    const res = await fetch('/heatmap_grid.json')
    if (!res.ok) throw new Error('网络响应异常')
    const gridData = await res.json()
    
    totalGrids.value = gridData.length
    let highest = 0

    gridData.forEach((grid: any) => {
      if (grid.green_ratio > highest) highest = grid.green_ratio

      const r = Math.floor(255 * (1 - grid.green_ratio))
      const g = Math.floor(255 * grid.green_ratio + (100 * (1 - grid.green_ratio))) 
      const b = Math.floor(50 * (1 - grid.green_ratio))
      const color = `rgb(${r},${g},${b})`

      L.circleMarker([grid.lat, grid.lng], {
        radius: 7,
        fillColor: color,
        color: 'transparent',
        fillOpacity: 0.8
      }).addTo(map!)
        .bindPopup(`
          <div style="color: #333; font-family: monospace;">
            <strong>网格坐标:</strong> ${grid.lat}, ${grid.lng}<br>
            <strong>总轨迹点:</strong> ${grid.count}<br>
            <strong>绿色出行占比:</strong> <span style="color: green; font-weight: bold;">${(grid.green_ratio * 100).toFixed(1)}%</span>
          </div>
        `)
    })
    maxRatio.value = (highest * 100).toFixed(1)
    isLoading.value = false
    ElMessage.success('碳普惠网格数据加载完毕')
  } catch (error) {
    console.error("加载热力图数据失败", error)
    isLoading.value = false
    ElMessage.warning('未检测到 heatmap_grid.json 文件。请确保已运行后端 python 脚本生成数据。')
  }
}

onMounted(() => {
  nextTick(() => { initMap() })
})

onUnmounted(() => {
  if (map) { map.remove(); map = null }
})
</script>

<style scoped>
.cyber-battle-station {
  display: flex;
  width: 100%;
  height: calc(100vh - 64px);
  background: #0F172A;
  padding: 16px;
  gap: 16px;
  box-sizing: border-box;
  font-family: 'Helvetica Neue', Helvetica, 'Microsoft YaHei', sans-serif;
}

.side-panel { display: flex; flex-direction: column; gap: 16px; flex-shrink: 0; z-index: 10; }
.left-panel { width: 340px; }
.right-panel { width: 420px; }
.center-panel { flex: 1; display: flex; position: relative; z-index: 5;}

.cyber-card { 
  background: #1E293B; 
  border: 1px solid #334155; 
  border-radius: 8px; 
  display: flex; 
  flex-direction: column; 
  overflow: hidden; 
}

.header-card { padding: 16px 20px; display: flex; flex-direction: row; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.8); }
.neon-text-green { margin: 0; font-size: 20px; font-weight: 800; color: #4fc76dc1; text-shadow: 0 0 10px rgba(57, 255, 20, 0.4); }
.neon-text-green-sub { color: #98dc96c1; font-weight: bold; }
.sys-status { width: 10px; height: 10px; border-radius: 50%; background: #a1dc96c1; box-shadow: 0 0 10px #39FF14; }
.blink-status-green { animation: breathe-green 2s infinite; }

.content-card { padding: 20px; flex: 1; overflow-y: auto; }
.flex-col { display: flex; flex-direction: column; gap: 20px; }
.flex-1 { flex: 1; }

.module-header { font-size: 13px; color: #cdd9e5; margin-bottom: 16px; font-weight: bold; }
.bracket { color: #888; font-family: monospace; }

/* 状态面板 */
.status-panel { background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; border: 1px dashed rgba(57, 255, 20, 0.3); }
.status-item { display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 13px; }
.status-item:last-child { margin-bottom: 0; }
.status-item .label { color: #8892b0; }
.status-item .val { font-family: 'Consolas', monospace; font-weight: bold; color: #e2e8f0; }
.text-green { color: #39FF14 !important; text-shadow: 0 0 5px rgba(57,255,20,0.5); }
.text-blue { color: #00F0FF !important; text-shadow: 0 0 5px rgba(0,240,255,0.5); }
.blink-text { animation: blink 1.5s infinite; }

/* 图例 */
.legend-box { background: rgba(0,0,0,0.3); padding: 20px 15px; border-radius: 8px; }
.gradient-bar { height: 12px; border-radius: 6px; background: linear-gradient(to right, rgb(255,100,50), rgb(127,150,50), rgb(57,255,20)); margin-bottom: 10px; box-shadow: 0 0 10px rgba(57,255,20,0.2); }
.labels { display: flex; justify-content: space-between; font-size: 11px; color: #cbd5e1; font-weight: bold; }
.desc-text { font-size: 11px; color: #64748b; line-height: 1.5; margin-top: 15px; font-style: italic; }

/* 地图区 */
.map-frame {
  flex: 1; position: relative; border-radius: 12px;
  border: 2px solid rgba(57, 255, 20, 0.2);
  overflow: hidden;
  box-shadow: 0 0 20px rgba(57, 255, 20, 0.05);
}
.leaflet-map { width: 100%; height: 100%; background: #020B16; z-index: 1; }
:deep(.leaflet-popup-content-wrapper) { background: rgba(255, 255, 255, 0.95); border-radius: 4px; box-shadow: 0 3px 14px rgba(0,0,0,0.4); }

.frame-corner { position: absolute; width: 30px; height: 30px; border: 3px solid transparent; z-index: 10; pointer-events: none;}
.green-line { border-color: #a1dc96c1; }
.top-left { top: 0; left: 0; border-right: none; border-bottom: none; border-radius: 12px 0 0 0; }
.top-right { top: 0; right: 0; border-left: none; border-bottom: none; border-radius: 0 12px 0 0; }
.bottom-left { bottom: 0; left: 0; border-right: none; border-top: none; border-radius: 0 0 0 12px; }
.bottom-right { bottom: 0; right: 0; border-left: none; border-top: none; border-radius: 0 0 12px 0; }

/* 右侧 AI 框 */
.info-card { display: flex; align-items: center; gap: 12px; padding: 12px 16px; margin-bottom: 16px; background: rgba(57, 255, 20, 0.05); border-left: 4px solid #39FF14; border-radius: 8px;}
.info-icon { font-size: 28px; filter: drop-shadow(0 0 4px #39FF14); }
.info-title { font-size: 14px; font-weight: bold; color: #39FF14; margin-bottom: 4px; }
.info-desc { font-size: 12px; color: #cbd5e1; line-height: 1.5; }

.ai-header { padding: 16px 20px; border-bottom: 1px solid rgba(57, 255, 20, 0.2); }
.green-bg-soft { background: linear-gradient(90deg, rgba(57, 255, 20, 0.1), transparent); }
.ai-title { font-size: 16px; font-weight: bold; color: #39FF14; display: flex; align-items: center; gap: 8px; text-shadow: 0 0 8px rgba(57,255,20,0.4);}

.ai-body { padding: 24px; flex: 1; overflow-y: auto; }
.ai-stream-text { color: #e2e8f0; line-height: 1.8; font-size: 14px; }
.ai-h3 { color: #E6A23C; font-size: 15px; margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;}
.highlight-bar-green { display: inline-block; width: 4px; height: 14px; background: #39FF14; box-shadow: 0 0 8px #39FF14; border-radius: 2px;}
.ai-strong-green { color: #39FF14; font-weight: bold; }
.ai-stream-text p { margin-bottom: 20px; text-align: justify; }

.ai-processing { display: flex; flex-direction: column; align-items: center; margin-top: 60px; gap: 16px; font-weight: bold; letter-spacing: 1px;}
.spinner { width: 40px; height: 40px; border: 3px solid rgba(57,255,20,0.2); border-top-color: #39FF14; border-radius: 50%; animation: spin 1s linear infinite; box-shadow: 0 0 15px rgba(57,255,20,0.4); }

@keyframes breathe-green { 0%, 100% { opacity: 1; box-shadow: 0 0 10px #a1dc96c1; } 50% { opacity: 0.6; box-shadow: 0 0 20px #a1dc96f4; } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
@keyframes spin { to { transform: rotate(360deg); } }
</style>