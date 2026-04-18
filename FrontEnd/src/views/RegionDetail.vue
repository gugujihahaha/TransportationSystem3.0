<template>
  <div class="region-detail-container">
    <div class="breadcrumb-header">
      <router-link to="/dashboard" class="back-link">
        <span class="back-arrow">←</span> 返回全局态势
      </router-link>
      <span class="separator">/</span>
      <span class="current-page">区域绿色出行档案</span>
      <span class="separator">/</span>
      <span class="region-name">{{ regionName }}</span>
    </div>

    <div v-if="loading" class="status-container">
      <div class="loading-spinner"></div>
      <p>正在加载 {{ regionName }} 区域数据...</p>
    </div>
    
    <div v-else-if="!regionData" class="status-container">
      <div class="empty-icon">📂</div>
      <p>该区域（{{ regionName }}）暂无轨迹数据。</p>
    </div>

    <div v-else class="detail-content fade-up">
      <!-- 第一行：两个卡片左右对称 -->
      <div class="dashboard-row row-1">
        <!-- 左侧：核心指标 -->
        <div class="glass-card metrics-card">
          <div class="card-title">区域出行核心特征</div>
          <div class="metrics-grid">
            <div class="metric-item">
              <div class="metric-label">轨迹样本总量</div>
              <div class="metric-value text-blue">{{ regionData.pointCount.toLocaleString() }}</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">绿色出行比例</div>
              <div class="metric-value text-green">{{ (regionData.greenRatio * 100).toFixed(1) }}%</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">全市绿色排名</div>
              <div class="metric-value text-orange">TOP {{ regionData.rank }}</div>
            </div>
          </div>
        </div>

        <!-- 右侧：对比卡片 + 建议 -->
        <div class="glass-card compare-card">
          <div class="card-title">绿色出行水平对比</div>
          <div class="compare-content">
            <div class="compare-item">
              <span class="compare-label">本区绿色比例</span>
              <span class="compare-value">{{ (regionData.greenRatio * 100).toFixed(1) }}%</span>
            </div>
            <div class="compare-item">
              <span class="compare-label">全市平均绿色比例</span>
              <span class="compare-value">{{ (avgGreenRatio * 100).toFixed(1) }}%</span>
            </div>
            <div class="compare-diff" :class="diffClass">
              {{ diffText }}
            </div>
            <div class="suggestion-box">
              <span class="suggestion-icon">💡</span>
              <span class="suggestion-label">建议</span>
              <span class="suggestion-text">{{ suggestionText }}</span>
            </div>
            <div class="data-note">
              * 全市平均基于所有区县真实数据计算<br>
              * 绿色出行 = 步行 + 骑行
            </div>
          </div>
        </div>
      </div>

      <!-- 第二行：绿色出行热点网格表格（全宽） -->
      <div class="dashboard-row row-2">
        <div class="glass-card table-card full-width">
          <div class="card-title">🌿 区域内绿色出行热点网格</div>
          <el-table 
            :data="regionData.topGreenGrids" 
            style="width: 100%" 
            class="cyber-table"
            :row-style="{ background: 'transparent' }"
            :cell-style="{ borderBottom: '1px solid rgba(255,255,255,0.08)' }"
            :header-cell-style="{ background: 'rgba(56,189,248,0.1)', color: '#e2e8f0', fontWeight: 'bold', borderBottom: '1px solid rgba(56,189,248,0.3)' }"
          >
            <el-table-column type="index" label="排位" width="80" align="center">
              <template #default="scope">
                <span :class="['rank-badge', `rank-${scope.$index + 1}`]">{{ scope.$index + 1 }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="name" label="网格/核心地标名称" />
            <el-table-column prop="ratio" label="绿色出行占比" align="right">
              <template #default="scope">
                <span class="text-highlight" style="color: #00FF88;">{{ scope.row.ratio }}</span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const regionName = ref(route.params.name || '海淀区')
const loading = ref(true)
const regionData = ref(null)
const allRegionData = ref({})

// 计算全市平均绿色比例（基于所有区真实数据）
const avgGreenRatio = computed(() => {
  const values = Object.values(allRegionData.value).map(r => r.greenRatio)
  if (values.length === 0) return 0
  return values.reduce((a, b) => a + b, 0) / values.length
})

// 对比文字和样式
const diffText = computed(() => {
  if (!regionData.value) return ''
  const diff = regionData.value.greenRatio - avgGreenRatio.value
  if (diff > 0.05) return `高于全市平均 ${(diff * 100).toFixed(1)}%`
  if (diff < -0.05) return `低于全市平均 ${(-diff * 100).toFixed(1)}%`
  return '接近全市平均水平'
})
const diffClass = computed(() => {
  if (!regionData.value) return ''
  const diff = regionData.value.greenRatio - avgGreenRatio.value
  if (diff > 0.05) return 'higher'
  if (diff < -0.05) return 'lower'
  return 'equal'
})

// 基于对比结果给出建议（合理推论）
const suggestionText = computed(() => {
  if (!regionData.value) return ''
  const diff = regionData.value.greenRatio - avgGreenRatio.value
  if (diff > 0.05) {
    return '该区绿色出行比例高于全市平均，建议继续保持并推广慢行交通设施。'
  } else if (diff < -0.05) {
    return '该区绿色出行比例低于全市平均，建议优化公交接驳、增加非机动车道，提升绿色出行吸引力。'
  } else {
    return '该区绿色出行比例与全市平均持平，可进一步挖掘潜在绿色出行需求，如增设共享单车停放点。'
  }
})

const loadData = async () => {
  try {
    loading.value = true
    const response = await fetch('/region_data.json')
    if (response.ok) {
      allRegionData.value = await response.json()
      regionData.value = allRegionData.value[regionName.value] || null
    }
  } catch (err) {
    console.error('加载区域数据失败:', err)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.region-detail-container { min-height: calc(100vh - 64px); color: #e2e8f0; padding: 0 10px; box-sizing: border-box; }
.breadcrumb-header { display: flex; align-items: center; font-size: 14px; margin-bottom: 24px; color: #94a3b8; }
.back-link { color: #38bdf8; text-decoration: none; display: flex; align-items: center; }
.separator { margin: 0 10px; color: #475569; }
.region-name { color: #fff; font-weight: 600; margin-left: 10px; padding-left: 10px; border-left: 2px solid #38bdf8; }

.status-container { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 400px; color: #94a3b8; }
.loading-spinner { width: 40px; height: 40px; border: 4px solid rgba(56, 189, 248, 0.2); border-top-color: #38bdf8; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 16px; }
@keyframes spin { to { transform: rotate(360deg); } }

.detail-content { display: flex; flex-direction: column; gap: 20px; }
.dashboard-row { display: grid; gap: 20px; }
.row-1 { grid-template-columns: 1fr 1fr; }
.row-2 { grid-template-columns: 1fr; }

.glass-card { background: rgba(15, 25, 45, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 16px; padding: 20px; }
.card-title { font-size: 16px; color: #e2e8f0; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; }
.card-title::before { content: ''; width: 4px; height: 16px; background: #38bdf8; margin-right: 8px; border-radius: 2px; }

.metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
.metric-item { background: rgba(0, 0, 0, 0.3); border-radius: 12px; padding: 20px 10px; text-align: center; }
.metric-label { font-size: 13px; color: #94a3b8; margin-bottom: 10px; }
.metric-value { font-size: 32px; font-weight: bold; font-family: 'Din', sans-serif; }
.text-blue { color: #38bdf8; }
.text-green { color: #00FF88; }
.text-orange { color: #facc15; }

.compare-card .compare-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.compare-item {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px dashed rgba(255,255,255,0.1);
  padding-bottom: 8px;
}
.compare-label {
  color: #94a3b8;
}
.compare-value {
  font-weight: bold;
  color: #e2e8f0;
}
.compare-diff {
  margin-top: 4px;
  text-align: center;
  font-weight: bold;
  padding: 6px;
  border-radius: 4px;
}
.higher {
  background: rgba(0, 255, 136, 0.15);
  color: #00FF88;
}
.lower {
  background: rgba(255, 69, 0, 0.15);
  color: #FF4500;
}
.equal {
  background: rgba(255, 255, 255, 0.1);
  color: #FFD700;
}
.suggestion-box {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  padding: 8px 10px;
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px;
  font-size: 13px;
}
.suggestion-icon {
  font-size: 16px;
}
.suggestion-label {
  font-weight: bold;
  color: #38bdf8;
}
.suggestion-text {
  color: #cbd5e1;
  line-height: 1.5;
}
.data-note {
  font-size: 11px;
  color: #64748b;
  margin-top: 12px;
  text-align: center;
}
.full-width {
  width: 100%;
}
.cyber-table { background: transparent !important; font-size: 14px; }
.rank-badge { display: inline-block; width: 24px; height: 24px; line-height: 24px; border-radius: 4px; background: rgba(255,255,255,0.1); text-align: center; }
.rank-1 { background: #00FF88; color: #000; }
.rank-2 { background: #00A8FF; }
.rank-3 { background: #FFD700; color: #000; }

.fade-up { animation: fadeInUp 0.5s ease forwards; }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
</style>