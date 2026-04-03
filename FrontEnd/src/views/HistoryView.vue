<template>
  <div class="fui-history-container">
    <div class="header-section">
      <h2 class="main-title">
        <span class="text-glow">历史轨迹分析档案</span>
      </h2>
      <p class="subtitle">全量时空识别记录回溯与多模态审计结果查询</p>
      
      <button class="cyber-btn sync-btn" @click="refreshData" :disabled="loading">
        <span v-if="loading" class="loading-spinner"></span>
        <span v-else>🔄 同步最新数据流</span>
      </button>
    </div>

    <div class="content-panel panel-glass">
      <div v-if="loading" class="status-box text-cyan">
        <div class="loading-ring"></div>
        <p>读取历史档案中...</p>
      </div>

      <div v-else-if="trajectoryStore.history.length === 0" class="status-box text-slate">
        <div class="empty-icon">📂</div>
        <p>暂无任何轨迹分析档案</p>
        <button class="cyber-btn primary mt-4" @click="router.push('/dashboard')">前往发起在线预测</button>
      </div>

      <div v-else class="table-wrapper">
        <table class="history-table">
          <thead>
            <tr>
              <th>档案生成时间</th>
              <th>系统判定方式</th>
              <th>模型置信度</th>
              <th class="text-right">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(record, index) in currentRecords" :key="index">
              <td class="font-mono text-sm text-slate-300">
                {{ formatDate(record.created_at) }}
              </td>
              
              <td>
                <span class="mode-badge" :style="getBadgeStyle(getRecordMode(record))">
                  <span class="badge-icon">{{ getBadgeIcon(getRecordMode(record)) }}</span>
                  <span class="badge-text uppercase">{{ getRecordMode(record) }}</span>
                </span>
              </td>
              
              <td>
                <div class="confidence-wrapper">
                  <span class="text-cyan font-bold w-12">{{ getRecordConfidence(record) }}</span>
                  <div class="progress-track">
                    <div class="progress-bar" :style="{ width: getRecordConfidence(record) }"></div>
                  </div>
                </div>
              </td>
              
              <td class="text-right">
                <button class="view-btn" @click="goToDetail(record)">>调阅详情</button>
              </td>
            </tr>
          </tbody>
        </table>

        <div class="pagination-controls" v-if="totalPages > 1">
          <span class="page-info text-slate-400">
            共 <span class="text-cyan font-bold">{{ trajectoryStore.history.length }}</span> 份档案，
            当前第 {{ currentPage }} / {{ totalPages }} 页
          </span>
          <div class="btn-group">
            <button class="page-btn" :disabled="currentPage === 1" @click="currentPage--">上一页</button>
            <button class="page-btn" :disabled="currentPage === totalPages" @click="currentPage++">下一页</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTrajectoryStore } from '@/stores/trajectory'

const router = useRouter()
const trajectoryStore = useTrajectoryStore()

const loading = ref(false)
const currentPage = ref(1)
const ITEMS_PER_PAGE = 10

const refreshData = async () => {
  loading.value = true
  await trajectoryStore.fetchHistory()
  loading.value = false
}

onMounted(async () => {
  if (trajectoryStore.history.length === 0) {
    await refreshData()
  }
})

// 分页逻辑
const totalPages = computed(() => Math.ceil(trajectoryStore.history.length / ITEMS_PER_PAGE))
const currentRecords = computed(() => {
  const start = (currentPage.value - 1) * ITEMS_PER_PAGE
  const end = currentPage.value * ITEMS_PER_PAGE
  return trajectoryStore.history.slice(start, end)
})

// 防御性取值函数
const getRecordMode = (record: any): string => String(record.pred_label || record.mode || record.predicted_mode || 'Unknown')
const getRecordConfidence = (record: any): string => {
  const conf = record.confidence || record.prob || 0
  return (Number(conf) * 100).toFixed(0) + '%'
}

// 时间格式化
const formatDate = (dateString?: string) => {
  if (!dateString) return '未知时间'
  const date = new Date(dateString)
  return isNaN(date.getTime()) ? '未知时间' : date.toLocaleString('zh-CN', { hour12: false })
}

// 智能跳转：带有 record.id 回溯对应场景
const goToDetail = (record: any) => {
  // 根据后端的 scene 字段分发路由
  if (record.scene === 'green') {
    router.push({ path: '/green-travel', query: { id: record.id } })
  } else {
    router.push({ path: '/congestion-analysis', query: { id: record.id } })
  }
}

// 徽章样式映射 
const getBadgeStyle = (mode: string) => {
  const m = mode.toLowerCase()
  if (m.includes('walk')) return { background: 'rgba(16, 185, 129, 0.2)', color: '#10b981', border: '1px solid rgba(16, 185, 129, 0.5)' }
  if (m.includes('bike')) return { background: 'rgba(59, 130, 246, 0.2)', color: '#3b82f6', border: '1px solid rgba(59, 130, 246, 0.5)' }
  if (m.includes('bus')) return { background: 'rgba(249, 115, 22, 0.2)', color: '#f97316', border: '1px solid rgba(249, 115, 22, 0.5)' }
  if (m.includes('subway')) return { background: 'rgba(168, 85, 247, 0.2)', color: '#a855f7', border: '1px solid rgba(168, 85, 247, 0.5)' }
  if (m.includes('train')) return { background: 'rgba(99, 102, 241, 0.2)', color: '#6366f1', border: '1px solid rgba(99, 102, 241, 0.5)' }
  if (m.includes('car')) return { background: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.5)' }
  return { background: 'rgba(148, 163, 184, 0.2)', color: '#94a3b8', border: '1px solid rgba(148, 163, 184, 0.5)' }
}

const getBadgeIcon = (mode: string) => {
  const m = mode.toLowerCase()
  if (m.includes('walk')) return '🚶'
  if (m.includes('bike')) return '🚴'
  if (m.includes('bus')) return '🚌'
  if (m.includes('subway')) return '🚇'
  if (m.includes('train')) return '🚆'
  if (m.includes('car')) return '🚗'
  return '❓'
}
</script>

<style scoped>
.fui-history-container {
  padding: 40px 5%;
  min-height: 85vh;
  color: #fff;
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif;
}

/* 顶部标题区 */
.header-section {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-bottom: 30px;
  position: relative;
  border-bottom: 2px solid rgba(0, 240, 255, 0.2);
  padding-bottom: 20px;
}
.main-title { font-size: 2rem; font-weight: 900; letter-spacing: 2px; margin-bottom: 10px; }
.text-glow { color: #00f0ff; text-shadow: 0 0 15px rgba(0, 240, 255, 0.4); }
.subtitle { color: #94a3b8; letter-spacing: 1px; }
.sync-btn {
  position: absolute; right: 0; bottom: 20px;
  background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.4);
  color: #00f0ff; padding: 10px 20px; border-radius: 4px; font-weight: bold;
  cursor: pointer; transition: all 0.3s;
}
.sync-btn:hover:not(:disabled) { background: rgba(0, 240, 255, 0.2); box-shadow: 0 0 15px rgba(0, 240, 255, 0.2); }

/* 面板与状态 */
.panel-glass {
  background: rgba(13, 20, 40, 0.5); border: 1px solid rgba(0, 240, 255, 0.15);
  backdrop-filter: blur(12px); border-radius: 12px; padding: 30px; min-height: 400px;
}
.status-box { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; font-size: 1.2rem; letter-spacing: 2px; }
.empty-icon { font-size: 4rem; opacity: 0.5; margin-bottom: 20px; }
.text-cyan { color: #00f0ff; }
.text-slate { color: #94a3b8; }
.loading-ring { width: 40px; height: 40px; border: 3px solid rgba(0,240,255,0.2); border-top-color: #00f0ff; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 20px; }

/* 表格与徽章 */
.table-wrapper { width: 100%; overflow-x: auto; }
.history-table { width: 100%; border-collapse: collapse; text-align: left; }
.history-table th { border-bottom: 2px solid rgba(0, 240, 255, 0.3); color: #00f0ff; padding: 15px; letter-spacing: 1px; }
.history-table td { padding: 15px; border-bottom: 1px solid rgba(0, 240, 255, 0.05); }
.history-table tr:hover td { background: rgba(0, 240, 255, 0.05); }
.mode-badge { display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: bold; }

/* 进度条 */
.confidence-wrapper { display: flex; align-items: center; gap: 15px; }
.progress-track { flex: 1; height: 6px; background: rgba(0, 0, 0, 0.5); border-radius: 3px; border: 1px solid rgba(0, 240, 255, 0.2); overflow: hidden; }
.progress-bar { height: 100%; background: linear-gradient(90deg, #0070f3, #00f0ff); transition: width 0.5s ease; }

/* 按钮与底部分页 */
.view-btn { background: rgba(0, 240, 255, 0.1); border: 1px solid #00f0ff; color: #00f0ff; padding: 6px 15px; border-radius: 4px; cursor: pointer; transition: 0.3s; }
.view-btn:hover { background: #00f0ff; color: #000; }
.cyber-btn.primary { background: rgba(0, 240, 255, 0.1); border: 1px solid #00f0ff; color: #00f0ff; padding: 10px 25px; border-radius: 4px; cursor: pointer; transition: 0.3s; }
.cyber-btn.primary:hover { background: #00f0ff; color: #000; box-shadow: 0 0 15px #00f0ff; }

.pagination-controls { display: flex; justify-content: space-between; align-items: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(0, 240, 255, 0.1); }
.btn-group { display: flex; gap: 10px; }
.page-btn { background: transparent; border: 1px solid rgba(0, 240, 255, 0.4); color: #00f0ff; padding: 6px 15px; border-radius: 4px; cursor: pointer; }
.page-btn:hover:not(:disabled) { background: rgba(0, 240, 255, 0.2); }
.page-btn:disabled { opacity: 0.3; cursor: not-allowed; border-color: #475569; color: #475569; }

@keyframes spin { 100% { transform: rotate(360deg); } }
</style>