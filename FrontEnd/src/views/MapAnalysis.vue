<template>
  <div class="map-analysis">
    <div class="sidebar left-sidebar">
      <div class="sidebar-section">
        <h3><span class="neon-line"></span> 轨迹数据导入</h3>
        <TrajectoryUpload @upload="handleUpload" />
      </div>

      <div class="sidebar-section">
        <h3><span class="neon-line"></span> 交通模态图例</h3>
        <div class="mode-filters">
          <div
            v-for="mode in transportModes"
            :key="mode.id"
            class="mode-filter-item"
            :class="{ active: modeFilters[mode.id] }"
            @click="toggleModeFilter(mode.id)"
          >
            <div class="mode-color-dot" :style="{ background: mode.color, boxShadow: `0 0 10px ${mode.color}` }"></div>
            <span class="mode-name">{{ mode.name }}</span>
            <span class="mode-count">{{ getModeCount(mode.id) }}</span>
          </div>
        </div>
      </div>

      <div v-if="selectedTrajectory" class="sidebar-section">
        <h3><span class="neon-line"></span> 轨迹特征参数</h3>
        <div class="trajectory-details glass-panel">
          <div class="detail-item">
            <span class="detail-label">最终收敛分类：</span>
            <ModeTag :mode="selectedTrajectory.predicted_mode" />
          </div>
          <div class="detail-item">
            <span class="detail-label">置信度 (Confidence)：</span>
            <div class="progress-wrapper">
              <el-progress
                :percentage="selectedTrajectory.confidence * 100"
                :color="getConfidenceColor(selectedTrajectory.confidence)"
                :stroke-width="8"
                :show-text="false"
              />
              <span class="progress-text" :style="{ color: getConfidenceColor(selectedTrajectory.confidence) }">
                {{ (selectedTrajectory.confidence * 100).toFixed(1) }}%
              </span>
            </div>
          </div>
          <div class="detail-item">
            <span class="detail-label">欧氏距离：</span>
            <span class="detail-value tech-font">{{ formatDistance(selectedTrajectory.stats.distance) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">时空跨度：</span>
            <span class="detail-value tech-font">{{ formatDuration(selectedTrajectory.stats.duration) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">均值速率：</span>
            <span class="detail-value tech-font">{{ formatSpeed(selectedTrajectory.stats.avg_speed) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">峰值速率：</span>
            <span class="detail-value tech-font">{{ formatSpeed(selectedTrajectory.stats.max_speed) }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="map-container">
      <MapView
        :trajectories="filteredPredictions"
        :selected-trajectory="selectedTrajectory"
        @select="handleSelectTrajectory"
      />

      <div v-show="selectedTrajectory" class="right-floating-island">
        <AIAnalysisReport ref="aiReportRef" />
      </div>

      <div v-if="selectedTrajectory" class="timeline-panel floating-island">
        <div class="timeline-header">
          <h4><span class="status-indicator"></span> 时空轨迹回溯</h4>
          <el-button text class="primary-action-btn" @click="showTimeline = !showTimeline">
            <el-icon><component :is="showTimeline ? ArrowDown : ArrowUp" /></el-icon>
          </el-button>
        </div>
        <div v-if="showTimeline" class="timeline-content">
          <el-slider
            v-model="timelineProgress"
            :min="0"
            :max="100"
            :step="1"
            show-tooltip
            class="cyber-slider"
            @change="handleTimelineChange"
          />
          <div class="timeline-info">
            <span class="tech-time">{{ formatTimelineTime(timelineProgress) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { ArrowUp, ArrowDown } from '@element-plus/icons-vue'
import { useTrajectoryStore } from '@/stores/trajectory'
import MapView from '@/components/MapView.vue'
import TrajectoryUpload from '@/components/TrajectoryUpload.vue'
import ModeTag from '@/components/ModeTag.vue'
import AIAnalysisReport from '@/components/AIAnalysisReport.vue'

const trajectoryStore = useTrajectoryStore()

const transportModes = computed(() => trajectoryStore.transportModes)
const predictions = computed(() => trajectoryStore.predictions)
const selectedTrajectory = computed(() => trajectoryStore.selectedTrajectory)

const aiReportRef = ref<any>(null) 
const modeFilters = ref<Record<string, boolean>>({
  walk: true, bike: true, bus: true, car: true, subway: true, train: true, airplane: true,
})
const showTimeline = ref(false)
const timelineProgress = ref(0)

const filteredPredictions = computed(() => {
  return predictions.value.filter(pred => modeFilters.value[pred.predicted_mode])
})

onMounted(() => {
  trajectoryStore.loadTransportModes()
})

async function handleUpload(file: File, model: string) {
  try {
    await trajectoryStore.predictTrajectory(file, model)
    ElMessage.success('预测完成，正在生成分析报告...')
    
    if (aiReportRef.value) {
      aiReportRef.value.generateReport()
    }
  } catch (error) {
    ElMessage.error('预测失败，请检查数据格式')
  }
}

function handleSelectTrajectory(trajectory: any) {
  trajectoryStore.selectTrajectory(trajectory)
}
function toggleModeFilter(mode: string) {
  modeFilters.value[mode] = !modeFilters.value[mode]
}
function getModeCount(mode: string): number {
  return filteredPredictions.value.filter(p => p.predicted_mode === mode).length
}
function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return '#00f0ff'
  if (confidence >= 0.6) return '#E6A23C'
  return '#F56C6C'
}
function formatDistance(meters: number): string {
  return meters < 1000 ? `${meters.toFixed(0)} m` : `${(meters / 1000).toFixed(2)} km`
}
function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600), minutes = Math.floor((seconds % 3600) / 60), secs = Math.floor(seconds % 60)
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`
  if (minutes > 0) return `${minutes}m ${secs}s`
  return `${secs}s`
}
function formatSpeed(mps: number): string { return `${mps.toFixed(2)} m/s` }
function handleTimelineChange(value: number) { console.log('Timeline progress:', value) }
function formatTimelineTime(progress: number): string {
  if (!selectedTrajectory.value || selectedTrajectory.value.points.length === 0) return '0%'
  const index = Math.floor((progress / 100) * (selectedTrajectory.value.points.length - 1))
  const point = selectedTrajectory.value.points[index]
  return point ? new Date(point.timestamp).toLocaleTimeString('zh-CN') : '0%'
}
</script>

<style scoped>
.map-analysis {
  display: flex;
  flex: 1;
  min-height: 0;
  background: #050608;
}

.left-sidebar {
  width: 320px;
  background: rgba(10, 12, 16, 0.85);
  backdrop-filter: blur(20px);
  border-right: 1px solid rgba(0, 240, 255, 0.1);
  box-shadow: 5px 0 30px rgba(0, 0, 0, 0.5);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px;
  z-index: 10;
}
.left-sidebar::-webkit-scrollbar { width: 4px; }
.left-sidebar::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.3); border-radius: 4px; }

.sidebar-section h3 {
  margin: 0 0 16px 0;
  font-size: 15px;
  font-weight: 600;
  color: #fff;
  display: flex;
  align-items: center;
  gap: 8px;
}
.neon-line {
  width: 3px; height: 14px;
  background: #00f0ff;
  box-shadow: 0 0 8px #00f0ff;
}

.mode-filters { display: flex; flex-direction: column; gap: 8px; }
.mode-filter-item {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px; border-radius: 8px;
  cursor: pointer; transition: all 0.3s;
  background: rgba(255, 255, 255, 0.02); border: 1px solid transparent;
}
.mode-filter-item:hover { transform: translateX(4px); background: rgba(255, 255, 255, 0.05); }
.mode-filter-item.active {
  background: linear-gradient(90deg, rgba(0, 240, 255, 0.1), transparent);
  border-color: rgba(0, 240, 255, 0.3);
}
.mode-color-dot { width: 10px; height: 10px; border-radius: 50%; }
.mode-name { flex: 1; font-size: 13px; color: #e5e8eb; }
.mode-count { font-size: 12px; color: #00f0ff; font-family: 'Consolas', monospace; }

.glass-panel {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 12px; padding: 16px;
  display: flex; flex-direction: column; gap: 12px;
}
.detail-item {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 13px; padding-bottom: 8px; border-bottom: 1px dashed rgba(255,255,255,0.05);
}
.detail-item:last-child { border-bottom: none; padding-bottom: 0; }
.detail-label { color: #8b949e; }
.detail-value { color: #e6edf3; font-weight: bold; }
.tech-font { font-family: 'Consolas', monospace; color: #a5d6ff; }
.progress-wrapper { display: flex; align-items: center; gap: 10px; width: 120px; }
:deep(.el-progress) { flex: 1; }
.progress-text { font-family: 'Consolas', monospace; font-size: 12px; font-weight: bold; }

.map-container { flex: 1; position: relative; overflow: hidden; }

.right-floating-island {
  position: absolute;
  top: 24px;
  right: 24px;
  width: 420px;
  z-index: 20;
  animation: slideInRight 0.5s ease forwards;
}

@keyframes slideInRight {
  from { transform: translateX(100px); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

.floating-island {
  position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
  width: 60%; min-width: 400px;
  background: rgba(10, 12, 16, 0.85); backdrop-filter: blur(16px);
  border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 16px;
  padding: 16px 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); z-index: 20;
}
.timeline-header { display: flex; justify-content: space-between; align-items: center; }
.timeline-header h4 { margin: 0; font-size: 14px; color: #e5e8eb; display: flex; align-items: center; gap: 8px; }
.status-indicator { width: 8px; height: 8px; background: #00f0ff; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.5); opacity: 0.5; } }
:deep(.el-slider__bar) { background: linear-gradient(90deg, #4a90e2, #00f0ff) !important; }
:deep(.el-slider__button) { border: 2px solid #00f0ff !important; background: #050608 !important; }
.timeline-info { text-align: center; margin-top: -4px; }
.tech-time { font-family: 'Consolas', monospace; font-size: 13px; color: #00f0ff; }
</style>