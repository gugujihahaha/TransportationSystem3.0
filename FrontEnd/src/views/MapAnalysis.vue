<template>
  <div class="map-analysis">
    <div class="sidebar">
      <div class="sidebar-section">
        <h3>文件上传</h3>
        <TrajectoryUpload @upload="handleUpload" />
      </div>

      <div class="sidebar-section">
        <h3>交通方式图例</h3>
        <div class="mode-filters">
          <div
            v-for="mode in transportModes"
            :key="mode.id"
            class="mode-filter-item"
            :class="{ active: modeFilters[mode.id] }"
            @click="toggleModeFilter(mode.id)"
          >
            <div
              class="mode-color-dot"
              :style="{ background: mode.color }"
            ></div>
            <span class="mode-name">{{ mode.name }}</span>
            <span class="mode-count">{{ getModeCount(mode.id) }}</span>
          </div>
        </div>
      </div>

      <div v-if="selectedTrajectory" class="sidebar-section">
        <h3>轨迹详情</h3>
        <div class="trajectory-details">
          <div class="detail-item">
            <span class="detail-label">识别结果：</span>
            <ModeTag :mode="selectedTrajectory.predicted_mode" />
          </div>
          <div class="detail-item">
            <span class="detail-label">置信度：</span>
            <el-progress
              :percentage="selectedTrajectory.confidence * 100"
              :color="getConfidenceColor(selectedTrajectory.confidence)"
              :stroke-width="12"
            />
          </div>
          <div class="detail-item">
            <span class="detail-label">总距离：</span>
            <span class="detail-value">{{ formatDistance(selectedTrajectory.stats.distance) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">总时长：</span>
            <span class="detail-value">{{ formatDuration(selectedTrajectory.stats.duration) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">平均速度：</span>
            <span class="detail-value">{{ formatSpeed(selectedTrajectory.stats.avg_speed) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">最高速度：</span>
            <span class="detail-value">{{ formatSpeed(selectedTrajectory.stats.max_speed) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">起止时间：</span>
            <span class="detail-value">{{ formatTimeRange(selectedTrajectory.points) }}</span>
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
    </div>

    <div v-if="selectedTrajectory" class="timeline-panel">
      <div class="timeline-header">
        <h4>时间轴</h4>
        <el-button
          text
          @click="showTimeline = !showTimeline"
        >
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
          @change="handleTimelineChange"
        />
        <div class="timeline-info">
          <span>{{ formatTimelineTime(timelineProgress) }}</span>
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
import { TRANSPORT_MODE_COLORS, TRANSPORT_MODE_NAMES } from '@/utils/colors'

const trajectoryStore = useTrajectoryStore()

const transportModes = computed(() => trajectoryStore.transportModes)
const predictions = computed(() => trajectoryStore.predictions)
const selectedTrajectory = computed(() => trajectoryStore.selectedTrajectory)
const loading = computed(() => trajectoryStore.loading)

const modeFilters = ref<Record<string, boolean>>({
  walk: true,
  bike: true,
  bus: true,
  car: true,
  subway: true,
  train: true,
  airplane: true,
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
    ElMessage.success('预测完成！')
  } catch (error) {
    ElMessage.error('预测失败，请检查文件格式')
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
  if (confidence >= 0.8) return '#67C23A'
  if (confidence >= 0.6) return '#E6A23C'
  return '#F56C6C'
}

function formatDistance(meters: number): string {
  if (meters < 1000) return `${meters.toFixed(0)} m`
  return `${(meters / 1000).toFixed(2)} km`
}

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`
  if (minutes > 0) return `${minutes}m ${secs}s`
  return `${secs}s`
}

function formatSpeed(mps: number): string {
  return `${mps.toFixed(2)} m/s`
}

function formatTimeRange(points: any[]): string {
  if (points.length === 0) return 'N/A'
  const start = new Date(points[0].timestamp)
  const end = new Date(points[points.length - 1].timestamp)
  
  const formatDate = (date: Date) => {
    return date.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }
  
  return `${formatDate(start)} - ${formatDate(end)}`
}

function handleTimelineChange(value: number) {
  console.log('Timeline progress:', value)
}

function formatTimelineTime(progress: number): string {
  if (!selectedTrajectory.value || selectedTrajectory.value.points.length === 0) return '0%'
  const index = Math.floor((progress / 100) * (selectedTrajectory.value.points.length - 1))
  const point = selectedTrajectory.value.points[index]
  if (!point) return '0%'
  return new Date(point.timestamp).toLocaleTimeString('zh-CN')
}
</script>

<style scoped>
.map-analysis {
  display: flex;
  flex: 1;
  min-height: 0;
  background: #0f1117;
}

.sidebar {
  width: 280px;
  background: #1a1f2e;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 20px;
}

.sidebar-section {
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  padding-bottom: 20px;
}

.sidebar-section:last-child {
  border-bottom: none;
}

.sidebar-section h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
}

.mode-filters {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.mode-filter-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  background: rgba(255, 255, 255, 0.02);
}

.mode-filter-item:hover {
  background: rgba(255, 255, 255, 0.05);
}

.mode-filter-item.active {
  background: rgba(74, 144, 226, 0.15);
}

.mode-color-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
}

.mode-name {
  flex: 1;
  font-size: 14px;
  color: #e5e8eb;
}

.mode-count {
  font-size: 12px;
  color: #909399;
  background: rgba(255, 255, 255, 0.05);
  padding: 2px 8px;
  border-radius: 10px;
}

.trajectory-details {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
}

.detail-label {
  color: #909399;
}

.detail-value {
  font-weight: 600;
  color: #e5e8eb;
}

.map-container {
  flex: 1;
  position: relative;
}

.timeline-panel {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: #1a1f2e;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  padding: 12px 20px;
  box-shadow: 0 -2px 12px rgba(0, 0, 0, 0.3);
}

.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.timeline-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #fff;
}

.timeline-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.timeline-info {
  text-align: center;
  font-size: 12px;
  color: #909399;
}
</style>
