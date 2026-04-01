<template>
  <div class="trajectory-dashboard">
    
    <div class="upload-panel glass-card">
      <div class="panel-header">
        <div class="glow-title">
          <span class="title-text">多模态轨迹输入源</span>
          <div class="tech-line"></div>
        </div>
      </div>

      <div class="model-selector">
        <label class="neon-label">预测引擎模型：</label>
        <el-select 
          v-model="selectedModel" 
          placeholder="请选择核心神经模型" 
          popper-class="dark-sci-fi-popper"
        >
          <el-option 
            v-for="model in models" 
            :key="model.id" 
            :label="model.name" 
            :value="model.id"
          />
        </el-select>
      </div>

      <el-upload
        class="cyber-upload-area"
        drag
        :auto-upload="false"
        :show-file-list="false"
        accept=".csv,.json,.plt"
        @change="handleFileChange"
      >
        <div class="upload-content">
          <el-icon class="upload-icon glow-icon"><UploadFilled /></el-icon>
          <div class="upload-text">
            <p class="main-hint">将轨迹数据流拖拽至此，或点击建立连接</p>
            <p class="sub-hint">支持格式: CSV / JSON / PLT (上限 10MB)</p>
            <el-button type="primary" link class="neon-link-btn" @click.stop="showFormatDialog = true">
              [ 查阅数据矩阵标准 ]
            </el-button>
          </div>
        </div>
      </el-upload>
    </div>

    <div class="ai-panel">
      <AIAnalysisReport ref="aiReportRef" />
    </div>

    <el-dialog 
      v-model="showFormatDialog" 
      title="数据矩阵协议说明" 
      width="600px"
      custom-class="cyber-dialog"
    >
      <div class="format-info">
        <h4 class="neon-text">PLT 格式（Geolife）：</h4>
        <pre class="cyber-code-block">
Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,My Track,0,0,2,8421376
0
39.8626216,116.3859383,0,1000.7,39761.082037037,2008-11-09,01:58:08
        </pre>

        <h4 class="neon-text">CSV 格式：</h4>
        <pre class="cyber-code-block">
latitude,longitude,timestamp
39.9042,116.4074,2024-01-01 08:00:00
        </pre>

        <h4 class="neon-text">JSON 格式：</h4>
        <pre class="cyber-code-block">
[
  {
    "latitude": 39.9042,
    "longitude": 116.4074,
    "timestamp": "2024-01-01 08:00:00"
  }
]
        </pre>

        <el-alert type="warning" :closable="false" class="cyber-alert" style="margin-top: 20px">
          <template #title>
            <strong>系统警告：</strong>
          </template>
          <ul class="alert-list">
            <li>时间戳需严格遵循：YYYY-MM-DD HH:MM:SS</li>
            <li>为保证模型精度，至少包含 10 个空间坐标点</li>
            <li>单次传输负载不得超过 10MB</li>
          </ul>
        </el-alert>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
// 引入刚刚编写的 AI 分析组件
import AIAnalysisReport from './AIAnalysisReport.vue'

const emit = defineEmits<{
  upload: [file: File, model: string]
}>()

const showFormatDialog = ref(false)
const selectedModel = ref('exp1')
const aiReportRef = ref(null) // 用于后续可能的主动触发报告生成

const models = [
  { id: 'exp1', name: 'Exp1 (运动学特征基线)' },
  { id: 'exp2', name: 'Exp2 (+OSM 空间路网)' },
  { id: 'exp3', name: 'Exp3 (+实时气象感知)' },
  { id: 'exp4', name: 'Exp4 (Loss 优化终极版)' },
]

function handleFileChange(file: any) {
  if (!file.raw) return

  const validTypes = ['text/csv', 'application/json', '.csv', '.json', '.plt']
  const isValid = validTypes.some(type => 
    file.raw.type.includes(type) || file.name.endsWith(type)
  )

  if (!isValid) {
    ElMessage.error('识别失败：格式不符合系统协议 (仅限 CSV/JSON/PLT)')
    return
  }

  if (file.size > 10 * 1024 * 1024) {
    ElMessage.error('传输超载：数据矩阵大小不能超过 10MB')
    return
  }

  ElMessage.success('数据链接成功，正在传输至核心引擎...')
  emit('upload', file.raw, selectedModel.value)
  
  // 提示：你可以在这里或者父组件中，通过调用 aiReportRef.value.generateReport() 来联动生成报告
}
</script>

<style scoped>
/* 定义核心视觉变量 */
.trajectory-dashboard {
  --neon-cyan: #00f0ff;
  --tech-blue: #4A90E2;
  --deep-bg: #030816;
  --glass-bg: rgba(3, 8, 22, 0.6);
  --glass-border: rgba(0, 240, 255, 0.2);
  
  display: grid;
  grid-template-columns: 1fr 1fr; /* 左右双栏等宽 */
  gap: 24px;
  width: 100%;
  min-height: 500px;
}

@media (max-width: 992px) {
  .trajectory-dashboard {
    grid-template-columns: 1fr; /* 小屏自动折叠为上下结构 */
  }
}

/* 玻璃拟态卡片基础 */
.glass-card {
  background: var(--glass-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(0, 240, 255, 0.05);
  padding: 24px;
  display: flex;
  flex-direction: column;
}

/* 面板头部 */
.panel-header {
  margin-bottom: 24px;
}

.glow-title {
  position: relative;
  display: inline-block;
}

.title-text {
  font-size: 1.25rem;
  font-weight: 600;
  color: #fff;
  letter-spacing: 2px;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
}

.tech-line {
  height: 2px;
  width: 100%;
  background: linear-gradient(90deg, var(--neon-cyan), transparent);
  margin-top: 8px;
  box-shadow: 0 0 8px var(--neon-cyan);
}

/* 模型选择器重构 */
.model-selector {
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.neon-label {
  font-size: 14px;
  color: var(--neon-cyan);
  text-shadow: 0 0 5px rgba(0, 240, 255, 0.4);
  white-space: nowrap;
}

/* 深度覆写 Element Plus 的 Select 样式 */
:deep(.el-select) {
  flex: 1;
}
:deep(.el-input__wrapper) {
  background-color: rgba(3, 8, 22, 0.8) !important;
  border: 1px solid var(--tech-blue) !important;
  box-shadow: none !important;
  border-radius: 8px;
  transition: all 0.3s;
}
:deep(.el-input__wrapper.is-focus) {
  border-color: var(--neon-cyan) !important;
  box-shadow: 0 0 10px rgba(0, 240, 255, 0.3) !important;
}
:deep(.el-input__inner) {
  color: #fff !important;
}

/* 拖拽上传区域暗黑重构 */
.cyber-upload-area {
  flex: 1;
  display: flex;
  flex-direction: column;
}
:deep(.el-upload) {
  flex: 1;
}
:deep(.el-upload-dragger) {
  width: 100%;
  height: 100%;
  min-height: 250px;
  background: rgba(0, 240, 255, 0.03) !important;
  border: 1px dashed var(--tech-blue) !important;
  border-radius: 12px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  align-items: center;
  justify-content: center;
}
:deep(.el-upload-dragger:hover) {
  border-color: var(--neon-cyan) !important;
  background: rgba(0, 240, 255, 0.08) !important;
  box-shadow: 0 0 20px rgba(0, 240, 255, 0.2) inset;
}

.upload-content {
  text-align: center;
}

.upload-icon {
  font-size: 56px;
  color: var(--tech-blue);
  margin-bottom: 16px;
  transition: all 0.3s;
}
:deep(.el-upload-dragger:hover) .upload-icon {
  color: var(--neon-cyan);
  transform: translateY(-5px);
  filter: drop-shadow(0 0 10px var(--neon-cyan));
}

.main-hint {
  font-size: 15px;
  color: #fff;
  letter-spacing: 1px;
}

.sub-hint {
  font-size: 12px;
  color: #8892b0;
  margin-top: 8px;
  margin-bottom: 16px;
}

.neon-link-btn {
  color: var(--neon-cyan) !important;
  font-family: 'Fira Code', monospace;
  text-shadow: 0 0 5px rgba(0, 240, 255, 0.3);
}
.neon-link-btn:hover {
  text-shadow: 0 0 15px var(--neon-cyan);
}

/* 弹窗内部样式重构 */
.format-info {
  color: #cdd9e5;
}
.neon-text {
  color: var(--neon-cyan);
  text-shadow: 0 0 5px rgba(0, 240, 255, 0.3);
  font-weight: 500;
  margin: 16px 0 8px;
}
.cyber-code-block {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(74, 144, 226, 0.3);
  border-left: 3px solid var(--neon-cyan);
  padding: 12px;
  border-radius: 4px;
  font-family: 'Fira Code', Consolas, monospace;
  font-size: 13px;
  color: #a5d6ff;
  overflow-x: auto;
}
.cyber-alert {
  background: rgba(230, 162, 60, 0.1) !important;
  border: 1px solid rgba(230, 162, 60, 0.3) !important;
  color: #e6a23c !important;
}
.alert-list {
  padding-left: 20px;
  margin-top: 8px;
  line-height: 1.8;
}

/* --- 全局样式注入 (为了覆盖 body 下的 El-Dialog) --- */
/* 注意：实际项目中建议将这段放在全局 css 中，或者保留在这里 */
:global(.cyber-dialog) {
  background: rgba(10, 15, 30, 0.95) !important;
  border: 1px solid rgba(0, 240, 255, 0.3) !important;
  box-shadow: 0 0 30px rgba(0, 0, 0, 0.8), 0 0 20px rgba(0, 240, 255, 0.1) !important;
  backdrop-filter: blur(10px);
}
:global(.cyber-dialog .el-dialog__title) {
  color: #00f0ff !important;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.5);
}
</style>