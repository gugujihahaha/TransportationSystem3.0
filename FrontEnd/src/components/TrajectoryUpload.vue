<template>
  <div class="trajectory-upload">
    <div class="model-selector">
      <label>选择预测模型：</label>
      <el-select v-model="selectedModel" placeholder="请选择模型">
        <el-option 
          v-for="model in models" 
          :key="model.id" 
          :label="model.name" 
          :value="model.id"
        />
      </el-select>
    </div>

    <el-upload
      class="upload-area"
      drag
      :auto-upload="false"
      :show-file-list="false"
      accept=".csv,.json,.plt"
      @change="handleFileChange"
    >
      <div class="upload-content">
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-text">
          <p>拖拽文件到此处或点击上传</p>
          <p class="upload-hint">支持 CSV、JSON 或 PLT 格式</p>
        </div>
      </div>
    </el-upload>

    <el-dialog v-model="showFormatDialog" title="文件格式说明" width="600px">
      <div class="format-info">
        <h4>PLT 格式（Geolife）：</h4>
        <pre class="code-block">
Geolife trajectory
WGS84
Altitude is in Feet
Reserved
3
1
0
10
39.9042,116.4074,0,100,39245.3333,2024-01-01,08:00:00
39.9050,116.4080,0,100,39245.3334,2024-01-01,08:00:05
        </pre>

        <h4>CSV 格式：</h4>
        <pre class="code-block">
latitude,longitude,timestamp
39.9042,116.4074,2024-01-01 08:00:00
39.9050,116.4080,2024-01-01 08:00:05
        </pre>

        <h4>JSON 格式：</h4>
        <pre class="code-block">
[
  {
    "latitude": 39.9042,
    "longitude": 116.4074,
    "timestamp": "2024-01-01 08:00:00"
  }
]
        </pre>

        <el-alert type="info" :closable="false" style="margin-top: 20px">
          <template #title>
            <strong>注意：</strong>
          </template>
          <ul>
            <li>时间戳格式：YYYY-MM-DD HH:MM:SS</li>
            <li>至少包含 10 个GPS点</li>
            <li>文件大小建议不超过 10MB</li>
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

const emit = defineEmits<{
  upload: [file: File, model: string]
}>()

const showFormatDialog = ref(false)
const selectedModel = ref('exp1')

const models = [
  { id: 'exp1', name: 'exp1 (9维轨迹特征)' },
  { id: 'exp2', name: 'exp2 (9维轨迹特征 + OSM)' },
  { id: 'exp3', name: 'exp3 (9维轨迹特征 + 天气)' },
  { id: 'exp4', name: 'exp4 (9维轨迹特征 + OSM + 天气)' },
]

function handleFileChange(file: any) {
  if (!file.raw) return

  const validTypes = ['text/csv', 'application/json', '.csv', '.json', '.plt']
  const isValid = validTypes.some(type => 
    file.raw.type.includes(type) || file.name.endsWith(type)
  )

  if (!isValid) {
    ElMessage.error('仅支持 CSV、JSON 或 PLT 格式的文件')
    return
  }

  if (file.size > 10 * 1024 * 1024) {
    ElMessage.error('文件大小不能超过 10MB')
    return
  }

  emit('upload', file.raw, selectedModel.value)
}
</script>

<style scoped>
.trajectory-upload {
  width: 100%;
}

.model-selector {
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.model-selector label {
  font-size: 14px;
  color: #e5e8eb;
  white-space: nowrap;
}

.model-selector .el-select {
  flex: 1;
}

.upload-area {
  width: 100%;
}

.upload-content {
  padding: 40px 0;
  text-align: center;
}

.upload-icon {
  font-size: 48px;
  color: #409EFF;
  margin-bottom: 16px;
}

.upload-text {
  font-size: 14px;
  color: #606266;
}

.upload-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 8px;
}

.format-info {
  padding: 0 20px;
}

.format-info h4 {
  margin: 16px 0 8px;
  font-size: 16px;
  color: #303133;
}

.code-block {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  font-size: 12px;
  color: #606266;
  overflow-x: auto;
}

:deep(.el-upload-dragger) {
  width: 100%;
  height: 200px;
}
</style>
