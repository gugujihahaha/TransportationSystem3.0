<template>
  <div class="ai-report-container glass-card">
    <div class="report-header">
      <div class="title-wrapper">
        <i class="el-icon-cpu neon-icon"></i>
        <span class="title-text">TrafficRec 智能感知分析报告</span>
      </div>
      <div class="status-tag" :class="{ 'is-loading': loading }">
        {{ loading ? 'AI 正在深度解构中...' : '深度学习引擎已就绪' }}
      </div>
    </div>

    <div class="report-content-wrapper" ref="scrollContainer">
      <div v-if="!reportText && !loading" class="empty-state">
        <p>暂无分析数据，请上传轨迹以激活多模态感知引擎</p>
      </div>
      
      <div v-else class="markdown-body ai-text-stream">
        {{ reportText }}
        <span v-if="loading" class="cursor-blink">|</span>
      </div>
    </div>

    <div class="report-footer">
      <el-button 
        type="primary" 
        class="neon-button" 
        :loading="loading"
        @click="generateReport"
      >
        {{ reportText ? '重新生成分析' : '生成 AI 绿色报告' }}
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onUnmounted } from 'vue';
import { ElMessage } from 'element-plus';

const reportText = ref('');
const loading = ref(false);
const scrollContainer = ref<HTMLElement | null>(null);

// 核心：流式接口调用
const generateReport = async () => {
  if (loading.value) return;
  
  reportText.value = '';
  loading.value = true;

  try {
    // 使用 fetch 模拟 SSE 处理，以便更好地处理 Authorization Header
    const response = await fetch('/api/v1/experiment/streamReport', {
      method: 'POST', // 或 GET，根据你后端接口定义
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}` // 确保携带 Token
      },
      body: JSON.stringify({
        // 传递当前实验 ID 或轨迹特征数据
        trajectoryId: 'current-task-001' 
      })
    });

    if (!response.body) throw new Error('ReadableStream not supported');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      // 模拟打字机平滑感
      reportText.value += chunk;

      // 自动滚动到底部
      await nextTick();
      if (scrollContainer.value) {
        scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight;
      }
    }
  } catch (error) {
    ElMessage.error('AI 报告生成失败，请检查网络或认证状态');
    console.error(error);
  } finally {
    loading.value = false;
  }
};

onUnmounted(() => {
  loading.value = false;
});
</script>

<style scoped>
/* 2.0 风格核心变量 */
:tep {
  --neon-cyan: #00f0ff;
  --tech-blue: #4A90E2;
  --deep-bg: #030816;
}

.glass-card {
  background: rgba(3, 8, 22, 0.7);
  backdrop-filter: blur(15px);
  border: 1px solid rgba(0, 240, 255, 0.2);
  border-radius: 12px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 240, 255, 0.05);
  padding: 20px;
  color: #fff;
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(0, 240, 255, 0.3);
  padding-bottom: 15px;
  margin-bottom: 20px;
}

.title-text {
  font-size: 1.2rem;
  font-weight: bold;
  letter-spacing: 2px;
  color: var(--neon-cyan);
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
}

.status-tag {
  font-size: 0.8rem;
  color: #888;
}

.status-tag.is-loading {
  color: var(--neon-cyan);
  animation: breathe 2s infinite;
}

.report-content-wrapper {
  height: 400px;
  overflow-y: auto;
  padding: 10px;
  font-family: 'Fira Code', monospace;
  line-height: 1.6;
  scrollbar-width: thin;
}

/* 霓虹滚动条 */
.report-content-wrapper::-webkit-scrollbar {
  width: 4px;
}
.report-content-wrapper::-webkit-scrollbar-thumb {
  background: var(--neon-cyan);
  box-shadow: 0 0 5px var(--neon-cyan);
}

.ai-text-stream {
  white-space: pre-wrap;
  color: rgba(255, 255, 255, 0.9);
  text-shadow: 0 0 2px rgba(255, 255, 255, 0.2);
}

.cursor-blink {
  display: inline-block;
  width: 2px;
  background: var(--neon-cyan);
  margin-left: 4px;
  animation: blink 1s infinite;
}

.neon-button {
  background: transparent !important;
  border: 1px solid var(--neon-cyan) !important;
  color: var(--neon-cyan) !important;
  box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
  transition: all 0.3s;
}

.neon-button:hover {
  background: var(--neon-cyan) !important;
  color: var(--deep-bg) !important;
  box-shadow: 0 0 20px var(--neon-cyan);
}

@keyframes blink {
  50% { opacity: 0; }
}

@keyframes breathe {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>