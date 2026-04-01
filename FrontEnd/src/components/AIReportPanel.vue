<template>
  <div class="ai-report-container">
    <div class="panel-header">
      <h3 class="panel-title">🧠 交通大模型智能分析报告</h3>
      <div class="header-actions">
        <button @click="generateAIReport" class="btn-primary" :disabled="isGenerating">
          {{ isGenerating ? '大模型推理中...' : '生成最新分析' }}
        </button>
        <button @click="exportToPDF" class="btn-export" :disabled="!reportContent || isGenerating">
          📄 一键导出PDF
        </button>
      </div>
    </div>

    <div class="report-content" id="pdf-content">
      <div v-if="isGenerating" class="skeleton-loader">
        <div class="skeleton-line title"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line short"></div>
      </div>
      
      <div v-else-if="reportContent" class="markdown-body" v-html="formattedReport"></div>
      
      <div v-else class="empty-state">
        <p>点击上方按钮，基于当前多源融合数据生成智能研判报告。</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import html2pdf from 'html2pdf.js'

const isGenerating = ref(false)
const reportContent = ref('')

// 处理换行符，简单的Markdown渲染转换
const formattedReport = computed(() => {
  if (!reportContent.value) return ''
  return reportContent.value
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
})

const generateAIReport = async () => {
  isGenerating.value = true
  reportContent.value = ''
  
  try {
    // 构造 Prompt
    const prompt = `你是一个专业的交通智能分析专家。请基于以下数据生成分析报告：
    1. 早高峰私家车占比45%，公交28%。
    2. 共享单车出行减少碳排放约 1,245 kg。
    请按“数据现状、成因剖析、调控建议”三个部分输出，使用Markdown格式。`

    const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // YOUR_API_KEY 
        'Authorization': `Bearer sk-8447085e5bad43ffa6267f85f1f5613f` 
      },
      body: JSON.stringify({
        model: 'deepseek-chat', // 对应平台的模型名称
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    // 拿到大模型的真实回复
    reportContent.value = data.choices[0].message.content
    isGenerating.value = false

  } catch (error) {
    console.error("AI生成失败", error)
    reportContent.value = "**生成报告失败**\n请检查网络连接或 API Key 是否填写正确。"
    isGenerating.value = false
  }
}

// 核心：一键导出PDF功能
const exportToPDF = () => {
  const element = document.getElementById('pdf-content')
  const opt = {
    margin:       10,
    filename:     `交通智能分析报告_${new Date().getTime()}.pdf`,
    // 加上 as const 解决 TS 报错
    image:        { type: 'jpeg' as const, quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#ffffff' },
    jsPDF:        { unit: 'mm' as const, format: 'a4' as const, orientation: 'portrait' as const }
  }

  // 为了保证导出的PDF字迹清晰且白底黑字，这里临时在克隆节点上修改样式
  const clone = element?.cloneNode(true) as HTMLElement
  clone.style.backgroundColor = 'white'
  clone.style.color = 'black'
  clone.style.padding = '20px'
  
  // 生成并下载
  html2pdf().set(opt).from(clone).save()
}
</script>

<style scoped>
.ai-report-container {
  background: rgba(30, 41, 59, 0.7);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 8px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding-bottom: 15px;
  margin-bottom: 15px;
}

.panel-title {
  color: #38bdf8;
  font-size: 18px;
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 10px;
}

button {
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s;
  border: none;
}

.btn-primary {
  background: #0ea5e9;
  color: white;
}
.btn-primary:hover:not(:disabled) { background: #0284c7; }
.btn-primary:disabled { background: #475569; cursor: not-allowed; }

.btn-export {
  background: #10b981;
  color: white;
}
.btn-export:hover:not(:disabled) { background: #059669; }
.btn-export:disabled { background: #475569; cursor: not-allowed; }

.report-content {
  flex: 1;
  overflow-y: auto;
  color: #f8fafc;
  line-height: 1.6;
  font-size: 15px;
  padding: 10px;
}

.empty-state {
  text-align: center;
  color: #64748b;
  margin-top: 40px;
}

/* 骨架屏动画 */
.skeleton-loader { width: 100%; }
.skeleton-line {
  height: 12px;
  background: linear-gradient(90deg, #334155 25%, #475569 50%, #334155 75%);
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
  margin-bottom: 15px;
  border-radius: 4px;
}
.skeleton-line.title { height: 20px; width: 40%; margin-bottom: 24px; }
.skeleton-line.short { width: 60%; }
@keyframes loading {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>