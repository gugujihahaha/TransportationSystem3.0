<template>
  <div class="ai-report-container">
    <div class="panel-header">
      <h3 class="panel-title">🧠 星火大模型智能分析报告</h3>
      <div class="header-actions">
        <button @click="generateAIReport" class="btn-primary" :disabled="isGenerating">
          {{ isGenerating ? '引擎推理中...' : '生成最新分析' }}
        </button>
        <button @click="exportToPDF" class="btn-export" :disabled="!reportContent || isGenerating">
          📄 一键导出PDF
        </button>
      </div>
    </div>

    <div class="report-content" id="pdf-content" ref="scrollBox">
      <div v-if="isGenerating && !reportContent" class="skeleton-loader">
        <div class="skeleton-line title"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line short"></div>
      </div>
      
      <div v-else-if="reportContent" class="markdown-body">
        <span v-html="formattedReport"></span>
        <span v-if="isGenerating" class="typing-cursor">|</span>
      </div>
      
      <div v-else class="empty-state">
        <p>点击上方按钮，基于当前多源融合数据生成智能研判报告。</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import html2pdf from 'html2pdf.js'

const isGenerating = ref(false)
const reportContent = ref('')
const scrollBox = ref<HTMLElement | null>(null)

// 处理换行符，简单的Markdown渲染转换
const formattedReport = computed(() => {
  if (!reportContent.value) return ''
  return reportContent.value
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
})

// 监听内容变化，实现自动平滑滚动到底部
watch(reportContent, () => {
  nextTick(() => {
    if (scrollBox.value) {
      scrollBox.value.scrollTop = scrollBox.value.scrollHeight
    }
  })
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

    // 🚀 使用配置好的 Vite 代理请求星火大模型
    const response = await fetch('/spark-api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // 【关键】在这里换成你的星火 API 密码 (API Password)
        'Authorization': `Bearer ZOawgFgAMWrgzoramwRS:BkjUHBXpuOrXCpVQfFtJ` 
      },
      body: JSON.stringify({
        model: 'lite', // 星火免费版模型
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7,
        stream: true // 开启流式输出核心参数
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    if (!response.body) throw new Error('浏览器不支持 ReadableStream')

    // 处理数据流
    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      const chunk = decoder.decode(value, { stream: true })
      const lines = chunk.split('\n')
      
      for (const line of lines) {
        if (line.startsWith('data: ') && !line.includes('[DONE]')) {
          try {
            const data = JSON.parse(line.substring(6))
            const text = data.choices[0].delta.content || ''
            reportContent.value += text
          } catch (e) {
            // 忽略 JSON 解析截断导致的错误，继续读取下一块
          }
        }
      }
    }

  } catch (error) {
    console.error("AI生成失败", error)
    reportContent.value = "**生成报告失败**\n请检查网络连接或 API Key 是否有效。"
  } finally {
    isGenerating.value = false
  }
}

// 核心：一键导出白底黑字高清 PDF
const exportToPDF = () => {
  const element = document.getElementById('pdf-content')
  if (!element) return

  const opt = {
    margin:       10,
    filename:     `交通智能分析报告_${new Date().getTime()}.pdf`,
    image:        { type: 'jpeg' as const, quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#ffffff' }, // 强制底色为纯白
    jsPDF:        { unit: 'mm' as const, format: 'a4' as const, orientation: 'portrait' as const }
  }

  // 克隆节点进行样式魔改，防止影响页面上的深色UI
  const clone = element.cloneNode(true) as HTMLElement
  
  // 1. 强制容器背景白、文字黑
  clone.style.backgroundColor = 'white'
  clone.style.color = 'black'
  clone.style.padding = '20px'
  clone.style.height = 'auto' // 释放高度限制，打印全部内容
  clone.style.overflow = 'visible'
  
  // 2. 递归将所有内部标签文字变黑
  const allChildren = clone.querySelectorAll('*')
  allChildren.forEach((child: any) => {
    child.style.color = 'black'
  })

  // 3. 移除导出时多余的光标，保持文档正式感
  const cursor = clone.querySelector('.typing-cursor')
  if (cursor) cursor.remove()
  
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
  overflow-y: auto; /* 保留内部滚动 */
  color: #f8fafc;
  line-height: 1.6;
  font-size: 15px;
  padding: 10px;
  /* 优化滚动条样式 */
  scrollbar-width: thin;
  scrollbar-color: rgba(56, 189, 248, 0.5) transparent;
}

/* Chrome/Safari 滚动条 */
.report-content::-webkit-scrollbar {
  width: 6px;
}
.report-content::-webkit-scrollbar-thumb {
  background-color: rgba(56, 189, 248, 0.5);
  border-radius: 10px;
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

/* 打字机光标动画 */
.typing-cursor {
  display: inline-block;
  width: 8px;
  height: 15px;
  background-color: #38bdf8;
  vertical-align: middle;
  margin-left: 2px;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style>