import { useAuthStore } from '@/stores/auth';

export const trajectoryApi = {
  // 原生 fetch 改造的预测接口 
  async predict(file: File, modelId: string) {
    const authStore = useAuthStore(); // 👈 第一步：去兜里把钥匙拿出来

    const formData = new FormData()
    formData.append('file', file)
    formData.append('modelId', modelId)

    // 使用 /api 触发 vite.config.ts 的跨域代理
    const response = await fetch('/api/trajectory/predict', {
      method: 'POST',
      headers: {
        // 👈 第二步：把钥匙（Token）塞进请求头里！注意 Bearer 后面有个空格
        'Authorization': `Bearer ${authStore.token}` 
      },
      body: formData
    })

    if (!response.ok) {
      // 捕获 401 错误，可以提示用户重新登录
      if (response.status === 401) {
          throw new Error('登录已过期，请重新登录');
      }
      throw new Error(`请求失败，状态码: ${response.status}`)
    }
    return await response.json()
  },

  // LLM 流式报告接口 (SSE 新增功能)
  async streamReport(
    params: { model_id: string; mode: string; confidence: string },
    onMessage: (text: string) => void,
    onDone: () => void,
    onError: (err: any) => void
  ) {
    const authStore = useAuthStore(); // 👈 第一步：同样，把钥匙拿出来

    try {
      // 必须使用 /api 前缀以匹配 vite.config.ts 的代理配置，避免跨域报错
      const response = await fetch('/api/trajectory/generate_report_stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          // 👈 第二步：流式接口也要带上钥匙
          'Authorization': `Bearer ${authStore.token}`
        },
        body: JSON.stringify(params)
      });

      if (!response.ok) {
        if (response.status === 401) throw new Error('登录已过期，请重新登录');
        throw new Error('网络请求失败');
      }

      if (!response.body) throw new Error('ReadableStream not supported in this browser.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;
      let fullText = '';

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          // 解码二进制 chunk
          const chunkStr = decoder.decode(value, { stream: true });
          const lines = chunkStr.split('\n\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.substring(6);
              if (dataStr.trim() === '') continue;
              
              try {
                const parsed = JSON.parse(dataStr);
                if (parsed.status === 'generating') {
                  fullText += parsed.content;
                  onMessage(fullText); // 把拼接好的最新文本传给 Vue 组件
                } else if (parsed.status === 'done') {
                  onDone();
                }
              } catch (e) {
                console.warn('解析流式 JSON 失败:', dataStr);
              }
            }
          }
        }
      }
    } catch (error) {
      onError(error);
    }
  }
}