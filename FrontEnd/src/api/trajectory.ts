
export const trajectoryApi = {
  // 原生 fetch 改造的预测接口 
  async predict(file: File, modelId: string) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model_id', modelId)

    // 使用 /api 触发 vite.config.ts 的跨域代理
    const response = await fetch('/api/trajectory/predict', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
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
    try {
      // 必须使用 /api 前缀以匹配 vite.config.ts 的代理配置，避免跨域报错
      const response = await fetch('/api/trajectory/generate_report_stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(params)
      });

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