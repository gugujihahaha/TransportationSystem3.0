import { useAuthStore } from '@/stores/auth';

export const trajectoryApi = {
  // 获取所有交通方式的配置
  async getModes() {
    const authStore = useAuthStore();
    const response = await fetch('/api/trajectory/modes', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${authStore.token}`
      }
    });

    if (!response.ok) {
      if (response.status === 401) {
        authStore.logout();
        throw new Error('登录已过期，请重新登录');
      }
      throw new Error(`获取交通配置失败，状态码: ${response.status}`);
    }
    return await response.json();
  },

  //预测接口 
  async predict(file: File, modelId: string, scene: string = 'unknown') {
    const authStore = useAuthStore(); 
    const formData = new FormData()
    formData.append('file', file)
    formData.append('modelId', modelId)
    formData.append('scene', scene)
    const response = await fetch('/api/trajectory/predict', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${authStore.token}` 
      },
      body: formData
    })

    if (!response.ok) {
      if (response.status === 401) {
          throw new Error('登录已过期，请重新登录');
      }
      throw new Error(`请求失败，状态码: ${response.status}`)
    }
    return await response.json()
  },
  async getHistoryById(id: string) {
    const authStore = useAuthStore();
    const response = await fetch(`/api/trajectory/history/${id}`, {
      method: 'GET',
      headers: { 'Authorization': `Bearer ${authStore.token}` }
    });
    if (!response.ok) throw new Error('获取失败');
    return await response.json();
  },
  async getHistory() {
    const authStore = useAuthStore();
    const response = await fetch('/api/trajectory/history', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${authStore.token}` 
      }
    });

    if (!response.ok) {
      if (response.status === 401) {
        authStore.logout();
        throw new Error('登录已过期');
      }
      throw new Error(`获取历史失败，状态码: ${response.status}`);
    }
    return await response.json();
  },

  // LLM 流式报告接口
  async streamReport(
    params: { 
      model_id: string; 
      mode: string; 
      confidence: string; 
      scene: string; 
      distance?: string; 
      co2?: string; 
    },
    onMessage: (text: string) => void,
    onDone: () => void,
    onError: (err: any) => void
  ) {
    const authStore = useAuthStore();
    try {
      const response = await fetch('/api/trajectory/generate_report_stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Authorization': `Bearer ${authStore.token}`
        },
        body: JSON.stringify(params)
      });

      if (!response.ok) throw new Error('网络请求失败');
      if (!response.body) throw new Error('浏览器不支持流式传输');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let fullText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunkStr = decoder.decode(value, { stream: true });
        const lines = chunkStr.split('\n\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.substring(6));
              if (parsed.status === 'generating') {
                fullText += parsed.content;
                onMessage(fullText);
              } else if (parsed.status === 'done') {
                onDone();
              }
            } catch (e) { continue; }
          }
        }
      }
    } catch (error) { onError(error); }
  }
}