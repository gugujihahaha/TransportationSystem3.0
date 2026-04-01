import axios from 'axios'
import type { DatasetStats, DataCleaningStats } from '@/types'
import { useAuthStore } from '@/stores/auth'

const API_BASE_URL = '/api'

// 🚀 创建一个专属的 axios 实例
const apiClient = axios.create({
  baseURL: API_BASE_URL
})

// 🚀 添加请求拦截器：每次发请求前，自动把 token 塞进 headers
apiClient.interceptors.request.use((config) => {
  const authStore = useAuthStore()
  if (authStore.token) {
    config.headers.Authorization = `Bearer ${authStore.token}`
  }
  return config
})

export const datasetApi = {
  async getStats(): Promise<DatasetStats> {
    // 下面的请求直接用 apiClient，不再需要手动拼 API_BASE_URL
    const response = await apiClient.get<DatasetStats>('/dataset/stats')
    return response.data
  },

  async getModeDistribution(): Promise<{
    total: number
    distribution: Array<{
      mode: string
      count: number
      percentage: number
    }>
  }> {
    const response = await apiClient.get('/dataset/mode-distribution')
    return response.data
  },

  async getCleaningStats(): Promise<DataCleaningStats> {
    const response = await apiClient.get<DataCleaningStats>('/dataset/cleaning-stats')
    return response.data
  },
}