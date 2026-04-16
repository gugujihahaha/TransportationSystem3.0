import axios from 'axios'
import type { DatasetStats, DataCleaningStats } from '@/types'
import { useAuthStore } from '@/stores/auth'

const API_BASE_URL = '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL
})

apiClient.interceptors.request.use((config) => {
  const authStore = useAuthStore()
  if (authStore.token) {
    config.headers.Authorization = `Bearer ${authStore.token}`
  }
  return config
})

export const datasetApi = {
  async getStats(): Promise<DatasetStats> {
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