import axios from 'axios'
import type { DatasetStats, DataCleaningStats } from '@/types'

const API_BASE_URL = 'http://localhost:8000/api'

export const datasetApi = {
  async getStats(): Promise<DatasetStats> {
    const response = await axios.get<DatasetStats>(
      `${API_BASE_URL}/dataset/stats`
    )

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
    const response = await axios.get(
      `${API_BASE_URL}/dataset/mode-distribution`
    )

    return response.data
  },

  async getCleaningStats(): Promise<DataCleaningStats> {
    const response = await axios.get<DataCleaningStats>(
      `${API_BASE_URL}/dataset/cleaning-stats`
    )

    return response.data
  },
}
