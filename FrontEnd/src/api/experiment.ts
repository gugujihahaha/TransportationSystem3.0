import axios from 'axios'
import type { ExperimentInfo, EvaluationReport, PredictionSummary } from '@/types'
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

export const experimentApi = {
  async getExperiments(): Promise<ExperimentInfo[]> {
    const response = await apiClient.get<ExperimentInfo[]>('/experiments')
    return response.data
  },

  async getExperimentReport(expId: string): Promise<EvaluationReport> {
    const response = await apiClient.get<EvaluationReport>(`/experiments/${expId}/report`)
    return response.data
  },

  async getExperimentConfusionMatrix(expId: string): Promise<Blob> {
    const response = await apiClient.get(`/experiments/${expId}/confusion-matrix`, {
      responseType: 'blob',
    })
    return response.data
  },

  async getExperimentPredictions(expId: string): Promise<PredictionSummary> {
    const response = await apiClient.get<PredictionSummary>(`/experiments/${expId}/predictions`)
    return response.data
  },

  async getExperimentErrorAnalysis(expId: string): Promise<any[]> {
    const response = await apiClient.get(`/experiments/${expId}/error-analysis`)
    return response.data
  }
}