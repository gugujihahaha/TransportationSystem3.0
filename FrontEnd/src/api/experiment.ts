import axios from 'axios'
import type { ExperimentInfo, EvaluationReport, PredictionSummary } from '@/types'

const API_BASE_URL = 'http://localhost:8000/api'

export const experimentApi = {
  async getExperiments(): Promise<ExperimentInfo[]> {
    const response = await axios.get<ExperimentInfo[]>(
      `${API_BASE_URL}/experiments`
    )

    return response.data
  },

  async getExperimentReport(expId: string): Promise<EvaluationReport> {
    const response = await axios.get<EvaluationReport>(
      `${API_BASE_URL}/experiments/${expId}/report`
    )

    return response.data
  },

  async getExperimentConfusionMatrix(expId: string): Promise<Blob> {
    const response = await axios.get(
      `${API_BASE_URL}/experiments/${expId}/confusion-matrix`,
      {
        responseType: 'blob',
      }
    )

    return response.data
  },

  async getExperimentPredictions(expId: string): Promise<PredictionSummary> {
    const response = await axios.get<PredictionSummary>(
      `${API_BASE_URL}/experiments/${expId}/predictions`
    )

    return response.data
  },

  async getExperimentErrorAnalysis(expId: string): Promise<any[]> {
    const response = await axios.get(
      `${API_BASE_URL}/experiments/${expId}/error-analysis`
    )

    return response.data
  },
}
