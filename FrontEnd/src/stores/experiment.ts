import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ExperimentInfo, EvaluationReport, PredictionSummary } from '@/types'
import { experimentApi } from '@/api/experiment'

export const useExperimentStore = defineStore('experiment', () => {
  const experiments = ref<ExperimentInfo[]>([])
  const selectedExperiment = ref<ExperimentInfo | null>(null)
  const currentReport = ref<EvaluationReport | null>(null)
  const currentPredictions = ref<PredictionSummary | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const completedExperiments = computed(() => {
    return experiments.value.filter(exp => exp.status === 'completed')
  })

  const notTrainedExperiments = computed(() => {
    return experiments.value.filter(exp => exp.status === 'not_trained')
  })

  async function loadExperiments() {
    loading.value = true
    error.value = null

    try {
      experiments.value = await experimentApi.getExperiments()
    } catch (e) {
      error.value = '加载实验信息失败'
      console.error(e)
    } finally {
      loading.value = false
    }
  }

  async function loadExperimentReport(expId: string) {
    loading.value = true
    error.value = null

    try {
      currentReport.value = await experimentApi.getExperimentReport(expId)
    } catch (e) {
      error.value = '加载评估报告失败'
      console.error(e)
    } finally {
      loading.value = false
    }
  }

  async function loadExperimentPredictions(expId: string) {
    loading.value = true
    error.value = null

    try {
      currentPredictions.value = await experimentApi.getExperimentPredictions(expId)
    } catch (e) {
      error.value = '加载预测结果失败'
      console.error(e)
    } finally {
      loading.value = false
    }
  }

  function selectExperiment(exp: ExperimentInfo) {
    selectedExperiment.value = exp
  }

  return {
    experiments,
    selectedExperiment,
    currentReport,
    currentPredictions,
    loading,
    error,
    completedExperiments,
    notTrainedExperiments,
    loadExperiments,
    loadExperimentReport,
    loadExperimentPredictions,
    selectExperiment,
  }
})
