import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { TrajectoryPrediction, TransportMode } from '@/types'
import { trajectoryApi } from '@/api/trajectory'

export const useTrajectoryStore = defineStore('trajectory', () => {
  const predictions = ref<TrajectoryPrediction[]>([])
  
  const history = ref<any[]>([]) 
  
  const selectedTrajectory = ref<TrajectoryPrediction | null>(null)
  const transportModes = ref<TransportMode[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  const filteredPredictions = computed(() => {
    return predictions.value
  })

  async function loadTransportModes() {
    try {
      transportModes.value = await trajectoryApi.getModes()
    } catch (e) {
      error.value = '加载交通方式配置失败'
      console.error(e)
    }
  }

  async function predictTrajectory(file: File, model: string = 'exp1') {
    loading.value = true
    error.value = null

    try {
      const result = await trajectoryApi.predict(file, model)
      predictions.value.push(result)
      selectedTrajectory.value = result
      return result
    } catch (e) {
      error.value = '预测失败，请检查文件格式'
      console.error(e)
      throw e
    } finally {
      loading.value = false
    }
  }

  async function fetchHistory() {
    try {
      history.value = await trajectoryApi.getHistory()
    } catch (e) {
      console.error('拉取历史失败:', e)
    }
  }

  function selectTrajectory(trajectory: TrajectoryPrediction) {
    selectedTrajectory.value = trajectory
  }

  function clearPredictions() {
    predictions.value = []
    selectedTrajectory.value = null
  }

  return {
    predictions,
    history,             
    selectedTrajectory,
    transportModes,
    loading,
    error,
    filteredPredictions,
    loadTransportModes,
    predictTrajectory,
    fetchHistory,        
    selectTrajectory,
    clearPredictions,
  }
})