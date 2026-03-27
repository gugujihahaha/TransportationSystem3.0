export interface TrajectoryPoint {
  lat: number
  lng: number
  timestamp: string
  speed?: number
}

export interface TrajectoryStats {
  distance: number
  duration: number
  avg_speed: number
  max_speed: number
}

export interface TrajectoryPrediction {
  trajectory_id: string
  predicted_mode: string
  confidence: number
  points: TrajectoryPoint[]
  stats: TrajectoryStats
}

export interface TransportMode {
  id: string
  name: string
  color: string
  icon: string
}

export interface ExperimentInfo {
  id: string
  name: string
  description: string
  features: string[]
  status: 'completed' | 'not_trained' | 'training'
}

export interface EvaluationReport {
  accuracy: number
  precision: Record<string, number>
  recall: Record<string, number>
  f1_score: Record<string, number>
  classification_report: Record<string, any>
}

export interface PredictionSummary {
  total_predictions: number
  mode_distribution: Record<string, number>
  accuracy: number
}

export interface DatasetStats {
  total_trajectories: number
  total_users: number
  mode_distribution: Record<string, number>
  avg_trajectory_length: number
  date_range: {
    start: string
    end: string
  }
  total_distance: string
}

export interface DataCleaningStep {
  name: string
  count: number
}

export interface DataCleaningStats {
  steps: DataCleaningStep[]
}

export interface TrainingProgress {
  task_id: string
  exp_name: string
  epoch: number
  total_epochs: number
  loss: number
  accuracy: number
  status: 'training' | 'completed' | 'failed' | 'cancelled'
}

export interface TrainingRequest {
  exp_name: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
}

export interface TrainingResponse {
  task_id: string
  status: string
  message: string
}
