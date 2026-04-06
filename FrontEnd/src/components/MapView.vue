<template>
  <div class="map-view" ref="mapContainer"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type { TrajectoryPoint, TrajectoryPrediction } from '@/types'

interface Props {
  trajectories: TrajectoryPrediction[]
  selectedTrajectory: TrajectoryPrediction | null
  showSpeedHeatmap?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showSpeedHeatmap: false,
})

const emit = defineEmits<{
  select: [trajectory: TrajectoryPrediction]
}>()

const mapContainer = ref<HTMLElement>()
let map: L.Map | null = null
let trajectoryLayers: L.Polyline[] = []
let markers: L.Layer[] = [] 

onMounted(() => {
  if (!mapContainer.value) return

  map = L.map(mapContainer.value).setView([39.9042, 116.4074], 12)

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap contributors, © CARTO',
    maxZoom: 19,
  }).addTo(map)
})

watch(() => props.trajectories, (newTrajectories) => {
  updateMap(newTrajectories)
}, { deep: true })

watch(() => props.selectedTrajectory, (selected) => {
  highlightTrajectory(selected)
})

function updateMap(trajectories: TrajectoryPrediction[]) {
  if (!map) return

  clearMap()

  const allLatLngs: [number, number][] = [] // 用于计算全局视野

  trajectories.forEach((traj) => {
    const points: [number, number][] = traj.points.map(p => {
      const pt: [number, number] = [p.lat, p.lng]
      allLatLngs.push(pt)
      return pt
    })

    const color = getModeColor(traj.predicted_mode)

    const polyline = L.polyline(points, {
      color,
      weight: 3,
      opacity: 0.8,
    }).addTo(map!)

    polyline.on('click', () => {
      emit('select', traj)
    })

    polyline.on('mouseover', (e: L.LeafletMouseEvent) => {
      const point = e.latlng
      const nearestPoint = findNearestPoint(traj, point)
      
      if (nearestPoint) {
        L.popup()
          .setLatLng(point)
          .setContent(`
            <div class="cyber-popup">
              <strong class="mode-title" style="color: ${color}">${getModeName(traj.predicted_mode)}</strong><br>
              <span class="info-label">速度:</span> ${nearestPoint.speed?.toFixed(2) || 'N/A'} m/s<br>
              <span class="info-label">置信度:</span> ${(traj.confidence * 100).toFixed(1)}%
            </div>
          `)
          .openOn(map!)
      }
    })

    polyline.on('mouseout', () => {
      map?.closePopup()
    })

    trajectoryLayers.push(polyline)

    if (traj.points.length > 0) {
      const startPoint = traj.points[0]
      const endPoint = traj.points[traj.points.length - 1]

      if (startPoint && endPoint) {
        // 提取纯净坐标
        const startCoord: [number, number] = [startPoint.lat, startPoint.lng]
        const endCoord: [number, number] = [endPoint.lat, endPoint.lng]

        const startMarker = L.circleMarker(startCoord, {
          radius: 8,
          fillColor: '#52C41A',
          color: '#fff',
          weight: 2,
        }).addTo(map!)

        const endBounds: [[number, number], [number, number]] = [endCoord, endCoord]
        const endMarker = L.rectangle(endBounds, {
          color: '#F5222D',
          weight: 2,
        }).addTo(map!)

        markers.push(startMarker, endMarker)
      }
    }
  })

  if (allLatLngs.length > 0) {
    map.fitBounds(allLatLngs, { padding: [50, 50] })
  }
}

function clearMap() {
  trajectoryLayers.forEach(layer => map?.removeLayer(layer))
  markers.forEach(marker => map?.removeLayer(marker))
  trajectoryLayers = []
  markers = []
}

function highlightTrajectory(selected: TrajectoryPrediction | null) {
  trajectoryLayers.forEach((layer, index) => {
    if (selected && props.trajectories[index] === selected) {
      layer.setStyle({ weight: 5, opacity: 1 })
    } else {
      layer.setStyle({ weight: 3, opacity: 0.8 })
    }
  })
}

function findNearestPoint(trajectory: TrajectoryPrediction, point: L.LatLng): TrajectoryPoint | null {
  let nearest: TrajectoryPoint | null = null
  let minDist = Infinity

  trajectory.points.forEach(p => {
    const dist = Math.sqrt(
      Math.pow(p.lat - point.lat, 2) + Math.pow(p.lng - point.lng, 2)
    )
    if (dist < minDist) {
      minDist = dist
      nearest = p
    }
  })

  return nearest
}

function getModeColor(mode: string): string {
  const colors: Record<string, string> = {
    walk: '#4A90E2',
    bike: '#52C41A',
    bus: '#FA8C16',
    car: '#F5222D',
    subway: '#722ED1',
    train: '#13C2C2',
    airplane: '#EB2F96',
  }
  return colors[mode] || '#00f0ff'
}

function getModeName(mode: string): string {
  const names: Record<string, string> = {
    walk: '步行',
    bike: '自行车',
    bus: '公交',
    car: '汽车/出租',
    subway: '地铁',
    train: '火车',
    airplane: '飞机',
  }
  return names[mode] || mode
}
</script>

<style scoped>
.map-view {
  width: 100%;
  height: 100%;
  min-height: 600px;
  background-color: transparent;
  border-radius: 8px;
  overflow: hidden;
}

:deep(.leaflet-popup-content-wrapper) {
  background: rgba(10, 16, 29, 0.85); 
  border: 1px solid rgba(0, 240, 255, 0.5); 
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.3); 
  border-radius: 4px;
  color: #fff;
  backdrop-filter: blur(4px);
}

:deep(.leaflet-popup-tip) {
  background: rgba(10, 16, 29, 0.85);
  border: 1px solid rgba(0, 240, 255, 0.5);
  border-top: none;
  border-left: none;
  box-shadow: 2px 2px 5px rgba(0, 240, 255, 0.2);
}

:deep(.leaflet-popup-content) {
  margin: 12px 16px;
  line-height: 1.6;
}

.cyber-popup {
  font-family: 'Courier New', Courier, monospace;
  font-size: 13px;
  letter-spacing: 0.5px;
}

.cyber-popup .mode-title {
  font-size: 15px;
  font-weight: bold;
  text-shadow: 0 0 5px currentColor; 
  display: inline-block;
  margin-bottom: 4px;
  border-bottom: 1px solid rgba(255,255,255,0.2);
  padding-bottom: 2px;
}

.cyber-popup .info-label {
  color: #a0aabf;
}

:deep(.leaflet-control-attribution) {
  background: rgba(0, 0, 0, 0.5) !important;
  color: #888 !important;
}
:deep(.leaflet-control-attribution a) {
  color: #00f0ff !important;
}
</style>