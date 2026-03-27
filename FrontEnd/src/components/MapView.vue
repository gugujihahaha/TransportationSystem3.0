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
let markers: L.Marker[] = []

onMounted(() => {
  if (!mapContainer.value) return

  map = L.map(mapContainer.value).setView([39.9042, 116.4074], 12)

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
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

  trajectories.forEach((traj) => {
    const points = traj.points.map(p => [p.lat, p.lng])
    const color = getModeColor(traj.predicted_mode)

    const polyline = L.polyline(points, {
      color,
      weight: 3,
      opacity: 0.8,
    }).addTo(map!)

    polyline.on('click', () => {
      emit('select', traj)
    })

    polyline.on('mouseover', (e) => {
      const point = e.latlng
      const nearestPoint = findNearestPoint(traj, point)
      
      if (nearestPoint) {
        const popup = L.popup()
          .setLatLng(point)
          .setContent(`
            <div>
              <strong>${getModeName(traj.predicted_mode)}</strong><br>
              速度: ${nearestPoint.speed?.toFixed(2) || 'N/A'} m/s<br>
              置信度: ${(traj.confidence * 100).toFixed(1)}%
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
      const startMarker = L.circleMarker(traj.points[0], {
        radius: 8,
        fillColor: '#52C41A',
        color: '#fff',
        weight: 2,
      }).addTo(map!)

      const endMarker = L.rectangle(
        [traj.points[traj.points.length - 1], traj.points[traj.points.length - 1]],
        {
          color: '#F5222D',
          weight: 2,
        }
      ).addTo(map!)

      markers.push(startMarker, endMarker)
    }
  })

  if (trajectories.length > 0) {
    const allPoints = trajectories.flatMap(t => t.points.map(p => [p.lat, p.lng]))
    const bounds = L.latLngBounds(
      allPoints.filter((_, i) => i % 2 === 0).map((_, i) => [allPoints[i * 2], allPoints[i * 2 + 1]])
    )
    map.fitBounds(bounds, { padding: [50, 50] })
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
  let nearest = null
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
  return colors[mode] || '#666'
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
}
</style>
