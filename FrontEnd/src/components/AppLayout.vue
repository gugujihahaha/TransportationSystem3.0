<template>
  <div class="app-layout">
    <div class="sidebar">
      <div class="logo-section">
        <div class="logo-icon">
          <svg viewBox="0 0 40 40" class="logo-svg">
            <circle cx="10" cy="30" r="6" fill="#4A90E2" />
            <circle cx="20" cy="20" r="6" fill="#52C41A" />
            <circle cx="30" cy="10" r="6" fill="#FA8C16" />
            <line x1="10" y1="30" x2="20" y2="20" stroke="#fff" stroke-width="2" />
            <line x1="20" y1="20" x2="30" y2="10" stroke="#fff" stroke-width="2" />
          </svg>
        </div>
      </div>

      <div class="nav-menu">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ active: currentPath === item.path }"
        >
          <el-icon :size="24">
            <component :is="item.icon" />
          </el-icon>
          <span class="nav-tooltip">{{ item.label }}</span>
        </router-link>
      </div>
    </div>

    <div class="main-content">
      <div class="header">
        <div class="header-title">
          <h2>{{ pageTitle }}</h2>
        </div>
        <div class="header-right">
          <div class="status-indicator">
            <div
              class="status-dot"
              :class="{ connected: backendConnected }"
            ></div>
            <span class="status-text">
              {{ backendConnected ? '后端已连接' : '后端未连接' }}
            </span>
          </div>
        </div>
      </div>

      <div class="page-content">
        <router-view />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import {
  Location,
  DataAnalysis,
  DataBoard,
  InfoFilled,
} from '@element-plus/icons-vue'

const route = useRoute()

const navItems = [
  {
    path: '/',
    label: '地图分析',
    icon: Location,
  },
  {
    path: '/model-comparison',
    label: '模型对比',
    icon: DataAnalysis,
  },
  {
    path: '/data-overview',
    label: '数据概览',
    icon: DataBoard,
  },
  {
    path: '/about',
    label: '关于',
    icon: InfoFilled,
  },
]

const currentPath = computed(() => route.path)

const pageTitle = computed(() => {
  const item = navItems.find(i => i.path === currentPath.value)
  return item ? item.label : '城市出行智能感知平台'
})

const backendConnected = ref(false)

let healthCheckInterval: number | null = null

async function checkBackendHealth() {
  try {
    const response = await fetch('http://localhost:8000/health')
    backendConnected.value = response.ok
  } catch (error) {
    backendConnected.value = false
  }
}

onMounted(() => {
  checkBackendHealth()
  healthCheckInterval = window.setInterval(checkBackendHealth, 5000)
})

onUnmounted(() => {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval)
  }
})
</script>

<style scoped>
.app-layout {
  display: flex;
  height: 100%;
  background: #0f1117;
  overflow: hidden;
}

.sidebar {
  width: 56px;
  background: #1a1f2e;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.logo-section {
  padding: 16px 0;
  display: flex;
  justify-content: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.logo-icon {
  width: 32px;
  height: 32px;
}

.logo-svg {
  width: 100%;
  height: 100%;
}

.nav-menu {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 16px 0;
  gap: 8px;
}

.nav-item {
  width: 40px;
  height: 40px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #909399;
  transition: all 0.3s;
  position: relative;
  cursor: pointer;
  text-decoration: none;
}

.nav-item:hover {
  background: rgba(74, 144, 226, 0.1);
  color: #4A90E2;
}

.nav-item.active {
  background: #4A90E2;
  color: #fff;
}

.nav-tooltip {
  position: absolute;
  left: 48px;
  background: #1a1f2e;
  color: #fff;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  z-index: 1000;
}

.nav-item:hover .nav-tooltip {
  opacity: 1;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.header {
  height: 48px;
  background: #1a1f2e;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  flex-shrink: 0;
}

.header-title h2 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #F56C6C;
  transition: background 0.3s;
}

.status-dot.connected {
  background: #67C23A;
}

.status-text {
  font-size: 12px;
  color: #909399;
}

.page-content {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  background: #0f1117;
  display: flex;
  flex-direction: column;
}
</style>
