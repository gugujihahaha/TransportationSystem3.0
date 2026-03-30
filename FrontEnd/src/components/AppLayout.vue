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
          :class="{ active: currentPath === item.path || (item.path !== '/' && currentPath.startsWith(item.path)) }"
        >
          <el-icon :size="22">
            <component :is="item.icon" />
          </el-icon>
          <span class="nav-tooltip">{{ item.label }}</span>
        </router-link>
      </div>
    </div>

    <div class="main-content">
      <div class="header">
        <div class="header-left">
          <h2 class="page-title">{{ pageTitle }}</h2>
          
          <el-tag v-if="currentPath.includes('congestion')" type="danger" effect="dark" class="model-scope-tag">
            <el-icon><Location /></el-icon> 提示：当前场景依赖北京 OSM 及天气数据，限定使用 Exp2-Exp4 模型
          </el-tag>
          <el-tag v-if="currentPath.includes('green')" type="success" effect="dark" class="model-scope-tag">
            <el-icon><Compass /></el-icon> 提示：当前场景调用 Exp1 纯轨迹基线模型，具备全国泛化跨城可用能力
          </el-tag>
        </div>
        
        <div class="header-right">
          <div class="status-indicator">
            <div class="status-dot" :class="{ connected: backendConnected }"></div>
            <span class="status-text">{{ backendConnected ? '后端引擎就绪' : '后端未连接' }}</span>
          </div>
          <div class="divider"></div>
          <div class="user-system">
            <template v-if="!isLogin">
              <span class="visitor-text">当前模式：游客体验</span>
              <el-button type="primary" size="small" @click="isLogin = true">系统登录</el-button>
            </template>
            <template v-else>
              <el-avatar :size="28" style="background: #4A90E2; font-size: 14px;">T</el-avatar>
              <span class="user-name">TrafficRec 评委</span>
              <el-button link type="info" @click="isLogin = false">退出</el-button>
            </template>
          </div>
        </div>
      </div>

      <div class="page-content">
        <router-view v-slot="{ Component }">
          <keep-alive include="DataDashboard">
            <component :is="Component" />
          </keep-alive>
        </router-view>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { House, DataBoard, MapLocation, Bicycle, Cpu, Location, Compass } from '@element-plus/icons-vue'

const route = useRoute()
const isLogin = ref(false)

const navItems = [
  { path: '/', label: '首页总览', icon: House },
  { path: '/dashboard', label: '数据与实验驾驶舱', icon: DataBoard },
  { path: '/congestion-analysis', label: '应用验证A：拥堵溯源', icon: MapLocation },
  { path: '/green-travel', label: '应用验证B：绿色出行', icon: Bicycle },
  { path: '/tech-support', label: '技术支撑与边界', icon: Cpu },
]

const currentPath = computed(() => route.path)
const pageTitle = computed(() => route.meta.title || '多模态交通识别系统')

const backendConnected = ref(false)
let healthCheckInterval: number | null = null

async function checkBackendHealth() {
  try {
    const response = await fetch('http://localhost:8000/api/experiments') // 用一个现有的接口测活
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
  if (healthCheckInterval) clearInterval(healthCheckInterval)
})
</script>

<style scoped>
.app-layout { display: flex; height: 100vh; background: #0b0d12; overflow: hidden; }
.sidebar { width: 64px; background: #161a23; border-right: 1px solid rgba(255, 255, 255, 0.05); display: flex; flex-direction: column; z-index: 10; }
.logo-section { padding: 20px 0; display: flex; justify-content: center; }
.logo-icon { width: 36px; height: 36px; }
.nav-menu { flex: 1; display: flex; flex-direction: column; padding: 20px 0; gap: 12px; }
.nav-item { width: 44px; height: 44px; margin: 0 auto; display: flex; align-items: center; justify-content: center; border-radius: 12px; color: #606266; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; cursor: pointer; text-decoration: none; }
.nav-item:hover { background: rgba(74, 144, 226, 0.1); color: #4A90E2; }
.nav-item.active { background: linear-gradient(135deg, #4A90E2, #357ABD); color: #fff; box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3); }
.nav-tooltip { position: absolute; left: 60px; background: #1a1f2e; color: #fff; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; white-space: nowrap; opacity: 0; pointer-events: none; transition: all 0.2s; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.1); }
.nav-item:hover .nav-tooltip { opacity: 1; transform: translateX(5px); }

.main-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; }
.header { height: 60px; background: rgba(22, 26, 35, 0.8); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255, 255, 255, 0.05); display: flex; align-items: center; justify-content: space-between; padding: 0 24px; z-index: 5; }
.header-left { display: flex; align-items: center; gap: 20px; }
.page-title { margin: 0; font-size: 17px; font-weight: 600; color: #e5eaf3; letter-spacing: 0.5px; }
.model-scope-tag { border: none; font-weight: 500; display: flex; align-items: center; gap: 4px;}

.header-right { display: flex; align-items: center; gap: 20px; }
.status-indicator { display: flex; align-items: center; gap: 8px; background: rgba(0,0,0,0.2); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.05);}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #F56C6C; box-shadow: 0 0 8px #F56C6C; transition: background 0.3s;}
.status-dot.connected { background: #67C23A; box-shadow: 0 0 8px #67C23A; }
.status-text { font-size: 12px; color: #909399; font-family: monospace;}
.divider { width: 1px; height: 20px; background: rgba(255,255,255,0.1); }
.user-system { display: flex; align-items: center; gap: 12px; }
.visitor-text { font-size: 13px; color: #909399; }
.user-name { font-size: 13px; color: #e5eaf3; font-weight: 500; }

.page-content { flex: 1; overflow: hidden; display: flex; flex-direction: column; padding: 24px; box-sizing: border-box;}
</style>