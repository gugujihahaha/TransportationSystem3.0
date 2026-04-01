<template>
  <header class="screen-header">
    <div class="header-inner">
      <div class="header-left">
        <div class="nav-btn" :class="{ active: currentPath === '/congestion-analysis' }" @click="$router.push('/congestion-analysis')">
          <el-icon><Location /></el-icon> 拥堵溯源
        </div>
        <div class="nav-btn" :class="{ active: currentPath === '/green-travel' }" @click="$router.push('/green-travel')">
          <el-icon><DataAnalysis /></el-icon> 绿色出行
        </div>
      </div>

      <div class="header-center" @click="$router.push('/')">
        <h1 class="glow-title">TrafficRec 感知引擎</h1>
        <div class="header-decoration">
          <span class="dec-line"></span>
          <span class="dec-point"></span>
          <span class="dec-line"></span>
        </div>
      </div>

      <div class="header-right">
        <div class="nav-btn" :class="{ active: currentPath === '/dashboard' }" @click="$router.push('/dashboard')">
          <el-icon><DataBoard /></el-icon> 态势感知
        </div>
        
        <div class="user-info-trigger" v-if="authStore.isAuthenticated()" @click="$router.push('/user-center')">
          <span class="user-status-dot"></span>
          <span class="user-name-text">{{ authStore.username }}</span>
        </div>

        <div class="header-time">{{ currentTime }}</div>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { Location, DataAnalysis, DataBoard } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()
const currentPath = computed(() => route.path)
const currentTime = ref('')

const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleTimeString('zh-CN', { hour12: false })
}

let timer: number
onMounted(() => {
  updateTime()
  timer = window.setInterval(updateTime, 1000)
})
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.screen-header {
  height: 70px;
  width: 100%;
  background: linear-gradient(to bottom, rgba(10, 20, 56, 0.9), rgba(10, 20, 56, 0.3));
  border-bottom: 1px solid rgba(0, 240, 255, 0.2);
  z-index: 100;
  position: relative;
}

.header-inner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 30px;
  height: 100%;
}

.header-left, .header-right { display: flex; gap: 15px; align-items: center; flex: 1; }
.header-right { justify-content: flex-end; }
.header-center { text-align: center; cursor: pointer; min-width: 300px; }

.glow-title {
  margin: 0;
  font-size: 24px;
  font-weight: bold;
  letter-spacing: 2px;
  background: linear-gradient(to right, #fff, #00f0ff);
  /* 🚀 修复 CSS 兼容性报错 */
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
}

.nav-btn {
  color: #84a2d4;
  font-size: 14px;
  cursor: pointer;
  padding: 6px 12px;
  border: 1px solid transparent;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  gap: 6px;
}

.nav-btn:hover, .nav-btn.active {
  color: #00e5ff;
  border: 1px solid rgba(0, 229, 255, 0.3);
  background: rgba(0, 229, 255, 0.05);
  box-shadow: inset 0 0 10px rgba(0, 229, 255, 0.1);
}

.user-info-trigger {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 5px 12px;
  background: rgba(0, 229, 255, 0.1);
  border-radius: 4px;
  margin: 0 15px;
}

.user-status-dot {
  width: 6px; height: 6px; background: #22c55e; border-radius: 50%;
  box-shadow: 0 0 8px #22c55e;
}

.user-name-text { font-size: 13px; font-weight: bold; color: #fff; }

.header-time {
  font-family: 'Courier New', monospace;
  color: #00e5ff;
  font-weight: bold;
  font-size: 16px;
  min-width: 80px;
}

.header-decoration { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 2px; }
.dec-line { width: 40px; height: 1px; background: #4A90E2; opacity: 0.5; }
.dec-point { width: 3px; height: 3px; background: #00f0ff; border-radius: 50%; }
</style>