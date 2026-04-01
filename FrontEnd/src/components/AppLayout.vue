<template>
  <div class="screen-wrapper">
    <div class="tech-bg"></div>

    <header class="screen-header">
      <div class="header-left">
        <div class="nav-item" :class="{ active: currentPath === '/congestion-analysis' }" @click="$router.push('/congestion-analysis')">
          <el-icon><Location /></el-icon> 拥堵时空溯源
        </div>
        <div class="nav-item" :class="{ active: currentPath === '/green-travel' }" @click="$router.push('/green-travel')">
          <el-icon><DataAnalysis /></el-icon> 绿色碳普惠
        </div>
      </div>

      <div class="header-center" style="cursor: pointer;" @click="$router.push('/')" title="返回首页总览">
        <h1 class="glow-title">TrafficRec 多模态出行感知引擎</h1>
        <div class="header-decoration">
          <span class="dec-line left"></span>
          <span class="dec-point"></span>
          <span class="dec-line right"></span>
        </div>
      </div>

      <div class="header-right">
        <div class="nav-item" :class="{ active: currentPath === '/dashboard' }" @click="$router.push('/dashboard')">
          <el-icon><DataBoard /></el-icon> 数据态势感知
        </div>
        
        <div class="user-control-area" v-if="authStore.isAuthenticated()">
          <div class="nav-item user-link" @click="$router.push('/user-center')" :class="{ active: currentPath === '/user-center' }">
            <el-icon><User /></el-icon> {{ authStore.username }}
          </div>
          <div class="time-box logout-trigger" @click="handleLogout">
            EXIT
          </div>
        </div>
        
        <div class="time-box">
          {{ currentTime }}
        </div>
      </div>
    </header>

    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade-slide" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { Location, DataAnalysis, DataBoard, User } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const currentPath = computed(() => route.path)
const currentTime = ref('')

// 更新时间逻辑
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

const handleLogout = () => {
  if (confirm('确定要断开系统连接并退出吗？')) {
    authStore.logout()
    router.push('/login')
  }
}
</script>

<style scoped>
/* 融合 2.0 风格的 CSS */
.screen-wrapper {
  min-height: 100vh;
  background-color: #030816;
  color: #fff;
  position: relative;
  overflow-x: hidden;
}

.screen-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 0 20px;
  height: 80px;
  background: linear-gradient(to bottom, rgba(0, 20, 50, 0.8), transparent);
  position: relative;
  z-index: 10;
}

.header-left, .header-right {
  display: flex;
  gap: 20px;
  padding-top: 20px;
  flex: 1;
}

.header-right { justify-content: flex-end; }

.header-center {
  flex: 1.5;
  text-align: center;
  padding-top: 15px;
}

.glow-title {
  margin: 0;
  font-size: 28px;
  font-weight: bold;
  color: #fff;
  letter-spacing: 2px;
  text-shadow: 0 0 10px #4A90E2, 0 0 20px #00f0ff;
}

.nav-item {
  color: #84a2d4;
  font-size: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 15px;
  transition: all 0.3s;
  position: relative;
}

.nav-item:hover, .nav-item.active {
  color: #00e5ff;
  text-shadow: 0 0 8px rgba(0, 229, 255, 0.8);
}

.nav-item.active::after {
  content: ''; position: absolute; bottom: -5px; left: 0; width: 100%; height: 2px;
  background: #00e5ff; box-shadow: 0 0 10px #00e5ff;
}

.user-control-area {
  display: flex;
  align-items: center;
  gap: 10px;
}

.logout-trigger {
  color: #ff4d4f !important;
  cursor: pointer;
  border: 1px solid rgba(255, 77, 79, 0.3);
}

.logout-trigger:hover {
  background: rgba(255, 77, 79, 0.1);
  border-color: #ff4d4f;
}

.time-box {
  background: rgba(0, 110, 255, 0.1);
  border: 1px solid rgba(0, 110, 255, 0.3);
  padding: 5px 15px;
  border-radius: 4px;
  font-family: 'Courier New', Courier, monospace;
  color: #00e5ff;
  min-width: 80px;
  text-align: center;
}

/* 原有装饰线 */
.header-decoration {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 5px;
}
.dec-line { width: 100px; height: 1px; background: linear-gradient(to right, transparent, #4A90E2, transparent); }
.dec-point { width: 4px; height: 4px; background: #4A90E2; border-radius: 50%; margin: 0 10px; box-shadow: 0 0 5px #4A90E2; }

.main-content { padding: 20px; position: relative; z-index: 1; }

.fade-slide-enter-active, .fade-slide-leave-active { transition: all 0.3s; }
.fade-slide-enter-from { opacity: 0; transform: translateY(20px); }
.fade-slide-leave-to { opacity: 0; transform: translateY(-20px); }
</style>