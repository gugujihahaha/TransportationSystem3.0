<template>
  <div class="app-container">
    <header class="cyber-header">
      <div class="header-left" @click="$router.push('/')">
        <div class="logo-box">TR</div>
        <div class="title-group">
          <h1 class="glow-title">TrafficRec 智能感知引擎</h1>
          <div class="sub-title">多源数据融合交通智能识别系统</div>
        </div>
      </div>

<nav class="header-nav">
  <div class="nav-item" :class="{ active: currentPath === '/' }" @click="$router.push('/')">首页总览</div>
  <div class="nav-item" :class="{ active: currentPath === '/insight' }" @click="$router.push('/insight')">数据洞察</div>
  
  <div class="nav-item" :class="{ active: currentPath === '/model-compare' }" @click="$router.push('/model-compare')">多模型对比</div>
  <div class="nav-item" :class="{ active: currentPath === '/carbon-heatmap' }" @click="$router.push('/carbon-heatmap')">碳普惠热力</div>
  
  <div class="nav-item" :class="{ active: currentPath === '/history' }" @click="$router.push('/history')">我的记录</div>
  <div class="nav-item" :class="{ active: currentPath === '/tech-support' }" @click="$router.push('/tech-support')">技术支撑</div>
</nav>

      <div class="header-right">
        <div class="sys-time">{{ currentTime }}</div>
        <div class="sys-user" v-if="authStore.isAuthenticated()" @click="$router.push('/user-center')">
          <span class="status-dot"></span>
          <span class="user-name">{{ authStore.username || 'ADMIN' }}</span>
        </div>
      </div>
    </header>

    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
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

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const currentPath = computed(() => route.path)

const currentTime = ref('')
let timer: number

const updateTime = () => {
  const now = new Date()
  const hours = String(now.getHours()).padStart(2, '0')
  const minutes = String(now.getMinutes()).padStart(2, '0')
  const seconds = String(now.getSeconds()).padStart(2, '0')
  currentTime.value = `${hours}:${minutes}:${seconds}`
}

onMounted(() => {
  updateTime()
  timer = window.setInterval(updateTime, 1000)
})

onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #050a19; 
  color: #e2e8f0;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  overflow: hidden;
}

.cyber-header {
  height: 72px;
  flex-shrink: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  background: linear-gradient(180deg, rgba(10, 15, 30, 0.95), rgba(5, 10, 25, 0.8));
  border-bottom: 1px solid rgba(0, 240, 255, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(10px);
  z-index: 1000;
}

/* --- 左侧标题区 --- */
.header-left {
  display: flex;
  align-items: center;
  gap: 15px;
  cursor: pointer;
  flex: 0 0 auto;
}
.logo-box {
  width: 40px;
  height: 40px;
  background: rgba(0, 240, 255, 0.1);
  border: 1px solid #00f0ff;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #00f0ff;
  font-weight: 900;
  font-size: 18px;
  font-family: 'Din', monospace;
  box-shadow: inset 0 0 10px rgba(0, 240, 255, 0.2);
}
.title-group { display: flex; flex-direction: column; }
.glow-title {
  margin: 0;
  font-size: 25px;
  font-weight: 900;
  letter-spacing: 2px;
  color: #ffffff;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
}
.sub-title {
  font-family: 'Din', monospace;
  font-size: 15px;
  color: #00f0ff;
  letter-spacing: 1px;
  opacity: 0.7;
  margin-top: 2px;
}

/* --- 中右侧聚合导航区 --- */
.header-nav {
  flex: 1;
  display: flex;
  justify-content: center; 
  align-items: center;
  gap: 16px;
  padding-right: 80px; 
}
.nav-item {
  position: relative;
  padding: 8px 16px;
  color: #94a3b8;
  font-size: 15px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
  border-radius: 4px;
}
.nav-item:hover {
  color: #e2e8f0;
  background: rgba(0, 240, 255, 0.05);
}
.nav-item.active {
  color: #00f0ff;
  background: rgba(0, 240, 255, 0.1);
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.5);
}
/* 底部发光指示条 */
.nav-item::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 0;
  height: 2px;
  background: #00f0ff;
  box-shadow: 0 0 8px #00f0ff;
  transition: width 0.3s ease;
}
.nav-item.active::after {
  width: 100%;
}

/* --- 最右侧状态区 --- */
.header-right {
  display: flex;
  align-items: center;
  gap: 20px;
  flex: 0 0 auto;
  border-left: 1px solid rgba(255, 255, 255, 0.1);
  padding-left: 20px;
}
.sys-time {
  font-family: 'Din', monospace;
  font-size: 20px;
  font-weight: bold;
  color: #00f0ff;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
  letter-spacing: 1px;
}
.sys-user {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: rgba(0, 255, 136, 0.1);
  border: 1px solid rgba(0, 255, 136, 0.3);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s;
}
.sys-user:hover {
  background: rgba(0, 255, 136, 0.2);
  box-shadow: 0 0 10px rgba(0, 255, 136, 0.2);
}
.status-dot {
  width: 6px; height: 6px; 
  background: #00FF88; 
  border-radius: 50%;
  box-shadow: 0 0 8px #00FF88;
  animation: breathe 2s infinite;
}
@keyframes breathe {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.user-name {
  font-size: 13px;
  font-weight: bold;
  color: #e2e8f0;
  font-family: 'Din', monospace, sans-serif;
}

.main-content {
  flex: 1;
  overflow-x: hidden;
  overflow-y: auto;
  position: relative;
}

/* 滚动条美化 */
.main-content::-webkit-scrollbar { width: 6px; }
.main-content::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.3); border-radius: 10px; }
.main-content::-webkit-scrollbar-track { background: rgba(10, 15, 30, 0.5); }

/* 页面切换动画 */
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>