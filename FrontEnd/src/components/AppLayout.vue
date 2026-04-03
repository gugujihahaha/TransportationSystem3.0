<template>
  <div class="app-container dark-theme">
    <header class="top-header">
      <div class="header-left">
        <img alt="Logo" class="logo" src="@/assets/logo.svg" />
        <h1 class="project-title">多模态交通方式智能识别系统</h1>
      </div>

      <nav class="header-center">
        <router-link to="/" class="nav-item" active-class="active">首页总览</router-link>
        <router-link to="/dashboard" class="nav-item" active-class="active">数据与实验驾驶舱</router-link>
        
        <div class="nav-item dropdown" :class="{ active: $route.path.includes('/verification') }">
          应用验证 <span class="arrow">▼</span>
          <div class="dropdown-content">
            <router-link to="/verification/congestion">北京典型拥堵路段交通方式构成解析</router-link>
            <router-link to="/verification/green">绿色出行行为量化评估与碳减排核算</router-link>
          </div>
        </div>
        
        <router-link to="/tech" class="nav-item" active-class="active">技术支撑</router-link>
        
        <router-link to="/model-comparison" class="nav-item" active-class="active">消融实验与模型对比</router-link>
        <router-link to="/user" class="nav-item" active-class="active">用户中心</router-link>
      </nav>

      <div class="header-right">
        <div class="time-display">{{ currentTime }}</div>
        <div class="user-greeting">
          <span class="greeting-text">Hi, {{ authStore.username || '研究员' }}</span>
          <button @click="handleLogout" class="logout-btn">退出</button>
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
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

// 动态时间逻辑
const currentTime = ref('')
let timer: number

const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  })
}

onMounted(() => {
  updateTime()
  timer = window.setInterval(updateTime, 1000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})

const handleLogout = () => {
  authStore.logout()
  router.push('/login')
}
</script>

<style scoped>
/* 全局深色大屏主题 */
.app-container {
  min-height: 100vh;
  background-color: #0b0f19;
  color: #e2e8f0;
  display: flex;
  flex-direction: column;
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', Arial, sans-serif;
}

.top-header {
  height: 64px;
  background: rgba(15, 23, 42, 0.9);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(56, 189, 248, 0.2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
  z-index: 100;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo {
  height: 32px;
  width: 32px;
}

.project-title {
  font-size: 20px;
  font-weight: 600;
  color: #38bdf8;
  margin: 0;
  letter-spacing: 1px;
  text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
}

.header-center {
  display: flex;
  gap: 24px;
  height: 100%;
}

.nav-item {
  color: #94a3b8;
  text-decoration: none;
  font-size: 15px;
  display: flex;
  align-items: center;
  padding: 0 12px;
  height: 100%;
  border-bottom: 2px solid transparent;
  transition: all 0.3s;
  cursor: pointer;
  position: relative;
}

.nav-item:hover, .nav-item.active {
  color: #38bdf8;
  border-bottom: 2px solid #38bdf8;
  background: linear-gradient(to top, rgba(56, 189, 248, 0.1), transparent);
}

.dropdown {
  position: relative;
}
.arrow {
  font-size: 10px;
  margin-left: 4px;
  transition: transform 0.3s;
}
.dropdown:hover .arrow {
  transform: rotate(180deg);
}
.dropdown-content {
  display: none;
  position: absolute;
  top: 64px;
  left: 50%;
  transform: translateX(-50%);
  background-color: #0f172a;
  min-width: 240px;
  box-shadow: 0 8px 16px rgba(0,0,0,0.5);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-radius: 4px;
  z-index: 1000;
}
.dropdown:hover .dropdown-content {
  display: block;
}
.dropdown-content a {
  color: #e2e8f0;
  padding: 12px 16px;
  text-decoration: none;
  display: block;
  font-size: 14px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.dropdown-content a:hover {
  background-color: rgba(56, 189, 248, 0.1);
  color: #38bdf8;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

.time-display {
  font-family: monospace;
  font-size: 14px;
  color: #0ea5e9;
  background: rgba(14, 165, 233, 0.1);
  padding: 4px 10px;
  border-radius: 4px;
  border: 1px solid rgba(14, 165, 233, 0.3);
}

.user-greeting {
  display: flex;
  align-items: center;
  gap: 12px;
}

.greeting-text {
  font-size: 14px;
  color: #cbd5e1;
}

.logout-btn {
  background: transparent;
  border: 1px solid #ef4444;
  color: #ef4444;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.logout-btn:hover {
  background: #ef4444;
  color: white;
}

.main-content::-webkit-scrollbar {
  width: 6px;
}

.main-content::-webkit-scrollbar-thumb {
  background: rgba(56, 189, 248, 0.3); 
  border-radius: 10px;
}

.main-content::-webkit-scrollbar-track {
  background: rgba(15, 23, 42, 0.5);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>