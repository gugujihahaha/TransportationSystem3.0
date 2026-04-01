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
        <div class="time-box">
          {{ currentTime }}
        </div>
        <div class="user-box">
          <div class="avatar-glow"><el-icon><UserFilled /></el-icon></div>
          <span class="username">专家评委</span>
        </div>
      </div>
    </header>

    <main class="screen-main">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <keep-alive include="DataOverview,ModelComparison">
            <component :is="Component" :key="$route.path" />
          </keep-alive>
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Location, DataAnalysis, DataBoard, UserFilled } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()
const currentPath = computed(() => route.path)

// 动态时间显示
const currentTime = ref('')
let timer: any = null
const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleString('zh-CN', { hour12: false })
}

onMounted(() => {
  updateTime()
  timer = setInterval(updateTime, 1000)
})

onUnmounted(() => {
  clearInterval(timer)
})
</script>

<style scoped>
/* 全屏暗黑科技基底 */
.screen-wrapper {
  width: 100vw;
  height: 100vh;
  background-color: #030816;
  overflow: hidden;
  position: relative;
  display: flex;
  flex-direction: column;
  font-family: "DingTalk JinBuTi", "Microsoft YaHei", sans-serif;
}

/* 动态网格背景 (模拟数据空间) */
.tech-bg {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  background-image: 
    linear-gradient(rgba(26, 61, 128, 0.15) 1px, transparent 1px),
    linear-gradient(90deg, rgba(26, 61, 128, 0.15) 1px, transparent 1px);
  background-size: 30px 30px;
  z-index: 0;
  opacity: 0.6;
}
.tech-bg::after {
  content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: radial-gradient(circle at center, transparent 0%, #030816 80%);
}

/* 头部大屏样式 */
.screen-header {
  position: relative;
  z-index: 10;
  height: 80px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  background: url('data:image/svg+xml;utf8,<svg viewBox="0 0 1920 80" xmlns="http://www.w3.org/2000/svg"><path d="M0,0 L1920,0 L1920,50 L1300,50 L1250,80 L670,80 L620,50 L0,50 Z" fill="rgba(11,25,56,0.8)" stroke="%231E488F" stroke-width="2"/></svg>') no-repeat top center;
  background-size: 100% 100%;
  padding: 0 20px;
}

.header-left, .header-right {
  display: flex;
  align-items: center;
  height: 50px;
  gap: 20px;
  width: 30%;
}
.header-right { justify-content: flex-end; }

.header-center {
  width: 40%;
  text-align: center;
  padding-top: 15px;
}

.glow-title {
  margin: 0;
  font-size: 28px;
  font-weight: bold;
  color: #fff;
  letter-spacing: 2px;
  text-shadow: 0 0 10px #4A90E2, 0 0 20px #4A90E2;
}

/* 导航按钮特效 */
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

.time-box {
  color: #00e5ff;
  font-family: monospace;
  font-size: 16px;
  letter-spacing: 1px;
}

.user-box { display: flex; align-items: center; gap: 8px; cursor: pointer;}
.avatar-glow {
  width: 30px; height: 30px; border-radius: 50%;
  background: rgba(0, 229, 255, 0.1); border: 1px solid #00e5ff;
  display: flex; align-items: center; justify-content: center; color: #00e5ff;
  box-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
}
.username { color: #fff; font-size: 14px;}

/* 路由主容器 */
.screen-main {
  position: relative;
  z-index: 10;
  flex: 1;
  width: 100%;
  height: calc(100vh - 80px);
  overflow: hidden;
}

/* 过渡动画 */
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>