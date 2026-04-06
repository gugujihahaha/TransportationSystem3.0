<template>
  <div class="fui-home-container">
    <div class="tech-bg">
      <div class="grid-overlay"></div>
      <div class="glow-sphere-1"></div>
      <div class="glow-sphere-2"></div>
    </div>

    <div class="content-wrapper" :class="{ 'is-loaded': isLoaded }">

        <!-- <PageHeader title="多源数据融合驱动的城市交通拥堵溯源与绿色出行评估" subtitle="基于轨迹-路网-天气多模态数据与递进式深度学习实验" /> -->
      
      <div class="data-scale-bar">
        <div class="stat-item">
          <span class="stat-label">处理轨迹点</span>
          <span class="stat-value">1500 万+</span>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
          <span class="stat-label">特征维度</span>
          <span class="stat-value">49 维</span>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
          <span class="stat-label">递进实验模型</span>
          <span class="stat-value">4 个</span>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
          <span class="stat-label">覆盖交通方式</span>
          <span class="stat-value">6 类</span>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
          <span class="stat-label">数据时间跨度</span>
          <span class="stat-value">2008-2012 (GeoLife) + 2024 报告</span>
        </div>
      </div>

      <HomePage />

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import HomePage from '@/components/HomePage.vue'
import PageHeader from '@/components/PageHeader.vue'
// 控制首屏进场动画
const isLoaded = ref(false)

onMounted(() => {
  setTimeout(() => {
    isLoaded.value = true
  }, 50)
})
</script>

<style scoped>
/* --- 全局与背景样式 --- */
.fui-home-container { 
  min-height: 100vh; 
  background-color: #070b19; 
  color: #fff; 
  position: relative; 
  overflow-x: hidden; 
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif; 
  padding-bottom: 20px; 
}

.tech-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
.grid-overlay { 
  position: absolute; inset: 0; 
  background-image: linear-gradient(rgba(0, 240, 255, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 240, 255, 0.05) 1px, transparent 1px); 
  background-size: 50px 50px; 
  transform: perspective(500px) rotateX(60deg); 
  transform-origin: top; 
  opacity: 0.3; 
}
.glow-sphere-1 { 
  position: absolute; top: -10%; left: -10%; width: 50vw; height: 50vw; 
  background: radial-gradient(circle, rgba(0, 87, 255, 0.15) 0%, transparent 70%); 
  border-radius: 50%; filter: blur(50px); 
}
.glow-sphere-2 { 
  position: absolute; bottom: -20%; right: -10%; width: 60vw; height: 60vw; 
  background: radial-gradient(circle, rgba(0, 240, 255, 0.1) 0%, transparent 70%); 
  border-radius: 50%; filter: blur(50px); 
}

/* 扩宽到 1400px 以减少两侧留白，适配大屏看板风格 */
.content-wrapper { 
  position: relative; 
  z-index: 1; 
  max-width: 1400px; 
  margin: 0 auto; 
  padding: 10px 20px; 
  display: flex;
  flex-direction: column;
  gap: 20px; 
  opacity: 0; 
  transform: translateY(20px); 
  transition: all 1s ease-out; 
}
.content-wrapper.is-loaded { opacity: 1; transform: translateY(0); }

/* --- 1. 学术研究标题区样式 --- */
.research-header {
  text-align: center;
  margin-top: 10px;
  margin-bottom: 16px;
}
.research-header h2 {
  font-size: 24px;
  font-weight: 600;
  color: #e2e8f0;
  text-shadow: 0 0 10px rgba(56, 189, 248, 0.2); /* 微弱的发光 */
  margin-bottom: 6px;
  letter-spacing: 1.5px;
}
.research-header p {
  font-size: 14px;
  color: #94a3b8;
  letter-spacing: 1px;
}

/* --- 2. 顶部数据规模卡片条样式 --- */
.data-scale-bar {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-around;
  align-items: center;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 12px;
  padding: 12px 20px;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  margin-bottom: 5px; /* 让卡片和下面内容的衔接更紧凑 */
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
}

.stat-label {
  font-size: 12px;
  color: #94a3b8;
  font-weight: normal;
}

.stat-value {
  font-size: 18px;
  font-weight: bold;
  color: #38bdf8;
  font-family: 'Din', 'Rajdhani', monospace, 'Microsoft YaHei';
}

.stat-divider {
  width: 1px;
  height: 30px;
  background-color: rgba(56, 189, 248, 0.2);
}

@media (max-width: 1024px) {
  .data-scale-bar {
    justify-content: center;
    gap: 15px;
  }
  .stat-divider {
    display: none;
  }
  .research-header h2 {
    font-size: 20px;
  }
}
</style>