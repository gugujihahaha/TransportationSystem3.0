<template>
  <div class="fui-home-container">
    <div class="tech-bg">
      <div class="grid-overlay"></div>
      <div class="glow-sphere-1"></div>
      <div class="glow-sphere-2"></div>
    </div>

    <div class="content-wrapper" :class="{ 'is-loaded': isLoaded }">
      <header class="hero-section">
        <div class="tagline">National Award Level Project</div>
        <h1 class="main-title">
          <span class="text-glow">TrafficRec</span> 
          <span class="sub-text">多模态出行大模型</span>
        </h1>
        <p class="description">
          基于四阶递进式 PyTorch 引擎与星火大模型，精准溯源城市拥堵，构建碳普惠智能评估体系。
        </p>
        
        <div class="action-group">
          <button class="cyber-btn primary" @click="navigateTo('/data-dashboard')">
            <span class="btn-text">进入数据舱</span>
            <span class="btn-glitch"></span>
          </button>
          <button class="cyber-btn secondary" @click="navigateTo('/user-center')">
            <span class="btn-text">驾驶员档案</span>
          </button>
        </div>
      </header>

      <section class="engine-matrix">
        <div 
          v-for="(engine, index) in engines" 
          :key="index" 
          class="engine-card"
          :style="{ animationDelay: `${index * 0.15 + 0.5}s` }"
        >
          <div class="card-border"></div>
          <div class="card-content">
            <div class="engine-icon">{{ engine.icon }}</div>
            <h3 class="engine-name">{{ engine.name }}</h3>
            <p class="engine-desc">{{ engine.desc }}</p>
            <div class="engine-metric">
              <span class="label">当前 F1-Score:</span>
              <span class="value" :class="engine.colorClass">{{ engine.score }}</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const isLoaded = ref(false)

// 四大引擎配置数据 (可根据后续 Exp2-4 的数据进行替换)
const engines = ref([
  {
    icon: '🛰️',
    name: 'Exp1: 纯轨迹基线',
    desc: '基于运动学特征提取，剥离外部依赖，实现轻量级底层识别。',
    score: '0.83', // 占位，刚从你的数据里提取的综合水平
    colorClass: 'text-cyan'
  },
  {
    icon: '🗺️',
    name: 'Exp2: 路网增强',
    desc: '融合 OSM 城市路网拓扑，精准区分公交与私家车共流问题。',
    score: '0.89', // 占位
    colorClass: 'text-blue'
  },
  {
    icon: '⛈️',
    name: 'Exp3: 气象解耦',
    desc: '引入历史/实时天气多维矩阵，消除环境因素导致的预测偏差。',
    score: '0.92', // 占位
    colorClass: 'text-gold'
  },
  {
    icon: '🎯',
    name: 'Exp4: 焦损优化',
    desc: '运用 Focal Loss 解决数据长尾分布不平衡，达到 SOTA 精度。',
    score: '0.95', // 占位
    colorClass: 'text-green'
  }
])

const navigateTo = (path: string) => {
  router.push(path)
}

onMounted(() => {
  // 触发入场动画
  setTimeout(() => {
    isLoaded.value = true
  }, 100)
})
</script>

<style scoped>
/* 核心色彩与变量定义 */
.fui-home-container {
  --theme-dark: #070b19;
  --theme-cyan: #00f0ff;
  --theme-blue: #0057ff;
  --theme-gold: #ffb800;
  --theme-green: #00ff88;
  --glass-bg: rgba(13, 20, 40, 0.65);
  --glass-border: rgba(0, 240, 255, 0.2);
  
  position: relative;
  min-height: 100vh;
  background-color: var(--theme-dark);
  color: #fff;
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 动态深色科技背景 */
.tech-bg {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  z-index: 0;
  overflow: hidden;
}
.grid-overlay {
  width: 200%; height: 200%;
  background-image: 
    linear-gradient(rgba(0, 240, 255, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 240, 255, 0.05) 1px, transparent 1px);
  background-size: 40px 40px;
  transform: perspective(500px) rotateX(60deg) translateY(-100px) translateZ(-200px);
  animation: gridMove 20s linear infinite;
}
.glow-sphere-1, .glow-sphere-2 {
  position: absolute;
  border-radius: 50%;
  filter: blur(120px);
  opacity: 0.4;
  animation: float 10s ease-in-out infinite alternate;
}
.glow-sphere-1 {
  width: 40vw; height: 40vw;
  background: var(--theme-blue);
  top: -10%; left: -10%;
}
.glow-sphere-2 {
  width: 30vw; height: 30vw;
  background: var(--theme-cyan);
  bottom: -10%; right: -5%;
  animation-delay: -5s;
}

/* 主内容区排版与动画 */
.content-wrapper {
  position: relative;
  z-index: 1;
  width: 90%;
  max-width: 1400px;
  padding: 40px 0;
  opacity: 0;
  transform: translateY(30px);
  transition: all 1s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.content-wrapper.is-loaded {
  opacity: 1;
  transform: translateY(0);
}

/* 头部排版 */
.hero-section {
  text-align: center;
  margin-bottom: 60px;
}
.tagline {
  color: var(--theme-cyan);
  font-size: 14px;
  letter-spacing: 4px;
  text-transform: uppercase;
  margin-bottom: 15px;
  font-weight: 600;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
}
.main-title {
  font-size: 4rem;
  font-weight: 900;
  margin-bottom: 20px;
  letter-spacing: 2px;
}
.text-glow {
  background: linear-gradient(90deg, #fff, var(--theme-cyan));
  -webkit-background-clip: text; /* 兼容基于 WebKit 的浏览器 (Chrome, Safari等) */
  background-clip: text;         
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 30px rgba(0, 240, 255, 0.3);
}
.sub-text {
  font-size: 3rem;
  font-weight: 300;
  margin-left: 15px;
  color: #e0e0e0;
}
.description {
  font-size: 1.1rem;
  color: #94a3b8;
  max-width: 700px;
  margin: 0 auto 40px;
  line-height: 1.8;
}

/* 赛博按钮样式 */
.action-group {
  display: flex;
  gap: 20px;
  justify-content: center;
}
.cyber-btn {
  position: relative;
  padding: 15px 40px;
  font-size: 16px;
  font-weight: bold;
  letter-spacing: 2px;
  color: #fff;
  background: transparent;
  border: 1px solid var(--theme-cyan);
  cursor: pointer;
  overflow: hidden;
  transition: all 0.3s ease;
  backdrop-filter: blur(5px);
}
.cyber-btn.primary {
  background: rgba(0, 240, 255, 0.1);
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.3), inset 0 0 20px rgba(0, 240, 255, 0.1);
}
.cyber-btn:hover {
  background: var(--theme-cyan);
  color: #000;
  box-shadow: 0 0 30px var(--theme-cyan);
}
.cyber-btn.secondary {
  border-color: #475569;
}
.cyber-btn.secondary:hover {
  background: #475569;
  color: #fff;
}

/* 毛玻璃引擎卡片矩阵 */
.engine-matrix {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 25px;
  margin-top: 50px;
}
.engine-card {
  position: relative;
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 30px 25px;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  opacity: 0;
  animation: slideUpFade 0.6s ease forwards;
}
.engine-card:hover {
  transform: translateY(-10px) scale(1.02);
  border-color: var(--theme-cyan);
  box-shadow: 0 10px 30px rgba(0, 240, 255, 0.15), inset 0 0 20px rgba(0, 240, 255, 0.05);
}
.engine-icon {
  font-size: 2.5rem;
  margin-bottom: 15px;
  filter: drop-shadow(0 0 8px rgba(255,255,255,0.3));
}
.engine-name {
  font-size: 1.2rem;
  font-weight: 600;
  color: #fff;
  margin-bottom: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  padding-bottom: 10px;
}
.engine-desc {
  font-size: 0.9rem;
  color: #cbd5e1;
  line-height: 1.6;
  margin-bottom: 20px;
  min-height: 45px;
}
.engine-metric {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(0,0,0,0.3);
  padding: 10px 15px;
  border-radius: 6px;
  border-left: 3px solid var(--theme-cyan);
}
.engine-metric .label {
  font-size: 0.85rem;
  color: #94a3b8;
}
.engine-metric .value {
  font-size: 1.2rem;
  font-weight: bold;
  font-family: 'Courier New', Courier, monospace;
}
.text-cyan { color: var(--theme-cyan); }
.text-blue { color: #4dabf7; }
.text-gold { color: var(--theme-gold); }
.text-green { color: var(--theme-green); }

/* 动画定义 */
@keyframes gridMove {
  0% { background-position: 0 0; }
  100% { background-position: 0 40px; }
}
@keyframes float {
  0% { transform: translateY(0) scale(1); }
  100% { transform: translateY(-30px) scale(1.05); }
}
@keyframes slideUpFade {
  from { opacity: 0; transform: translateY(40px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>