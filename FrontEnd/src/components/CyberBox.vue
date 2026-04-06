<template>
  <div class="cyber-box-wrapper" :style="{ minHeight: height }">
    <svg class="cyber-border-svg" width="100%" height="100%" preserveAspectRatio="none">
      <defs>
        <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        <linearGradient id="cyber-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#00f2ff" stop-opacity="0.8" />
          <stop offset="100%" stop-color="#08425d" stop-opacity="0.2" />
        </linearGradient>
      </defs>
      <polygon 
        points="0,15 15,0 100%,0 100%,calc(100% - 15px) calc(100% - 15px),100% 0,100%"
        class="border-path"
        filter="url(#neon-glow)"
      />
      <polyline points="0,30 0,15 15,0 40,0" class="corner-highlight" />
      <polyline points="100%,calc(100% - 30px) 100%,calc(100% - 15px) calc(100% - 15px),100% calc(100% - 40px),100%" class="corner-highlight" />
    </svg>
    
    <div v-if="title" class="cyber-title">
      <span class="title-text">{{ title }}</span>
      <div class="title-dec"></div>
    </div>

    <div class="cyber-content">
      <slot></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps({
  title: { type: String, default: '' },
  height: { type: String, default: '100%' }
})
</script>

<style scoped>
.cyber-box-wrapper {
  position: relative;
  width: 100%;
  background: rgba(4, 11, 25, 0.65);
  backdrop-filter: blur(10px);
  clip-path: polygon(0 15px, 15px 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%);
  padding: 20px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
}

.cyber-border-svg {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none;
  z-index: 1;
}

.border-path {
  fill: transparent;
  stroke: url(#cyber-grad);
  stroke-width: 1.5;
}

.corner-highlight {
  fill: transparent;
  stroke: #00f2ff;
  stroke-width: 3;
  filter: drop-shadow(0 0 5px #00f2ff);
}

.cyber-title {
  position: relative;
  z-index: 2;
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.title-text {
  color: #fff;
  font-size: 1.1rem;
  font-weight: bold;
  letter-spacing: 2px;
  text-shadow: 0 0 10px rgba(0, 242, 255, 0.8);
  padding-right: 15px;
}

.title-dec {
  flex-grow: 1;
  height: 2px;
  background: linear-gradient(90deg, #00f2ff 0%, transparent 100%);
  opacity: 0.5;
}

.cyber-content {
  position: relative;
  z-index: 2;
  flex-grow: 1;
  width: 100%;
  height: 100%;
}
</style>