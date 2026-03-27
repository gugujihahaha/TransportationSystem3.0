<template>
  <div class="stat-card">
    <div class="stat-icon" :style="{ background: iconBg }">
      <el-icon :size="32" :color="iconColor">
        <component :is="iconComponent" />
      </el-icon>
    </div>
    <div class="stat-content">
      <div class="stat-label">{{ label }}</div>
      <div class="stat-value">{{ formattedValue }}</div>
      <div v-if="trend" class="stat-trend" :class="trendClass">
        <el-icon><component :is="trendIcon" /></el-icon>
        {{ trendText }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

interface Props {
  label: string
  value: number | string
  icon: string
  iconBg?: string
  iconColor?: string
  trend?: 'up' | 'down' | 'stable'
  trendValue?: number
}

const props = withDefaults(defineProps<Props>(), {
  iconBg: '#409EFF',
  iconColor: '#fff',
})

const iconComponent = computed(() => (ElementPlusIconsVue as any)[props.icon] || ElementPlusIconsVue.DataAnalysis)

const formattedValue = computed(() => {
  if (typeof props.value === 'number') {
    return props.value.toLocaleString()
  }
  return props.value
})

const trendClass = computed(() => {
  if (!props.trend) return ''
  return `trend-${props.trend}`
})

const trendIcon = computed(() => {
  if (props.trend === 'up') return ElementPlusIconsVue.ArrowUp
  if (props.trend === 'down') return ElementPlusIconsVue.ArrowDown
  return ElementPlusIconsVue.Minus
})

const trendText = computed(() => {
  if (!props.trend || !props.trendValue) return ''
  const sign = props.trend === 'up' ? '+' : ''
  return `${sign}${props.trendValue}%`
})
</script>

<style scoped>
.stat-card {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.08);
  transition: all 0.3s;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}

.stat-icon {
  width: 64px;
  height: 64px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.stat-content {
  flex: 1;
  min-width: 0;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 28px;
  font-weight: 600;
  color: #e5e8eb;
  line-height: 1.2;
  margin-bottom: 4px;
}

.stat-trend {
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 4px;
}

.trend-up {
  color: #67C23A;
}

.trend-down {
  color: #F56C6C;
}

.trend-stable {
  color: #909399;
}
</style>
