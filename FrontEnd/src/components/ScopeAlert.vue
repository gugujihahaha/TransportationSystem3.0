<template>
  <div class="scope-alert" :class="variant">
    <div class="scope-title">{{ title }}</div>
    <div class="scope-content">
      <slot>
        <p>{{ description }}</p>
      </slot>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title: string
  description?: string
  variant?: 'info' | 'warning' | 'success'
}

withDefaults(defineProps<Props>(), {
  variant: 'info',
  description: '',
})
</script>

<style scoped>
.scope-alert {
  padding: 16px 18px;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.03);
}

.scope-alert.info {
  background: rgba(74, 144, 226, 0.08);
  border-color: rgba(74, 144, 226, 0.2);
}

.scope-alert.warning {
  background: rgba(250, 140, 22, 0.1);
  border-color: rgba(250, 140, 22, 0.24);
}

.scope-alert.success {
  background: rgba(82, 196, 26, 0.1);
  border-color: rgba(82, 196, 26, 0.24);
}

.scope-title {
  font-size: 14px;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 8px;
}

.scope-content :deep(p) {
  margin: 0;
  color: #c6d4ed;
  font-size: 13px;
  line-height: 1.7;
}
</style>