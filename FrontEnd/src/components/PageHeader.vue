<!-- src/components/PageHeader.vue -->
<template>
  <div class="page-header">
    <div class="header-left">
      <div class="title-deco"></div>
      <div class="title-group">
        <h1 class="main-title">{{ title }}</h1>
        <p v-if="subtitle" class="subtitle">{{ subtitle }}</p>
      </div>
    </div>
    <div v-if="$slots.actions" class="header-actions">
      <slot name="actions"></slot>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  subtitle: { type: String, default: '' }
})
</script>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 28px;
  padding-bottom: 16px;
  border-bottom: 2px solid rgba(0, 240, 255, 0.2);
}
.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}
.title-deco {
  width: 5px;
  height: 32px;
  background: #00f0ff;
  box-shadow: 0 0 12px #00f0ff;
  border-radius: 2px;
}
.title-group {
  display: flex;
  flex-direction: column;
}
.main-title {
  font-size: 26px;
  font-weight: 700;
  color: #ffffff;
  margin: 0;
  letter-spacing: 1px;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.3);
}
.subtitle {
  font-size: 14px;
  color: #94a3b8;
  margin: 6px 0 0 0;
  letter-spacing: 0.5px;
}
.header-actions {
  display: flex;
  gap: 12px;
}
</style>