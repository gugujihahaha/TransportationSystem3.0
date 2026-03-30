<template>
  <div class="page-header-card">
    <div class="header-main">
      <div class="title-row">
        <span v-if="tag" class="header-tag">{{ tag }}</span>
        <h3>{{ title }}</h3>
      </div>
      <p>{{ description }}</p>
    </div>
    <div v-if="$slots.extra" class="header-extra">
      <slot name="extra" />
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title: string
  description: string
  tag?: string
}

defineProps<Props>()
</script>

<style scoped>
.page-header-card {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  padding: 20px 22px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(26, 31, 46, 0.92), rgba(18, 26, 44, 0.92));
  border: 1px solid rgba(74, 144, 226, 0.18);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
}

.header-main {
  min-width: 0;
}

.title-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-tag {
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(74, 144, 226, 0.14);
  color: #7db7ff;
  font-size: 12px;
  border: 1px solid rgba(74, 144, 226, 0.24);
}

.title-row h3 {
  margin: 0;
  color: #ffffff;
  font-size: 22px;
}

.header-main p {
  margin: 12px 0 0 0;
  color: #9eb0cf;
  font-size: 14px;
  line-height: 1.7;
}

.header-extra {
  flex-shrink: 0;
}
</style>