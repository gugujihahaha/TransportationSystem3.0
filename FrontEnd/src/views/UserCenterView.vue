<template>
  <div class="user-center-view">
    <PageHeader
      tag="系统完整性"
      title="用户中心"
      description="本页不作为答辩主讲内容，主要用于体现系统具备持续使用逻辑：上传、分析、报告与导出记录可沉淀。"
    />

    <div class="grid">
      <div class="card" v-for="item in modules" :key="item.title">
        <h3>{{ item.title }}</h3>
        <p>{{ item.desc }}</p>
        <el-empty :description="item.placeholder" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import PageHeader from '@/components/PageHeader.vue'

const modules = [
  {
    title: '我的分析记录',
    desc: '按时间查看使用过的场景、模型与结果摘要。',
    placeholder: '后续接入历史分析记录',
  },
  {
    title: '我的上传数据',
    desc: '保存上传过的轨迹文件及处理状态。',
    placeholder: '后续接入上传文件记录',
  },
  {
    title: '我的 AI 报告',
    desc: '查看历史拥堵解析报告和绿色出行报告。',
    placeholder: '后续接入 AI 报告记录',
  },
  {
    title: '我的导出文件',
    desc: '保存导出的 PDF 或图片记录。',
    placeholder: '后续接入导出文件记录',
  },
]
</script>

<style scoped>
.user-center-view {
  padding: 24px 28px 32px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}

.card {
  padding: 22px;
  border-radius: 18px;
  background: rgba(18, 25, 42, 0.92);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.card h3 {
  margin: 0;
  color: #ffffff;
}

.card p {
  margin: 12px 0 0 0;
  color: #a7b7d2;
  line-height: 1.75;
  font-size: 14px;
}
</style>