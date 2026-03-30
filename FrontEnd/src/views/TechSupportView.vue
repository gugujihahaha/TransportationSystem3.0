<template>
  <div class="tech-support-view">
    <PageHeader
      tag="答辩备用页"
      title="技术支撑与实现边界"
      description="本页用于承接评委追问，重点说明系统架构、实验细节、已实现能力与未来规划边界。"
    />

    <div class="grid">
      <section class="card">
        <h3>系统架构</h3>
        <ul>
          <li>数据层：轨迹数据、OSM 路网数据、气象环境数据</li>
          <li>模型层：Exp1~Exp4 实验模型</li>
          <li>服务层：后端接口、报告生成代理、导出能力</li>
          <li>前端层：门户页、驾驶舱、应用验证、技术支撑、用户中心</li>
        </ul>
      </section>

      <section class="card">
        <h3>实验细节</h3>
        <ul>
          <li>展示四阶段实验递进逻辑</li>
          <li>展示准确率、Macro-F1、类别级 F1 等关键指标</li>
          <li>后续可接入混淆矩阵与类别错误分析</li>
        </ul>
      </section>

      <section class="card">
        <h3>已实现内容</h3>
        <ul>
          <li>比赛型页面结构与导航骨架</li>
          <li>数据页、实验页、应用验证页、技术支撑页与用户中心入口</li>
          <li>后端健康检查状态展示</li>
        </ul>
      </section>

      <section class="card">
        <h3>后续联调内容</h3>
        <ul>
          <li>拥堵解析页接入真实模型切换与地图分段着色</li>
          <li>绿色出行页接入 Exp1 识别与碳减排核算</li>
          <li>AI 报告双通道机制：大模型 + 规则模板降级</li>
        </ul>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import PageHeader from '@/components/PageHeader.vue'
</script>

<style scoped>
.tech-support-view {
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
  margin: 0 0 16px 0;
  color: #ffffff;
}

.card ul {
  margin: 0;
  padding-left: 20px;
  color: #a7b7d2;
  line-height: 1.9;
  font-size: 14px;
}
</style>