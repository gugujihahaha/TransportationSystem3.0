<template>
  <div class="scene-view">
    <PageHeader
      tag="应用验证 B"
      title="绿色出行行为量化评估与碳减排核算"
      description="该页面用于验证 Exp1 的跨城市泛化能力。第一阶段先完成页面骨架与信息分层，下一阶段接入真实轨迹上传、识别结果与减排核算。"
    />

    <ScopeAlert
      title="适用范围说明"
      variant="success"
      description="本场景固定使用 Exp1。该模型仅依赖轨迹特征，不依赖北京本地 OSM 或天气数据，适用于任意城市轨迹的绿色出行识别与碳减排评估。"
    />

    <div class="scene-layout">
      <div class="left-panel card">
        <h3>场景说明</h3>
        <div class="desc-item">
          <strong>识别目标</strong>
          <p>将步行、骑行、公交、地铁等判定为绿色出行，将私家车等高碳排方式归为非绿色出行。</p>
        </div>
        <div class="desc-item">
          <strong>当前模型</strong>
          <p>Exp1 通用模型，适合展示跨城市泛化能力，不混入北京限定能力。</p>
        </div>
        <div class="desc-item">
          <strong>第一阶段状态</strong>
          <p>先保留为展示骨架，第二阶段接入轨迹上传、结果返回、减排统计与 AI 绿色报告。</p>
        </div>
      </div>

      <div class="center-panel card">
        <h3>地图与轨迹区（预留）</h3>
        <div class="map-placeholder">
          <div class="placeholder-title">后续接入任意城市轨迹地图展示</div>
          <p>绿色出行段将以生态绿高亮，非绿色出行段以灰色显示，并支持轨迹回放与结果联动。</p>
        </div>
      </div>

      <div class="right-panel card">
        <h3>结果看板（预留）</h3>
        <div class="metric-card" v-for="item in metrics" :key="item.title">
          <span class="metric-title">{{ item.title }}</span>
          <strong class="metric-value">{{ item.value }}</strong>
          <p>{{ item.desc }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import PageHeader from '@/components/PageHeader.vue'
import ScopeAlert from '@/components/ScopeAlert.vue'

const metrics = [
  {
    title: '绿色里程',
    value: '待接入',
    desc: '后续根据识别出的步行、骑行、公交、地铁段自动累计。',
  },
  {
    title: '非绿色里程',
    value: '待接入',
    desc: '用于与绿色里程形成对比，展示日常出行结构。',
  },
  {
    title: '碳减排总量',
    value: '待接入',
    desc: '后续按预设碳排放因子完成核算，并提供等价换算结果。',
  },
  {
    title: 'AI 绿色报告',
    value: '待接入',
    desc: '后续调用后端代理接口生成个性化绿色出行洞察与激励报告。',
  },
]
</script>

<style scoped>
.scene-view {
  padding: 24px 28px 32px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.scene-layout {
  display: grid;
  grid-template-columns: 320px 1fr 360px;
  gap: 18px;
}

.card {
  border-radius: 18px;
  background: rgba(18, 25, 42, 0.92);
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 20px;
}

.card h3 {
  margin: 0 0 18px 0;
  color: #ffffff;
  font-size: 18px;
}

.desc-item + .desc-item {
  margin-top: 16px;
}

.desc-item strong {
  color: #ffffff;
  font-size: 14px;
}

.desc-item p {
  margin: 8px 0 0 0;
  color: #a5b8d7;
  line-height: 1.75;
  font-size: 13px;
}

.map-placeholder {
  min-height: 520px;
  border-radius: 16px;
  border: 1px dashed rgba(82, 196, 26, 0.35);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(82, 196, 26, 0.04);
  text-align: center;
  padding: 24px;
}

.placeholder-title {
  color: #ffffff;
  font-size: 18px;
  font-weight: 600;
}

.map-placeholder p {
  margin: 12px 0 0 0;
  color: #9db0cf;
  line-height: 1.7;
  max-width: 480px;
}

.metric-card + .metric-card {
  margin-top: 14px;
}

.metric-card {
  padding: 16px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
}

.metric-title {
  display: block;
  color: #8fa6cb;
  font-size: 13px;
}

.metric-value {
  display: block;
  margin-top: 6px;
  color: #7fe27f;
  font-size: 22px;
}

.metric-card p {
  margin: 10px 0 0 0;
  color: #a7b8d4;
  font-size: 13px;
  line-height: 1.75;
}

@media (max-width: 1360px) {
  .scene-layout {
    grid-template-columns: 1fr;
  }
}
</style>