<template>
  <div class="scene-view">
    <PageHeader
      tag="应用验证 A"
      title="北京典型拥堵路段交通方式构成解析"
      description="该页面用于验证 OSM 与天气信息在复杂拥堵场景中的作用。第一阶段先搭建展示骨架，下一阶段再把真实模型切换、地图分段着色与 AI 报告联调接入。"
    />

    <ScopeAlert
      title="适用范围说明"
      variant="warning"
      description="本场景聚焦北京本地数据。Exp2~Exp4 依赖北京本地 OSM 与天气信息，当前仅适用于北京地区；Exp1 可作为对照组一起展示。"
    />

    <div class="scene-layout">
      <div class="left-panel card">
        <h3>场景与模型选择</h3>

        <div class="control-block">
          <div class="block-label">预置场景</div>
          <el-select model-value="东三环国贸段（预置案例）" disabled style="width: 100%">
            <el-option label="东三环国贸段（预置案例）" value="default" />
          </el-select>
        </div>

        <div class="control-block">
          <div class="block-label">实验模型</div>
          <div class="model-list">
            <div
              v-for="model in models"
              :key="model.id"
              class="model-item"
              :class="{ warning: model.limited }"
            >
              <div class="model-head">
                <strong>{{ model.name }}</strong>
                <el-tag :type="model.limited ? 'warning' : 'success'" size="small">
                  {{ model.tag }}
                </el-tag>
              </div>
              <p>{{ model.desc }}</p>
            </div>
          </div>
        </div>

        <div class="control-block">
          <div class="block-label">当前阶段说明</div>
          <el-alert type="info" :closable="false">
            第一阶段只完成比赛型页面骨架；下一阶段将在这里接入真实上传、模型预测与 AI 报告接口。
          </el-alert>
        </div>
      </div>

      <div class="center-panel card">
        <h3>地图展示区（预留）</h3>
        <div class="map-placeholder">
          <div class="placeholder-title">这里后续接入 MapView + 北京预置轨迹</div>
          <p>下一阶段会实现：轨迹分段着色、低速段高亮、模型切换联动结果对比。</p>
        </div>
      </div>

      <div class="right-panel card">
        <h3>结果解读区（预留）</h3>
        <div class="result-card" v-for="item in expectedOutputs" :key="item.title">
          <h4>{{ item.title }}</h4>
          <p>{{ item.desc }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import PageHeader from '@/components/PageHeader.vue'
import ScopeAlert from '@/components/ScopeAlert.vue'

const models = [
  {
    id: 'exp1',
    name: 'Exp1 · 纯轨迹基线模型',
    tag: '通用模型',
    limited: false,
    desc: '仅依赖轨迹运动学特征，用作跨城市可迁移基线与本场景对照模型。',
  },
  {
    id: 'exp2',
    name: 'Exp2 · 轨迹 + OSM',
    tag: '北京限定',
    limited: true,
    desc: '引入北京本地 OSM 路网语义信息，增强复杂道路环境下的交通方式判别能力。',
  },
  {
    id: 'exp3',
    name: 'Exp3 · 轨迹 + OSM + 天气',
    tag: '北京限定',
    limited: true,
    desc: '在 OSM 基础上继续引入天气环境变量，解释雨雪等天气扰动对出行方式的影响。',
  },
  {
    id: 'exp4',
    name: 'Exp4 · 方法优化版',
    tag: '北京限定',
    limited: true,
    desc: '在相近特征体系上进一步优化方法，兼顾整体性能与长尾类别识别能力。',
  },
]

const expectedOutputs = [
  {
    title: '模型切换对比',
    desc: '右侧将对比 exp1~exp4 的主要交通方式、置信度、低速段占比与相对上一实验的提升值。',
  },
  {
    title: 'AI 分析报告',
    desc: '后续接入后端代理接口，实现大模型报告与规则模板降级双通道输出。',
  },
  {
    title: 'PDF 导出',
    desc: '后续将地图截图、结果对比表和 AI 报告统一导出，服务答辩与汇报展示。',
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

.control-block + .control-block {
  margin-top: 18px;
}

.block-label {
  font-size: 13px;
  color: #8ea4c8;
  margin-bottom: 10px;
}

.model-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-item {
  padding: 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.model-item.warning {
  border-color: rgba(250, 140, 22, 0.24);
}

.model-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
}

.model-item p {
  margin: 10px 0 0 0;
  color: #a8bad8;
  line-height: 1.7;
  font-size: 13px;
}

.map-placeholder {
  min-height: 520px;
  border-radius: 16px;
  border: 1px dashed rgba(74, 144, 226, 0.35);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(74, 144, 226, 0.04);
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

.result-card + .result-card {
  margin-top: 14px;
}

.result-card {
  padding: 16px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
}

.result-card h4 {
  margin: 0;
  color: #ffffff;
  font-size: 15px;
}

.result-card p {
  margin: 10px 0 0 0;
  color: #a4b7d6;
  font-size: 13px;
  line-height: 1.75;
}

@media (max-width: 1360px) {
  .scene-layout {
    grid-template-columns: 1fr;
  }
}
</style>