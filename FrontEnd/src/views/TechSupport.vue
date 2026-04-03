<template>
  <div class="tech-support-container">
    <div class="tech-bg">
      <div class="grid-overlay"></div>
      <div class="glow-sphere"></div>
    </div>

    <div class="content-wrapper" :class="{ 'is-loaded': isLoaded }">
      
      <header class="page-header">
        <h1 class="main-title">
          <span class="text-glow">系统工程架构与防御沙盘</span>
        </h1>
        <p class="description">
          System Engineering Architecture & Q&A Defense Sandbox <br>
          全链路多模态数据处理流、底层核心技术栈，以及针对真实应用场景的系统局限性应对方案。
        </p>
      </header>

      <section class="glass-panel mb-30">
        <h2 class="section-title"><span class="icon">🔄</span> 多模态数据流向拓扑 (Data Pipeline)</h2>
        <div class="pipeline-container">
          <div class="pipeline-node data-layer">
            <div class="node-header">L1: 多源数据层 (Data Ingestion)</div>
            <div class="node-body">
              <span class="tech-tag">GPS 轨迹解析</span>
              <span class="tech-tag">OSM 路网拓扑抽取</span>
              <span class="tech-tag">历史气象对齐</span>
            </div>
          </div>
          
          <div class="pipeline-arrow"><el-icon><ArrowRightBold /></el-icon></div>

          <div class="pipeline-node engine-layer">
            <div class="node-header text-gold">L2: 核心推理引擎 (AI Engine)</div>
            <div class="node-body">
              <span class="tech-tag gold">49维特征融合矩阵</span>
              <span class="tech-tag gold">双路 Attention 机制</span>
              <span class="tech-tag gold">Focal Loss 优化器</span>
            </div>
          </div>

          <div class="pipeline-arrow"><el-icon><ArrowRightBold /></el-icon></div>

          <div class="pipeline-node service-layer">
            <div class="node-header text-cyan">L3: 业务服务层 (Business Logic)</div>
            <div class="node-body">
              <span class="tech-tag cyan">FastAPI 推理调度</span>
              <span class="tech-tag cyan">星火 LLM 报告生成</span>
              <span class="tech-tag cyan">SQLite 历史资产存储</span>
            </div>
          </div>

          <div class="pipeline-arrow"><el-icon><ArrowRightBold /></el-icon></div>

          <div class="pipeline-node ui-layer">
            <div class="node-header text-green">L4: 态势展现层 (Visualization)</div>
            <div class="node-body">
              <span class="tech-tag green">ECharts 宏观大屏</span>
              <span class="tech-tag green">Leaflet 时空溯源地图</span>
              <span class="tech-tag green">动态雷达交互栈</span>
            </div>
          </div>
        </div>
      </section>

      <div class="dual-column">
        
        <section class="glass-panel flex-1">
          <h2 class="section-title"><span class="icon">🛠️</span> 核心技术栈全景 (Tech Stack)</h2>
          <div class="tech-grid">
            <div class="stack-card">
              <div class="stack-icon" style="color: #ee4c2c;">🔥</div>
              <div class="stack-name">PyTorch</div>
              <div class="stack-role">底层深度学习框架</div>
            </div>
            <div class="stack-card">
              <div class="stack-icon" style="color: #009688;">⚡</div>
              <div class="stack-name">FastAPI</div>
              <div class="stack-role">高性能异步后端驱动</div>
            </div>
            <div class="stack-card">
              <div class="stack-icon" style="color: #4fc08d;">🟢</div>
              <div class="stack-name">Vue 3 (Composition)</div>
              <div class="stack-role">响应式大屏前端框架</div>
            </div>
            <div class="stack-card">
              <div class="stack-icon" style="color: #e44d26;">📊</div>
              <div class="stack-name">Apache ECharts</div>
              <div class="stack-role">数据科学可视化引擎</div>
            </div>
            <div class="stack-card">
              <div class="stack-icon" style="color: #2196f3;">🗺️</div>
              <div class="stack-name">Leaflet.js + OSM</div>
              <div class="stack-role">空间路网渲染引擎</div>
            </div>
            <div class="stack-card">
              <div class="stack-icon" style="color: #00a8ff;">💬</div>
              <div class="stack-name">星火大模型 (Spark)</div>
              <div class="stack-role">LLM 流式研判报告</div>
            </div>
          </div>
        </section>

        <section class="glass-panel flex-1 defense-section">
          <h2 class="section-title text-warning"><span class="icon">🛡️</span> 答辩防御沙盘 (Limitations & Solutions)</h2>
          <p class="defense-intro">本模块用于直面系统在极度复杂真实场景下的局限性，并提供清晰的工程与算法兜底策略。</p>
          
          <div class="defense-cards">
            <div class="defense-card">
              <div class="q-part">
                <span class="label">⚠️ 潜在提问 (边界泛化)：</span>
                模型在北京市训练，如果跨城部署到山城（如重庆），精度会大幅衰减吗？
              </div>
              <div class="a-part">
                <span class="label text-cyan">💡 防御策略 (解耦与迁移)：</span>
                当前系统架构已完全解耦“特征工程”与“推理模块”。在跨城部署时，只需替换目标城市的 OSM 底图，利用少样本进行<strong>迁移学习 (Transfer Learning)</strong> 即可快速对齐空间拓扑，极大降低了重新训练的算力成本。
              </div>
            </div>

            <div class="defense-card mt-15">
              <div class="q-part">
                <span class="label">⚠️ 潜在提问 (极端长尾分布)：</span>
                如果遭遇特大暴雨导致全城瘫痪、地铁停运，模型还能准确认知出行方式吗？
              </div>
              <div class="a-part">
                <span class="label text-gold">💡 防御策略 (时序惩罚权重)：</span>
                极端天气下多模态特征的联合分布必然发生偏移 (Distribution Shift)。目前的 Exp3 仅做了气象特征的静态拼接。下一代版本架构设计中，我们将引入<strong>实时气象动态惩罚权重</strong>，通过调整 Loss 函数对异常时空特征的敏感度，避免模型在此类绝对长尾场景下过拟合。
              </div>
            </div>
          </div>
        </section>

      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ArrowRightBold } from '@element-plus/icons-vue'

const isLoaded = ref(false)

onMounted(() => {
  setTimeout(() => {
    isLoaded.value = true
  }, 100)
})
</script>

<style scoped>
.tech-support-container {
  min-height: 100vh;
  background-color: #070b19;
  color: #fff;
  position: relative;
  overflow-x: hidden;
  font-family: 'Rajdhani', 'Microsoft YaHei', sans-serif;
  padding: 40px 20px 80px;
}

/* 背景特效 */
.tech-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
.grid-overlay { position: absolute; inset: 0; background-image: linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px); background-size: 40px 40px; opacity: 0.5; }
.glow-sphere { position: absolute; top: 20%; left: 50%; transform: translateX(-50%); width: 60vw; height: 60vw; background: radial-gradient(circle, rgba(0, 240, 255, 0.08) 0%, transparent 60%); border-radius: 50%; filter: blur(60px); }

.content-wrapper {
  position: relative;
  z-index: 1;
  max-width: 1400px;
  margin: 0 auto;
  opacity: 0;
  transform: translateY(20px);
  transition: all 0.8s ease-out;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.content-wrapper.is-loaded { opacity: 1; transform: translateY(0); }

/* Header */
.page-header { text-align: center; margin-bottom: 20px; padding-top: 10px;}
.main-title { font-size: 3rem; font-weight: 900; margin-bottom: 10px; }
.text-glow { color: #fff; text-shadow: 0 0 20px rgba(0, 240, 255, 0.6); }
.description { color: #94a3b8; font-size: 1.1rem; line-height: 1.6; max-width: 800px; margin: 0 auto; }

/* 玻璃面板 */
.glass-panel {
  background: rgba(13, 20, 40, 0.6);
  border: 1px solid rgba(0, 240, 255, 0.15);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 25px 30px;
  box-shadow: inset 0 0 20px rgba(0,240,255,0.02);
}
.mb-30 { margin-bottom: 20px; }
.section-title { font-size: 1.3rem; font-weight: bold; margin: 0 0 25px 0; display: flex; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px;}
.section-title .icon { margin-right: 10px; }

/* ================= 模块一：Pipeline 拓扑 ================= */
.pipeline-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 0;
  overflow-x: auto;
}

.pipeline-node {
  flex: 1;
  min-width: 220px;
  background: rgba(0,0,0,0.4);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  transition: all 0.3s;
}
.pipeline-node:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 20px rgba(0,240,255,0.1);
  border-color: rgba(0,240,255,0.4);
}

.node-header { font-size: 1.05rem; font-weight: bold; margin-bottom: 15px; color: #e2e8f0; text-align: center;}
.node-body { display: flex; flex-direction: column; gap: 10px; width: 100%;}
.tech-tag { 
  background: rgba(255,255,255,0.05); 
  border: 1px solid rgba(255,255,255,0.1); 
  padding: 8px 10px; 
  border-radius: 4px; 
  font-size: 0.9rem; 
  text-align: center;
  color: #cbd5e1;
}

/* 定制化节点色彩 */
.text-gold { color: #ffb800; }
.text-cyan { color: #00f0ff; }
.text-green { color: #00ff88; }

.tech-tag.gold { background: rgba(255,184,0,0.1); border-color: rgba(255,184,0,0.3); color: #ffb800; }
.tech-tag.cyan { background: rgba(0,240,255,0.1); border-color: rgba(0,240,255,0.3); color: #00f0ff; }
.tech-tag.green { background: rgba(0,255,136,0.1); border-color: rgba(0,255,136,0.3); color: #00ff88; }

.pipeline-arrow {
  color: #00f0ff;
  font-size: 24px;
  padding: 0 15px;
  animation: pulseMove 2s infinite;
}
@keyframes pulseMove {
  0% { transform: translateX(-5px); opacity: 0.5; }
  50% { transform: translateX(5px); opacity: 1; }
  100% { transform: translateX(-5px); opacity: 0.5; }
}

/* ================= 模块二：双列布局 ================= */
.dual-column {
  display: flex;
  gap: 25px;
}
.flex-1 { flex: 1; }

/* 技术栈网格 */
.tech-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
}
.stack-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 8px;
  padding: 20px 15px;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  transition: all 0.3s;
}
.stack-card:hover {
  background: rgba(255,255,255,0.05);
  border-color: rgba(255,255,255,0.2);
  transform: scale(1.02);
}
.stack-icon { font-size: 2.2rem; margin-bottom: 10px; }
.stack-name { font-size: 1.1rem; font-weight: bold; color: #e2e8f0; margin-bottom: 5px;}
.stack-role { font-size: 0.85rem; color: #64748b; }

/* ================= 模块三：防御沙盘 ================= */
.defense-section {
  border-color: rgba(230, 162, 60, 0.3);
  background: linear-gradient(135deg, rgba(13, 20, 40, 0.6) 0%, rgba(230, 162, 60, 0.05) 100%);
}
.text-warning { color: #E6A23C; border-bottom-color: rgba(230, 162, 60, 0.3) !important;}
.defense-intro { font-size: 0.95rem; color: #a3a6ad; margin-bottom: 20px; line-height: 1.6;}

.defense-cards { display: flex; flex-direction: column; gap: 20px; }
.defense-card {
  background: rgba(0,0,0,0.4);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 8px;
  overflow: hidden;
}
.q-part {
  background: rgba(255,255,255,0.05);
  padding: 15px 20px;
  font-size: 0.95rem;
  color: #e2e8f0;
  line-height: 1.6;
}
.q-part .label { color: #ff6b6b; font-weight: bold; }

.a-part {
  padding: 15px 20px;
  font-size: 0.95rem;
  color: #cbd5e1;
  line-height: 1.7;
  border-top: 1px solid rgba(255,255,255,0.05);
}
.a-part .label { font-weight: bold; display: block; margin-bottom: 5px;}
.a-part strong { color: #fff; }

@media (max-width: 1200px) {
  .pipeline-container { flex-direction: column; gap: 15px; }
  .pipeline-arrow { transform: rotate(90deg); margin: 10px 0;}
  @keyframes pulseMove {
    0% { transform: translateY(-5px) rotate(90deg); opacity: 0.5; }
    50% { transform: translateY(5px) rotate(90deg); opacity: 1; }
    100% { transform: translateY(-5px) rotate(90deg); opacity: 0.5; }
  }
  .dual-column { flex-direction: column; }
}
</style>