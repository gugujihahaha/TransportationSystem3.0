<template>
  <div class="home-container">
    
    <el-row :gutter="24" class="top-row">
      <el-col :span="14">
        <div class="hero-card glass-card">
          <div class="tech-corner top-left"></div>
          <div class="tech-corner bottom-right"></div>
          
          <h1 class="neon-title">
            <span class="glitch-text" data-text="TrafficRec">TrafficRec</span> 多模态感知引擎
          </h1>
          <p class="subtitle cyber-text">DATA RESEARCH COCKPIT // APPLICATION PLATFORM</p>
          <div class="tech-line-horizontal"></div>
          <p class="description">
            本引擎基于深度学习，融合 <span class="highlight">轨迹动力学、OSM 城市空间语义及气象环境数据</span>，
            构建了包含 49 维特征的多模态架构。专为解决复杂路况及长尾类别的交通方式高精度识别而生，全链路赋能拥堵溯源与碳普惠评估。
          </p>
          <div class="action-buttons">
            <el-button class="neon-button primary-glow" size="large" @click="$router.push('/dashboard')">
              <el-icon class="el-icon--left"><DataBoard /></el-icon> 启动数据驾驶舱
            </el-button>
          </div>
        </div>
      </el-col>

      <el-col :span="10">
        <div class="metrics-grid">
          <div class="metric-card glass-card hover-glow">
            <div class="metric-title">系统峰值 Accuracy</div>
            <div class="metric-value neon-blue">92.4%</div>
          </div>
          <div class="metric-card glass-card hover-glow">
            <div class="metric-title">系统峰值 Macro-F1</div>
            <div class="metric-value neon-cyan">89.1%</div>
          </div>
          <div class="metric-card glass-card hover-glow">
            <div class="metric-title">全域识别交通方式</div>
            <div class="metric-value neon-white">08 <span class="unit">类</span></div>
          </div>
          <div class="metric-card glass-card hover-glow">
            <div class="metric-title">实机验证落地场景</div>
            <div class="metric-value neon-white">02 <span class="unit">组</span></div>
          </div>
        </div>
      </el-col>
    </el-row>

    <div class="section-wrapper">
      <div class="cyber-section-title">
        <i class="el-icon-node neon-icon"></i> 核心神经元演进主轴
      </div>
      <div class="timeline-card glass-card">
        <el-steps :active="4" align-center class="cyber-steps">
          <el-step title="Exp1：运动学基线" description="纯轨迹时空特征，剥离地域依赖，验证模型全国泛化能力" />
          <el-step title="Exp2：空间语义融合" description="引入 OSM 城市空间数据，打破轨迹孤岛，重构路段语义" />
          <el-step title="Exp3：全天候气象耦合" description="加入气象维度，增强引擎对雨雪等极端环境扰动的鲁棒性" />
          <el-step title="Exp4：Loss 函数跃迁" description="引入 Focal Loss 优化算法，攻克火车/地铁等长尾分类死角" />
        </el-steps>
      </div>
    </div>

    <div class="section-wrapper">
      <div class="cyber-section-title">
        <i class="el-icon-monitor neon-icon"></i> 真实全息场景验证
      </div>
      <el-row :gutter="24" class="bottom-row">
        <el-col :span="12">
          <div class="scene-card glass-card interactive-card" @click="$router.push('/congestion-analysis')">
            <div class="scan-line"></div>
            <div class="scene-content">
              <el-icon class="scene-icon glow-icon"><MapLocation /></el-icon>
              <div class="scene-text">
                <h3 class="neon-cyan-text">微观拥堵溯源解析 (北京典型路段)</h3>
                <p>利用四组神经元模型横向对比，精准量化 OSM 与天气特征对特定拥堵路段中交通方式构成的解析提升。</p>
                <div class="tags">
                  <span class="cyber-tag danger-tag">[ 算力限定: Exp2 - Exp4 ]</span>
                </div>
              </div>
            </div>
          </div>
        </el-col>
        <el-col :span="12">
          <div class="scene-card glass-card interactive-card" @click="$router.push('/green-travel')">
            <div class="scan-line"></div>
            <div class="scene-content">
              <el-icon class="scene-icon glow-icon"><Bicycle /></el-icon>
              <div class="scene-text">
                <h3 class="neon-blue-text">绿色出行量化与碳减排引擎</h3>
                <p>调用跨城市通用基线，实现个人低碳出行行为的无感识别、碳减排量极速核算及 AI 环保报告生成。</p>
                <div class="tags">
                  <span class="cyber-tag success-tag">[ 算力限定: Exp1 通用基线 ]</span>
                </div>
              </div>
            </div>
          </div>
        </el-col>
      </el-row>
    </div>

  </div>
</template>

<script setup lang="ts">
import { MapLocation, Bicycle, DataBoard } from '@element-plus/icons-vue'
</script>

<style scoped>
/* 核心变量池 */
.home-container {
  --neon-cyan: #00f0ff;
  --tech-blue: #4A90E2;
  --deep-bg: #030816;
  --glass-bg: rgba(3, 8, 22, 0.55);
  --glass-border: rgba(0, 240, 255, 0.2);
  --text-main: #e5eaf3;
  --text-muted: #8892b0;
  
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 28px;
  overflow-y: auto;
  padding-right: 8px;
  padding-bottom: 20px;
}

/* 滚动条赛博化 */
.home-container::-webkit-scrollbar { width: 6px; }
.home-container::-webkit-scrollbar-thumb { 
  background: var(--tech-blue); 
  border-radius: 3px; 
  box-shadow: 0 0 10px var(--tech-blue);
}

/* 全局玻璃拟态基类 */
.glass-card {
  background: var(--glass-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 240, 255, 0.05);
  position: relative;
  overflow: hidden;
}

/* ================= Hero 区域 ================= */
.hero-card {
  height: 240px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 40px;
  border-left: 4px solid var(--neon-cyan);
}

/* 装饰角 */
.tech-corner {
  position: absolute;
  width: 20px;
  height: 20px;
  border: 2px solid var(--neon-cyan);
  opacity: 0.5;
}
.tech-corner.top-left { top: 10px; left: 10px; border-right: none; border-bottom: none; }
.tech-corner.bottom-right { bottom: 10px; right: 10px; border-left: none; border-top: none; }

.neon-title {
  font-size: 36px;
  font-weight: 800;
  color: #fff;
  margin: 0 0 8px 0;
  letter-spacing: 2px;
  text-shadow: 0 0 15px rgba(0, 240, 255, 0.6);
}

.glitch-text {
  color: var(--neon-cyan);
}

.cyber-text {
  color: var(--tech-blue);
  font-family: 'Fira Code', monospace;
  font-size: 13px;
  letter-spacing: 3px;
  margin: 0 0 16px 0;
}

.tech-line-horizontal {
  height: 1px;
  width: 60%;
  background: linear-gradient(90deg, var(--neon-cyan), transparent);
  margin-bottom: 20px;
  box-shadow: 0 0 10px var(--neon-cyan);
}

.description {
  color: var(--text-muted);
  font-size: 14px;
  line-height: 1.8;
  margin: 0 0 24px 0;
  max-width: 95%;
}
.highlight {
  color: var(--text-main);
  text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
}

.neon-button {
  background: transparent !important;
  border: 1px solid var(--neon-cyan) !important;
  color: var(--neon-cyan) !important;
  font-weight: bold;
  letter-spacing: 1px;
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.2), inset 0 0 10px rgba(0, 240, 255, 0.1);
  transition: all 0.3s;
}
.neon-button:hover {
  background: var(--neon-cyan) !important;
  color: var(--deep-bg) !important;
  box-shadow: 0 0 25px var(--neon-cyan);
}

/* ================= 指标矩阵 ================= */
.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 16px;
  height: 240px;
}
.metric-card {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  transition: all 0.3s;
}
.metric-card.hover-glow:hover {
  transform: translateY(-2px);
  border-color: var(--neon-cyan);
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(0, 240, 255, 0.15);
}
.metric-title { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; letter-spacing: 1px;}
.metric-value { 
  font-size: 32px; 
  font-weight: bold; 
  font-family: 'Fira Code', 'Courier New', monospace; 
}
.neon-blue { color: var(--tech-blue); text-shadow: 0 0 15px rgba(74, 144, 226, 0.8); }
.neon-cyan { color: var(--neon-cyan); text-shadow: 0 0 15px rgba(0, 240, 255, 0.8); }
.neon-white { color: #fff; text-shadow: 0 0 10px rgba(255, 255, 255, 0.5); }
.unit { font-size: 14px; font-weight: normal; color: var(--text-muted); text-shadow: none; }

/* ================= 通用 Section 标题 ================= */
.cyber-section-title {
  font-size: 18px;
  font-weight: 600;
  color: #fff;
  padding-left: 12px;
  border-left: 4px solid var(--neon-cyan);
  display: flex;
  align-items: center;
  gap: 10px;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
}
.section-wrapper { display: flex; flex-direction: column; gap: 16px; }

/* ================= 时间线 (步骤条) ================= */
.timeline-card { padding: 40px 20px; }

/* 深度覆写 Element Plus 步骤条样式 */
:deep(.el-step__title) { 
  color: var(--text-main) !important; 
  font-weight: bold; 
  font-size: 15px; 
}
:deep(.el-step__description) { 
  color: var(--text-muted) !important; 
  font-size: 13px; 
  margin-top: 10px; 
  line-height: 1.6; 
}
:deep(.el-step__head.is-finish .el-step__icon) {
  background: var(--glass-bg);
  border-color: var(--neon-cyan);
  color: var(--neon-cyan);
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.5);
}
:deep(.el-step__title.is-finish) {
  color: var(--neon-cyan) !important;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
}
:deep(.el-step__line) { 
  background-color: rgba(74, 144, 226, 0.2); 
}
:deep(.el-step__line-inner) {
  border-width: 2px !important;
}

/* ================= 场景卡片 ================= */
.scene-card { 
  cursor: pointer; 
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); 
  height: 160px; 
}
.scene-card:hover { 
  transform: translateY(-5px); 
  border-color: var(--neon-cyan); 
  box-shadow: 0 15px 30px rgba(0,0,0,0.6), inset 0 0 30px rgba(0, 240, 255, 0.1);
}
/* 雷达扫描线特效 */
.scan-line {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background: var(--neon-cyan);
  box-shadow: 0 0 10px var(--neon-cyan);
  opacity: 0;
  transition: opacity 0.3s;
}
.scene-card:hover .scan-line {
  opacity: 0.8;
  animation: scan 2s infinite linear;
}

.scene-content { display: flex; align-items: flex-start; gap: 24px; height: 100%; padding: 20px;}
.scene-icon { font-size: 50px; color: var(--tech-blue); transition: all 0.3s;}
.scene-card:hover .scene-icon {
  color: var(--neon-cyan);
  filter: drop-shadow(0 0 15px var(--neon-cyan));
  transform: scale(1.1);
}

.scene-text { flex: 1; display: flex; flex-direction: column; height: 100%;}
.scene-text h3 { margin: 0 0 12px 0; font-size: 17px; letter-spacing: 1px;}
.neon-cyan-text { color: var(--neon-cyan); }
.neon-blue-text { color: var(--tech-blue); }

.scene-text p { margin: 0 0 16px 0; font-size: 13px; color: var(--text-muted); line-height: 1.6; }

/* 赛博朋克风标签 */
.cyber-tag {
  font-family: 'Fira Code', monospace;
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 4px;
  background: rgba(0, 0, 0, 0.4);
}
.danger-tag {
  color: #ff4d4f;
  border: 1px solid rgba(255, 77, 79, 0.4);
  box-shadow: 0 0 8px rgba(255, 77, 79, 0.2);
}
.success-tag {
  color: #52c41a;
  border: 1px solid rgba(82, 196, 26, 0.4);
  box-shadow: 0 0 8px rgba(82, 196, 26, 0.2);
}

/* 动画定义 */
@keyframes scan {
  0% { top: 0; opacity: 0; }
  10% { opacity: 0.8; }
  90% { opacity: 0.8; }
  100% { top: 100%; opacity: 0; }
}
</style>