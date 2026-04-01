<template>
  <div class="home-container">
    <div class="tech-grid-bg"></div>

    <div class="content-wrapper relative-z">
      
      <el-row :gutter="24" class="top-row">
        <el-col :span="14">
          <div class="hero-card float-anim">
            <div class="glow-point"></div> <h1 class="gradient-text">TrafficRec 多模态系统</h1>
            <p class="subtitle">数据研究驾驶舱 + 应用验证平台 + 完整系统门户</p>
            <p class="description">
              本项目基于深度学习，融合 <strong>轨迹动力学、OSM 城市空间语义及气象环境数据</strong>，构建了包含 49 维特征的多模态框架。旨在解决复杂路况及长尾类别的交通方式高精度识别问题，并赋能拥堵溯源与碳普惠评估。
            </p>
            <div class="action-buttons">
              <el-button type="primary" size="large" class="cyber-btn" @click="$router.push('/dashboard')">
                <el-icon class="el-icon--left"><DataBoard /></el-icon> 进入核心数据驾驶舱
              </el-button>
            </div>
          </div>
        </el-col>
        
        <el-col :span="10">
          <div class="metrics-grid">
            <el-card class="metric-card dark-card cyber-border" shadow="hover">
              <div class="metric-title">系统最佳 Accuracy</div>
              <div class="metric-value text-blue pulse-text">92.4%</div>
            </el-card>
            <el-card class="metric-card dark-card cyber-border" shadow="hover">
              <div class="metric-title">系统最佳 Macro-F1</div>
              <div class="metric-value text-orange pulse-text">89.1%</div>
            </el-card>
            <el-card class="metric-card dark-card cyber-border" shadow="hover">
              <div class="metric-title">支持识别交通方式</div>
              <div class="metric-value text-green">8 类</div>
            </el-card>
            <el-card class="metric-card dark-card cyber-border" shadow="hover">
              <div class="metric-title">实机验证落地场景</div>
              <div class="metric-value">2 组</div>
            </el-card>
          </div>
        </el-col>
      </el-row>

      <div class="section-wrapper">
        <div class="section-title"><span class="title-decorate"></span>研究流程与实验演进主轴</div>
        <el-card class="timeline-card dark-card glass-panel" shadow="never">
          <el-steps :active="4" align-center class="custom-steps">
            <el-step title="Exp1：基础建模" description="纯轨迹时空特征，剥离地域依赖，验证模型全国泛化能力" />
            <el-step title="Exp2：路网融合" description="引入 OSM 城市数据，打破轨迹孤岛，解决复杂路段语义缺失" />
            <el-step title="Exp3：气象耦合" description="加入天气维度，增强模型对雨雪等极端环境扰动下的解释力" />
            <el-step title="Exp4：算法突破" description="引入高阶损失函数，攻克长尾类别（如火车/地铁）识别难题" />
          </el-steps>
        </el-card>
      </div>

      <div class="section-wrapper pb-20">
        <div class="section-title"><span class="title-decorate"></span>真实应用场景验证</div>
        <el-row :gutter="24" class="bottom-row">
          <el-col :span="12">
            <el-card class="scene-card dark-card cyber-hover" shadow="hover" @click="$router.push('/congestion-analysis')">
              <div class="scene-overlay"></div>
              <div class="scene-content">
                <el-icon class="scene-icon"><MapLocation /></el-icon>
                <div class="scene-text">
                  <h3>微观拥堵溯源解析 (北京典型路段)</h3>
                  <p>利用四组模型横向对比，量化 OSM 与天气特征对特定拥堵路段中交通方式构成的解析能力提升。</p>
                  <div class="tags"><el-tag size="small" type="danger" effect="dark">限 Exp2 - Exp4</el-tag></div>
                </div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="12">
            <el-card class="scene-card dark-card cyber-hover" shadow="hover" @click="$router.push('/green-travel')">
              <div class="scene-overlay"></div>
              <div class="scene-content">
                <el-icon class="scene-icon"><Bicycle /></el-icon>
                <div class="scene-text">
                  <h3>绿色出行量化与碳减排核算闭环</h3>
                  <p>运用跨城市通用基线模型，实现个人低碳出行行为识别、碳减排量计算及个性化 AI 激励报告生成。</p>
                  <div class="tags"><el-tag size="small" type="success" effect="dark">限 Exp1 通用基线</el-tag></div>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
// 原封不动保留
import { MapLocation, Bicycle, DataBoard } from '@element-plus/icons-vue'
</script>

<style scoped>
/* ========== 原有核心容器布局 ========== */
.home-container { height: 100%; display: flex; flex-direction: column; overflow-y: auto; padding-right: 8px; padding-bottom: 20px; position: relative;}
.home-container::-webkit-scrollbar { width: 6px; }
.home-container::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.3); border-radius: 3px; }
.content-wrapper { display: flex; flex-direction: column; gap: 24px; }
.pb-20 { padding-bottom: 20px; }

/* ========== 新增：底层赛博网格装饰 ========== */
.tech-grid-bg {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none;
  background-image: 
    linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
  background-size: 30px 30px;
}
.relative-z { position: relative; z-index: 1; }

/* ========== 在你原有的 dark-card 基础上强化玻璃质感 ========== */
:deep(.dark-card) { 
  background: rgba(16, 25, 43, 0.6) !important; 
  border: 1px solid rgba(0, 240, 255, 0.1) !important; 
  border-radius: 12px; 
  backdrop-filter: blur(10px);
  color: #fff;
}
.glass-panel { box-shadow: inset 0 0 20px rgba(0, 240, 255, 0.02), 0 8px 32px rgba(0, 0, 0, 0.3); }

/* ========== 原有：主视觉卡片 (Hero Card) ========== */
.hero-card { 
  height: 220px; display: flex; flex-direction: column; justify-content: center; padding: 0 40px; 
  background: linear-gradient(135deg, rgba(16, 25, 43, 0.9) 0%, rgba(16, 25, 43, 0.2) 100%); 
  border-left: 4px solid #00f0ff; border-radius: 12px; position: relative; overflow: hidden;
}
.glow-point { position: absolute; top: -50px; left: -50px; width: 150px; height: 150px; background: radial-gradient(circle, rgba(0,240,255,0.2) 0%, transparent 70%); }
.float-anim { animation: floating 6s ease-in-out infinite; }
@keyframes floating { 0% { transform: translateY(0px); } 50% { transform: translateY(-4px); } 100% { transform: translateY(0px); } }

.gradient-text { 
  font-size: 36px; 
  font-weight: 800; 
  margin: 0 0 12px 0; 
  letter-spacing: 2px;
  background: linear-gradient(90deg, #fff, #00f0ff, #4A90E2); 
  -webkit-background-clip: text; 
  background-clip: text; 
  color: transparent; 
  text-shadow: 0 0 20px rgba(0, 240, 255, 0.3);
}
.subtitle { color: #00f0ff; font-size: 16px; margin: 0 0 16px 0; font-weight: 500;}
.description { color: #a0aabf; font-size: 13px; line-height: 1.6; margin: 0 0 24px 0; max-width: 90%;}
.description strong { color: #e5eaf3; font-weight: bold;}

/* 强化 el-button 发光特效 */
.cyber-btn {
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
  transition: all 0.3s;
}
.cyber-btn:hover { box-shadow: 0 0 25px rgba(0, 240, 255, 0.6); transform: scale(1.02); }

/* ========== 原有：数据指标面板 ========== */
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 16px; height: 220px; }
:deep(.metric-card.el-card .el-card__body) { padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%;}
.metric-title { font-size: 13px; color: #84a2d4; margin-bottom: 8px; }
.metric-value { font-size: 28px; font-weight: bold; font-family: 'Courier New', Courier, monospace; }
.text-blue { color: #00f0ff; }
.text-orange { color: #E6A23C; }
.text-green { color: #67C23A; }
.pulse-text { animation: textPulse 2s infinite; }
@keyframes textPulse { 0% { opacity: 0.8; } 50% { opacity: 1; text-shadow: 0 0 10px currentColor; } 100% { opacity: 0.8; } }
.cyber-border:hover { border-color: #00f0ff !important; background: rgba(0, 240, 255, 0.05) !important;}

/* ========== 原有：标题与流程步骤 ========== */
.section-wrapper { display: flex; flex-direction: column; gap: 12px; }
.section-title { font-size: 16px; font-weight: 600; color: #fff; display: flex; align-items: center; gap: 10px;}
.title-decorate { width: 4px; height: 16px; background: #00f0ff; border-radius: 2px; box-shadow: 0 0 8px #00f0ff;}
.timeline-card { padding: 20px 10px; }

:deep(.el-step__title) { color: #00f0ff !important; font-weight: 600; font-size: 15px; }
:deep(.el-step__description) { color: #a0aabf !important; font-size: 13px; margin-top: 8px; line-height: 1.6;}
:deep(.el-step__head.is-finish) { color: #00f0ff; border-color: #00f0ff; }
:deep(.el-step__head.is-process) { color: #E6A23C; border-color: #E6A23C; }
:deep(.el-step__line) { background-color: rgba(0, 240, 255, 0.2) !important; }

/* ========== 原有：应用场景卡片 ========== */
.scene-card { cursor: pointer; transition: all 0.4s; height: 150px; position: relative; overflow: hidden; padding: 0;}
:deep(.scene-card.el-card .el-card__body) { padding: 0; height: 100%; }
.scene-overlay { position: absolute; top: 0; left: -100%; width: 50%; height: 100%; background: linear-gradient(90deg, transparent, rgba(0,240,255,0.1), transparent); transform: skewX(-20deg); transition: 0.5s;}
.cyber-hover:hover { transform: translateY(-5px); border-color: #00f0ff !important; box-shadow: 0 10px 20px rgba(0, 240, 255, 0.15) !important;}
.cyber-hover:hover .scene-overlay { left: 150%; }

.scene-content { display: flex; align-items: flex-start; gap: 20px; height: 100%; padding: 15px 20px; position: relative; z-index: 1;}
.scene-icon { font-size: 50px; color: #00f0ff; opacity: 0.9; margin-top: 5px; filter: drop-shadow(0 0 5px rgba(0,240,255,0.4));}
.scene-text { flex: 1; display: flex; flex-direction: column; height: 100%;}
.scene-text h3 { margin: 0 0 10px 0; font-size: 16px; color: #fff; }
.scene-text p { margin: 0 0 12px 0; font-size: 13px; color: #909399; line-height: 1.5; }
.tags { margin-top: auto; }
</style>