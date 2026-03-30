<template>
  <div class="home-container">
    <el-row :gutter="24" class="top-row">
      <el-col :span="14">
        <div class="hero-card">
          <h1 class="gradient-text">TrafficRec 多模态系统</h1>
          <p class="subtitle">数据研究驾驶舱 + 应用验证平台 + 完整系统门户</p>
          <p class="description">
            本项目基于深度学习，融合 <strong>轨迹动力学、OSM 城市空间语义及气象环境数据</strong>，构建了包含 49 维特征的多模态框架。旨在解决复杂路况及长尾类别的交通方式高精度识别问题，并赋能拥堵溯源与碳普惠评估。
          </p>
          <div class="action-buttons">
            <el-button type="primary" size="large" @click="$router.push('/dashboard')">
              <el-icon class="el-icon--left"><DataBoard /></el-icon> 进入核心数据驾驶舱
            </el-button>
          </div>
        </div>
      </el-col>
      <el-col :span="10">
        <div class="metrics-grid">
          <el-card class="metric-card dark-card" shadow="hover">
            <div class="metric-title">系统最佳 Accuracy</div>
            <div class="metric-value text-blue">92.4%</div>
          </el-card>
          <el-card class="metric-card dark-card" shadow="hover">
            <div class="metric-title">系统最佳 Macro-F1</div>
            <div class="metric-value text-orange">89.1%</div>
          </el-card>
          <el-card class="metric-card dark-card" shadow="hover">
            <div class="metric-title">支持识别交通方式</div>
            <div class="metric-value">8 类</div>
          </el-card>
          <el-card class="metric-card dark-card" shadow="hover">
            <div class="metric-title">实机验证落地场景</div>
            <div class="metric-value">2 组</div>
          </el-card>
        </div>
      </el-col>
    </el-row>

    <div class="section-wrapper">
      <div class="section-title">研究流程与实验演进主轴</div>
      <el-card class="timeline-card dark-card" shadow="never">
        <el-steps :active="4" align-center class="custom-steps">
          <el-step title="Exp1：基础建模" description="纯轨迹时空特征，剥离地域依赖，验证模型全国泛化能力" />
          <el-step title="Exp2：路网融合" description="引入 OSM 城市数据，打破轨迹孤岛，解决复杂路段语义缺失" />
          <el-step title="Exp3：气象耦合" description="加入天气维度，增强模型对雨雪等极端环境扰动下的解释力" />
          <el-step title="Exp4：算法突破" description="引入高阶损失函数，攻克长尾类别（如火车/地铁）识别难题" />
        </el-steps>
      </el-card>
    </div>

    <div class="section-wrapper">
      <div class="section-title">真实应用场景验证</div>
      <el-row :gutter="24" class="bottom-row">
        <el-col :span="12">
          <el-card class="scene-card dark-card" shadow="hover" @click="$router.push('/congestion-analysis')">
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
          <el-card class="scene-card dark-card" shadow="hover" @click="$router.push('/green-travel')">
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
</template>

<script setup lang="ts">
import { MapLocation, Bicycle, DataBoard } from '@element-plus/icons-vue'
</script>

<style scoped>
.home-container { height: 100%; display: flex; flex-direction: column; gap: 24px; overflow-y: auto; padding-right: 8px; padding-bottom: 20px;}
.home-container::-webkit-scrollbar { width: 6px; }
.home-container::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

.dark-card { background: rgba(26, 31, 46, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px;}

.hero-card { height: 220px; display: flex; flex-direction: column; justify-content: center; padding: 0 40px; background: linear-gradient(135deg, rgba(26, 31, 46, 0.8) 0%, rgba(15, 17, 23, 0) 100%); border-left: 4px solid #4A90E2; border-radius: 12px; }
.gradient-text { 
  font-size: 34px; 
  font-weight: 700; 
  background: linear-gradient(90deg, #4A90E2, #67C23A); 
  -webkit-background-clip: text; 
  background-clip: text; /* 新增这一行标准属性，消除警告 */
  color: transparent; 
  margin: 0 0 12px 0; 
  letter-spacing: 1px;
}
.subtitle { color: #e5eaf3; font-size: 16px; margin: 0 0 16px 0; font-weight: 500;}
.description { color: #909399; font-size: 13px; line-height: 1.6; margin: 0 0 24px 0; max-width: 90%;}
.description strong { color: #a3a6ad; }

.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 16px; height: 220px; }
.metric-card { display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
.metric-title { font-size: 13px; color: #909399; margin-bottom: 8px; }
.metric-value { font-size: 28px; font-weight: bold; color: #e5eaf3; font-family: 'Courier New', Courier, monospace; }
.text-blue { color: #4A90E2; }
.text-orange { color: #FA8C16; }

.section-wrapper { display: flex; flex-direction: column; gap: 12px; }
.section-title { font-size: 16px; font-weight: 600; color: #e5eaf3; padding-left: 12px; border-left: 3px solid #4A90E2; display: flex; align-items: center;}
.timeline-card { padding: 36px 20px; }

:deep(.el-step__title) { color: #e5eaf3 !important; font-weight: 600; font-size: 15px; }
:deep(.el-step__description) { color: #909399 !important; font-size: 13px; margin-top: 8px; line-height: 1.6; padding: 0 10px;}
:deep(.el-step__head.is-finish) { color: #4A90E2; border-color: #4A90E2; }
:deep(.el-step__line) { background-color: rgba(255,255,255,0.1); }

.scene-card { cursor: pointer; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); height: 150px; }
.scene-card:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(0,0,0,0.4); border-color: rgba(74, 144, 226, 0.5); background: rgba(30, 36, 50, 0.6);}
.scene-content { display: flex; align-items: flex-start; gap: 20px; height: 100%; padding: 10px;}
.scene-icon { font-size: 54px; color: #4A90E2; opacity: 0.9; padding-top: 5px;}
.scene-text { flex: 1; display: flex; flex-direction: column; height: 100%;}
.scene-text h3 { margin: 0 0 12px 0; font-size: 17px; color: #e5eaf3; }
.scene-text p { margin: 0 0 16px 0; font-size: 13px; color: #909399; line-height: 1.5; }
.tags { margin-top: auto; }
</style>