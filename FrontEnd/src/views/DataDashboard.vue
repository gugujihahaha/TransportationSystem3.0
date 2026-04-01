<template>
  <div class="scrollable-container">
    <div class="dashboard-container">
      <el-row :gutter="20" class="content-row">
        
        <el-col :span="6" class="full-height">
          <div class="panel dark-panel feature-panel">
            <div class="panel-header">
              <el-icon><DataLine /></el-icon> 多模态数据与 49维特征体系
            </div>
            <div class="panel-content">
              <div class="data-source-cards">
                <div class="source-card">
                  <span class="label">轨迹点量级</span>
                  <span class="value">百万级</span>
                </div>
                <div class="source-card">
                  <span class="label">覆盖路网</span>
                  <span class="value">北京 OSM</span>
                </div>
                <div class="source-card">
                  <span class="label">气象跨度</span>
                  <span class="value">5年历史</span>
                </div>
              </div>
              
              <div class="tree-container">
                <el-tree
                  :data="featureTreeData"
                  :props="defaultProps"
                  default-expand-all
                  class="custom-tree"
                >
                  <template #default="{ node, data }">
                    <span class="custom-tree-node">
                      <el-icon v-if="data.icon" :class="data.color"><component :is="data.icon" /></el-icon>
                      <span>{{ node.label }}</span>
                      <el-tag v-if="data.tag" size="small" :type="data.tagType || 'info'" effect="plain" class="tree-tag">{{ data.tag }}</el-tag>
                    </span>
                  </template>
                </el-tree>
              </div>
            </div>
          </div>
        </el-col>

        <el-col :span="12" class="full-height">
          <div class="panel dark-panel chart-panel">
            <div class="panel-header">
              <el-icon><TrendCharts /></el-icon> 多模态网络训练态势 (Training Metrics)
            </div>
            <div class="panel-content chart-container">
              <div class="mock-chart">
                <div class="chart-line loss-line"></div>
                <div class="chart-line acc-line"></div>
                <div class="chart-grid"></div>
                <span class="mock-text">模型收敛曲线 (Loss / Accuracy) 实时渲染区</span>
              </div>
            </div>
            <div class="metrics-footer">
              <div class="footer-item"><span class="dot primary"></span> Validation Loss: 0.241</div>
              <div class="footer-item"><span class="dot success"></span> F1-Score: 0.825</div>
              <div class="footer-item"><span class="dot warning"></span> Epochs: 150/150</div>
            </div>
          </div>
        </el-col>

        <el-col :span="6" class="full-height">
          <div class="panel dark-panel text-panel">
            <div class="panel-header">
              <el-icon><Opportunity /></el-icon> 模型评估与核心洞察
            </div>
            <div class="panel-content insight-content">
              
              <div class="insight-item">
                <div class="insight-badge">Highlight 1</div>
                <h3 class="insight-title">基于 Focal Loss 攻克类别不平衡</h3>
                <div class="insight-desc">
                  <ul>
                    <li>原始数据中“步行”与“公交”样本极度不平衡（比例约 8:1）。</li>
                    <li>引入 Focal Loss 动态调整权重后，模型对“公交/地铁”等弱势类别的召回率显著提升了 <strong>14.2%</strong>。</li>
                  </ul>
                </div>
              </div>

              <div class="insight-item">
                <div class="insight-badge">Highlight 2</div>
                <h3 class="insight-title">空间拓扑 (OSM) 的降维打击</h3>
                <div class="insight-desc">
                  <ul>
                    <li>仅靠 GPS 速度，汽车在堵车时极易与自行车混淆。</li>
                    <li>注入 OSM 路网特征（当前所在路段的限速、道路类型、距地铁站距离）后，机动车与非机动车的误判率断崖式下降。</li>
                  </ul>
                </div>
              </div>

              <div class="insight-metrics">
                <div class="metric-item">
                  <span class="label">基线准确率</span>
                  <span class="value" style="color: #909399;">68.5%</span>
                </div>
                <div class="metric-item">
                  <span class="label">多模态融合准确率</span>
                  <span class="value" style="color: #67C23A;">83.2% <el-icon><Top /></el-icon></span>
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
import { DataLine, TrendCharts, Opportunity, Top, Location, MapLocation, PartlyCloudy, Cpu } from '@element-plus/icons-vue'

const defaultProps = { children: 'children', label: 'label' }

// 左侧 49 维特征树形结构数据
const featureTreeData = [
  {
    label: '时序运动学特征 (GPS)', icon: 'Location', color: 'text-primary',
    children: [
      { label: '速度分布 (Mean, Max, Var)', tag: '核心', tagType: 'danger' },
      { label: '加速度与加加速度 (Jerk)' },
      { label: '航向角变化率 (Heading Change)' },
      { label: '停顿频率与时长 (Stop Rate)' }
    ]
  },
  {
    label: '空间语义特征 (OSM)', icon: 'MapLocation', color: 'text-warning',
    children: [
      { label: '轨迹点所在道路类型 (Highway Type)', tag: '核心', tagType: 'danger' },
      { label: '路段限速 (Max Speed)' },
      { label: '距最近公交站的距离 (Dist to Bus)' },
      { label: '距最近地铁站的距离 (Dist to Subway)' }
    ]
  },
  {
    label: '环境气象特征 (Weather)', icon: 'PartlyCloudy', color: 'text-success',
    children: [
      { label: '温度与体感温度 (Temp/Feels Like)' },
      { label: '降水量与湿度 (Precip/Humidity)', tag: '辅助', tagType: 'info' },
      { label: '风速与能见度 (Wind/Visibility)' }
    ]
  }
]
</script>

<style scoped>
/* ========== 全局滚动容器样式 ========== */
.scrollable-container {
  width: 100%;
  height: 100%;
  padding: 24px 32px;
  overflow-y: auto;
  overflow-x: hidden;
  box-sizing: border-box;
}

.scrollable-container::-webkit-scrollbar { width: 8px; }
.scrollable-container::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
.scrollable-container::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.3); border-radius: 4px; }
.scrollable-container::-webkit-scrollbar-thumb:hover { background: rgba(0, 240, 255, 0.6); }

/* 使原有内部容器撑满但去除内边距，交由外层容器控制 */
.dashboard-container { height: 100%; }
.content-row { min-height: 100%; align-items: stretch; }
.full-height { display: flex; flex-direction: column; }

/* ========== 原有样式完全保留 ========== */
.panel { background: rgba(16, 25, 43, 0.6); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); backdrop-filter: blur(10px); flex: 1; display: flex; flex-direction: column;}
.dark-panel { color: #e5eaf3; }
.panel-header { display: flex; align-items: center; gap: 8px; font-size: 18px; font-weight: bold; color: #fff; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
.panel-content { flex: 1; overflow: hidden;}

/* 左侧：特征体系 */
.data-source-cards { display: flex; justify-content: space-between; margin-bottom: 24px;}
.source-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); padding: 12px 16px; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; width: 30%;}
.source-card .label { font-size: 12px; color: #a0aabf; margin-bottom: 4px;}
.source-card .value { font-size: 16px; font-weight: bold; color: #00f0ff;}
.tree-container { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 16px; height: calc(100% - 90px); overflow-y: auto;}

/* Element Plus Tree 样式覆盖 (深色主题) */
:deep(.custom-tree.el-tree) { background: transparent; color: #c0c4cc;}
:deep(.el-tree-node__content:hover) { background-color: rgba(255, 255, 255, 0.05);}
:deep(.el-tree-node:focus > .el-tree-node__content) { background-color: transparent;}
.custom-tree-node { flex: 1; display: flex; align-items: center; font-size: 14px; padding-right: 8px;}
.custom-tree-node span { margin-left: 8px; }
.tree-tag { margin-left: auto; }
.text-primary { color: #409EFF; }
.text-warning { color: #E6A23C; }
.text-success { color: #67C23A; }

/* 中间：图表占位 */
.chart-panel { display: flex; flex-direction: column; }
.chart-container { display: flex; align-items: center; justify-content: center; position: relative;}
.mock-chart { width: 100%; height: 100%; border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; background: linear-gradient(180deg, rgba(0,240,255,0.02) 0%, transparent 100%);}
.mock-text { color: #5a6270; font-size: 16px; font-weight: bold; letter-spacing: 2px; z-index: 2;}
.chart-grid { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background-size: 40px 40px; background-image: linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px), linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px);}
.metrics-footer { display: flex; justify-content: center; gap: 30px; margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.05);}
.footer-item { display: flex; align-items: center; gap: 8px; font-size: 14px; font-family: monospace; color: #a0aabf;}
.dot { width: 8px; height: 8px; border-radius: 50%; }
.dot.primary { background-color: #409EFF; box-shadow: 0 0 8px #409EFF; }
.dot.success { background-color: #67C23A; box-shadow: 0 0 8px #67C23A; }
.dot.warning { background-color: #E6A23C; box-shadow: 0 0 8px #E6A23C; }

/* 右侧：洞察文本 */
.insight-content { overflow-y: auto;}
.insight-item { margin-bottom: 24px;}
.insight-item:last-child { margin-bottom: 0;}
.insight-title { margin: 12px 0 8px 0; font-size: 16px; color: #e5eaf3; line-height: 1.4; }
.insight-badge { display: inline-block; padding: 4px 12px; background: rgba(74, 144, 226, 0.15); color: #4A90E2; border-radius: 4px; font-size: 13px; font-weight: bold; width: fit-content; border: 1px solid rgba(74, 144, 226, 0.3);}
.insight-desc ul { margin: 0; padding-left: 20px; color: #a3a6ad; font-size: 13px; line-height: 1.8; }
.insight-desc li { margin-bottom: 6px; }
.insight-desc strong { color: #E6A23C; }
.insight-metrics { display: flex; justify-content: space-between; gap: 20px; background: rgba(0,0,0,0.2); padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.03); margin-top: auto;}
.metric-item { display: flex; flex-direction: column; gap: 4px; }
.metric-item .label { font-size: 12px; color: #909399; }
.metric-item .value { font-size: 20px; font-weight: bold; font-family: monospace; display: flex; align-items: center; gap: 4px;}
</style>