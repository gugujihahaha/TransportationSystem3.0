<template>
  <div class="green-container">
    <el-row :gutter="20" class="full-height">
      
      <el-col :span="14" class="full-height">
        <div class="panel dark-panel map-panel">
          <div class="panel-header">
            <div class="header-left">
              <el-icon><Compass /></el-icon> 跨城轨迹泛化识别与碳足迹渲染
            </div>
            <div class="header-actions">
              <el-tag size="small" type="success" effect="dark">驱动引擎: Exp4 焦点损失优化模型</el-tag>
            </div>
          </div>
          <div class="panel-content map-wrapper" v-loading="isAnalyzing" element-loading-text="PyTorch 模型跨城特征提取与推理中..." element-loading-background="rgba(11, 13, 18, 0.8)">
            <div ref="mapChartRef" class="echarts-map"></div>
            
            <div class="map-legend">
              <div class="legend-item"><span class="line green"></span> 绿色出行 (步行/骑行/公交/地铁)</div>
              <div class="legend-item"><span class="line grey"></span> 高碳出行 (私家车/网约车)</div>
            </div>
          </div>
        </div>
      </el-col>

      <el-col :span="10" class="full-height">
        <div class="panel dark-panel control-panel">
          <div class="panel-header">
            <el-icon><Bicycle /></el-icon> 个人低碳出行看板
          </div>
          <div class="panel-content flex-col">
            
            <div class="section-box">
              <div class="box-title">1. 上传历史轨迹进行泛化验证</div>
              <el-upload
                class="trajectory-upload"
                drag
                action="#"
                :auto-upload="false"
                :show-file-list="false"
                :on-change="handleFileUpload"
                :disabled="isAnalyzing"
              >
                <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                <div class="el-upload__text">拖拽轨迹文件到此处，或 <em>点击上传</em></div>
                <template #tip>
                  <div class="el-upload__tip text-center">支持 .csv, .plt 格式 (系统将自动调用后端预测)</div>
                </template>
              </el-upload>
            </div>

            <div class="section-box" :class="{ 'blur-mask': !hasData }">
              <div class="box-title">2. 减排量化核算 (基于真实识别结果)</div>
              <div class="carbon-dashboard">
                <div class="carbon-card">
                  <div class="icon-wrap text-green"><el-icon><Guide /></el-icon></div>
                  <div class="data-wrap">
                    <div class="val text-green">{{ stats.greenDistance }} <span class="unit">km</span></div>
                    <div class="label">识别绿色里程</div>
                  </div>
                </div>
                <div class="carbon-card">
                  <div class="icon-wrap text-blue"><el-icon><WindPower /></el-icon></div>
                  <div class="data-wrap">
                    <div class="val text-blue">{{ stats.co2Saved }} <span class="unit">kg</span></div>
                    <div class="label">累计减少 CO₂</div>
                  </div>
                </div>
                <div class="carbon-card tree-card">
                  <div class="icon-wrap text-green"><el-icon><Sunny /></el-icon></div>
                  <div class="data-wrap">
                    <div class="val text-green">{{ stats.treesPlanted }} <span class="unit">棵</span></div>
                    <div class="label">相当于种树</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="section-box report-box flex-1" :class="{ 'blur-mask': !hasData }">
              <div class="box-title ai-title">
                <el-icon><Tickets /></el-icon> AI 绿色出行摘要
              </div>
              <div class="ai-content">
                <div v-if="!aiReport" class="empty-text">等待模型分析完成...</div>
                <div v-else class="typing-text" v-html="aiReport"></div>
              </div>
            </div>

          </div>
        </div>
      </el-col>

    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, nextTick } from 'vue'
import { Compass, Bicycle, UploadFilled, Guide, WindPower, Sunny, Tickets } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { trajectoryApi } from '../api/trajectory'

// --- 状态与变量 ---
const hasData = ref(false)
const isAnalyzing = ref(false)
const aiReport = ref('')
const mapChartRef = ref<HTMLElement | null>(null)
let mapChart: echarts.ECharts | null = null

// 统计数据响应式对象
const stats = reactive({
  greenDistance: '0.00',
  co2Saved: '0.00',
  treesPlanted: '0.0'
})

// --- 核心：完全基于真实数据的解析逻辑 ---
// --- 核心：完全基于真实数据的解析逻辑 ---
const handleFileUpload = async (uploadFile: any) => {
  const file = uploadFile.raw;
  if (!file) return;

  isAnalyzing.value = true;
  hasData.value = false;
  aiReport.value = '';
  
  try {
    // 调用接口，使用 exp1 模型
    const response = await trajectoryApi.predict(file, 'exp1'); 
    const result = response as any; 
    
    console.log("========== [碳普惠] 后端真实返回 ==========", result);

    // 1. 获取真实的预测模式和真实物理距离
    const mode = result.predicted_mode || 'unknown';
    const distanceMeters = result.stats?.distance || 0;
    const distanceKm = (distanceMeters / 1000).toFixed(2); 

    // 👇 【关键点】：这里定义了 isGreen，判断是不是绿色出行
    const lowerMode = mode.toLowerCase();
    const isGreen = ['walk', 'bike', 'bus', 'subway', '步行', '自行车', '公交车', '地铁'].includes(lowerMode);

    // 3. 真实核算碳减排
    const calcGreenDist = isGreen ? distanceKm : '0.00';
    const calcCo2 = (Number(calcGreenDist) * 0.15).toFixed(2); 
    const calcTrees = (Number(calcCo2) * 0.05).toFixed(1);     

    stats.greenDistance = calcGreenDist;
    stats.co2Saved = calcCo2;
    stats.treesPlanted = calcTrees;

    hasData.value = true;
    
    // 4. 将真实的 GPS 点数组交由 ECharts 渲染，同时传入上面定义的 isGreen
    if (result.points && result.points.length > 0) {
      renderRealTrajectory(result.points, isGreen);
    } else {
      ElMessage.warning('后端推理成功，但未返回可渲染的经纬度 points 数组');
    }

    // 5. 根据真实数据拼接战报
    generateAIReport(calcGreenDist, calcCo2, calcTrees, translateMode(mode));

    ElMessage.success(`模型解析完毕！真实判定为：${translateMode(mode)}`);

  } catch (error) {
    ElMessage.error('模型推理失败，请检查轨迹文件格式或后端服务状态');
    console.error('Prediction Error:', error);
  } finally {
    isAnalyzing.value = false;
  }
}

// 辅助：中英文出行方式映射
const translateMode = (mode: string) => {
  const map: Record<string, string> = { 'car': '私家车', 'bus': '公交车', 'walk': '步行', 'bike': '自行车', 'subway': '地铁', 'train': '火车' };
  return map[mode.toLowerCase()] || mode;
}

// 动态生成绝对真实的报告文案
const generateAIReport = (dist: string, co2: string, trees: string, modeName: string) => {
  aiReport.value = ''
  let reportText = `<b>[解析完成]</b><br/>系统已成功使用 Exp1 纯轨迹泛化模型对上传的 GPS 序列进行特征提取。模型判定本次真实出行为：<b>${modeName}</b>。<br/><br/>`
  
  if (Number(dist) > 0) {
    reportText += `这是一次完美的低碳出行！您的绿色出行里程达到 <b>${dist}km</b>。<br/><br/>经核算，相比全程驾驶高碳排交通工具，您本次出行累计减少了 <b>${co2}kg</b> 碳排放，相当于种下了 <b>${trees}棵树</b>。感谢您为城市低碳生态做出的贡献！`
  } else {
    reportText += `经识别，本次出行为高碳排交通方式。如果将这 <b>${dist === '0.00' ? '段' : dist}</b> 行程替换为公共交通或慢行系统，您可以大幅减少城市碳足迹，期待您下次选择绿色出行！`
  }
  simulateTyping(reportText)
}

// --- ECharts 地图渲染 ---
const initMapChart = () => {
  if (!mapChartRef.value) return
  mapChart = echarts.init(mapChartRef.value)
  mapChart.setOption({
    backgroundColor: 'transparent',
    // 必须设置 scale: true，否则无法适应真实的经纬度极值坐标
    xAxis: { type: 'value', scale: true, show: false },
    yAxis: { type: 'value', scale: true, show: false },
    series: [] 
  })
}

// 核心：基于真实的 3000+ 个坐标点绘制轨迹
// 核心：基于真实的 3000+ 个坐标点绘制轨迹
const renderRealTrajectory = (points: any[], isGreen: boolean) => {
  if (!mapChart) return

  // 灵活解析各种可能格式的后端经纬度 (增加了对 longitude/latitude 全拼的支持)
  const echartsData = points.map(p => {
    if (Array.isArray(p)) return [p[0], p[1]]; 
    return [
      p.lng || p.lon || p.longitude || p.x || p[0], 
      p.lat || p.latitude || p.y || p[1]
    ];
  }).filter(p => p[0] !== undefined && p[1] !== undefined);

  if (echartsData.length === 0) {
    ElMessage.warning('警告：轨迹点解析失败，请按 F12 检查后端经纬度字段名！');
    return;
  }

  const lineColor = isGreen ? '#67C23A' : '#909399';
  const lineName = isGreen ? '绿色出行' : '高碳出行';

  mapChart.setOption({
    // 【关键修复】：必须在这里重新声明 xAxis 和 yAxis，否则会被末尾的 true 彻底清空！
    xAxis: { type: 'value', scale: true, show: false },
    yAxis: { type: 'value', scale: true, show: false },
    series: [
      {
        type: 'line', 
        data: echartsData,
        lineStyle: { color: lineColor, width: 4 }, 
        smooth: false, // 关闭平滑，展示真实的 GPS 噪点折线特征
        symbol: 'none', 
        name: lineName
      }
    ]
  }, true); // true 表示清空上一次画的线，只保留当前配置
}

// 打字机动效
let typingTimer: any = null
const simulateTyping = (text: string) => {
  if (typingTimer) clearInterval(typingTimer)
  aiReport.value = ''
  aiReport.value = text 
}

const handleResize = () => mapChart?.resize()

onMounted(async () => {
  await nextTick()
  initMapChart()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  mapChart?.dispose()
})
</script>

<style scoped>
/* 样式部分保持一致 */
.green-container { height: 100%; padding-bottom: 20px;}
.full-height { height: 100%; }
.panel { border-radius: 12px; height: 100%; display: flex; flex-direction: column; overflow: hidden; }
.dark-panel { background: rgba(26, 31, 46, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); }

.panel-header { padding: 16px 20px; font-size: 16px; font-weight: 600; color: #e5eaf3; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: space-between;}
.header-left { display: flex; align-items: center; gap: 8px;}
.panel-content { padding: 20px; flex: 1; overflow-y: auto; position: relative;}
.flex-col { display: flex; flex-direction: column; gap: 20px; }
.flex-1 { flex: 1; }

.map-wrapper { padding: 0; background: #0b0d12; }
.echarts-map { width: 100%; height: 100%; }
.map-legend { position: absolute; bottom: 20px; right: 20px; background: rgba(0,0,0,0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(4px); z-index: 10;}
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #e5eaf3; margin-bottom: 8px;}
.legend-item:last-child { margin-bottom: 0;}
.line { width: 20px; height: 4px; border-radius: 2px; }
.line.green { background: #67C23A; box-shadow: 0 0 5px #67C23A;}
.line.grey { background: #909399; }

.section-box { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 16px; border: 1px solid rgba(255,255,255,0.03); transition: all 0.3s;}
.box-title { font-size: 14px; color: #a3a6ad; margin-bottom: 16px; font-weight: 500; display: flex; align-items: center;}
.blur-mask { opacity: 0.3; pointer-events: none; filter: blur(2px); }

:deep(.el-upload-dragger) { background: rgba(255,255,255,0.02); border-color: rgba(255,255,255,0.1); transition: all 0.3s; padding: 20px;}
:deep(.el-upload-dragger:hover) { border-color: #67C23A; background: rgba(103, 194, 58, 0.05);}
:deep(.el-upload.is-disabled .el-upload-dragger) { cursor: not-allowed; border-color: #333; background: #111;}
.text-center { text-align: center; margin-top: 10px; }

.carbon-dashboard { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.carbon-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 16px 12px; display: flex; flex-direction: column; align-items: center; gap: 10px; text-align: center;}
.icon-wrap { font-size: 24px; }
.data-wrap .val { font-size: 24px; font-weight: bold; font-family: monospace; }
.data-wrap .unit { font-size: 12px; margin-left: 2px;}
.data-wrap .label { font-size: 12px; color: #909399; margin-top: 4px;}
.text-green { color: #67C23A; }
.text-blue { color: #4A90E2; }
.tree-card { background: rgba(103, 194, 58, 0.05); border-color: rgba(103, 194, 58, 0.2);}

.report-box { border-color: rgba(74, 144, 226, 0.3); background: rgba(74, 144, 226, 0.05); display: flex; flex-direction: column;}
.ai-title { color: #4A90E2; }
.ai-content { flex: 1; font-size: 14px; line-height: 1.8; color: #e5eaf3; background: rgba(0,0,0,0.3); padding: 16px; border-radius: 6px; border-left: 3px solid #4A90E2;}
.empty-text { color: #909399; font-style: italic; text-align: center; margin-top: 20px;}
:deep(b) { color: #67C23A; }
</style>