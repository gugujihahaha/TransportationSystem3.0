<template>
  <div class="user-page-container">
    <div class="cyber-content-wrapper">
      
      <div class="cyber-header">
        <h2 class="title-text">
          <span class="gradient-line"></span>
          系统用户中心 / USER CONTROL
        </h2>
        <p class="desc-text">欢迎回来, {{ authStore.username }}。当前系统权限：Level 1 开发者</p>
      </div>

      <div class="cyber-grid">
        <div class="glass-panel profile-card">
          <div class="avatar-box">
            <span class="emoji-icon">👨‍🚀</span>
            <div class="glow-ring"></div>
          </div>
          <h3 class="user-name">{{ authStore.username }}</h3>
          <div class="badge">SYSTEM OPERATOR</div>
          
          <div class="stats-list">
            <div class="stat-row">
              <span class="label">认证状态</span>
              <span class="value green-glow">已加密授权</span>
            </div>
            <div class="stat-row">
              <span class="label">数据访问</span>
              <span class="value">FULL ACCESS</span>
            </div>
          </div>
          
          <button @click="handleLogout" class="exit-button">断开连接 (LOGOUT)</button>
        </div>

        <div class="glass-panel detail-panel">
          <div class="section-title">
            <span class="blue-icon">◈</span> 系统模块凭证状态
          </div>
          
          <div class="module-cards">
            <div class="m-card" v-for="item in modules" :key="item.name">
              <div class="m-info">
                <h4>{{ item.name }}</h4>
                <p>{{ item.desc }}</p>
              </div>
              <div class="m-status">ACTIVE</div>
            </div>
          </div>

          <div class="token-section">
            <label>当前会话令牌 (JWT TOKEN)</label>
            <div class="token-display">
              <code>Bearer {{ authStore.token }}</code>
            </div>
            <p class="token-tip">该令牌用于所有后端 API 验证，请勿泄露。</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const modules = [
  { name: '多模态轨迹识别核心', desc: 'CNN-LSTM-Transformer 混合推理引擎' },
  { name: 'GeoLife 异构数据集', desc: '包含 182 名用户清洗后的轨迹特征' },
  { name: 'LLM 智能研报系统', desc: '基于 SSE 的流式生成评估报告' }
]

const handleLogout = () => {
  if (confirm('是否确定退出并销毁当前令牌？')) {
    authStore.logout()
    router.push('/login')
  }
}
</script>

<style scoped>
.user-page-container {
  min-height: 80vh;
  padding: 20px;
}

.cyber-content-wrapper { max-width: 1200px; margin: 0 auto; }

.cyber-header { margin-bottom: 30px; }
.title-text { font-size: 24px; color: #fff; display: flex; align-items: center; gap: 12px; font-weight: bold; letter-spacing: 1px; }
.gradient-line { width: 4px; height: 24px; background: linear-gradient(to bottom, #0070f3, #00f0ff); border-radius: 4px; box-shadow: 0 0 10px #00f0ff; }
.desc-text { color: #64748b; margin-top: 8px; font-size: 14px; }

.cyber-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 30px; }
@media (max-width: 900px) { .cyber-grid { grid-template-columns: 1fr; } }

.glass-panel {
  background: rgba(15, 23, 42, 0.4);
  border: 1px solid rgba(0, 240, 255, 0.2);
  border-radius: 16px;
  padding: 30px;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.profile-card { text-align: center; height: fit-content; }
.avatar-box { position: relative; width: 100px; height: 100px; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center; }
.emoji-icon { font-size: 50px; z-index: 2; }
.glow-ring { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 2px solid #00f0ff; border-radius: 50%; box-shadow: 0 0 20px rgba(0, 240, 255, 0.5); }

.user-name { font-size: 24px; color: #fff; margin-bottom: 5px; }
.badge { background: rgba(0, 240, 255, 0.1); color: #00f0ff; border: 1px solid rgba(0, 240, 255, 0.5); padding: 3px 12px; border-radius: 20px; font-size: 11px; display: inline-block; margin-bottom: 25px; letter-spacing: 1px; }

.stats-list { border-top: 1px solid rgba(255,255,255,0.1); padding-top: 20px; }
.stat-row { display: flex; justify-content: space-between; margin-bottom: 15px; font-size: 13px; }
.label { color: #94a3b8; }
.green-glow { color: #22c55e; text-shadow: 0 0 5px #22c55e; }

.exit-button { width: 100%; padding: 12px; border: 1px solid #ef4444; background: transparent; color: #f87171; border-radius: 8px; cursor: pointer; margin-top: 10px; transition: 0.3s; font-weight: bold; }
.exit-button:hover { background: rgba(239, 68, 68, 0.2); border-color: #ff4d4f; color: #fff; }

.section-title { font-size: 18px; color: #fff; margin-bottom: 25px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; font-weight: bold; }
.m-card { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; transition: 0.3s; }
.m-card:hover { background: rgba(0, 240, 255, 0.05); border-color: rgba(0, 240, 255, 0.2); }
.m-info h4 { color: #fff; margin: 0 0 5px 0; font-size: 15px; }
.m-info p { color: #94a3b8; font-size: 12px; margin: 0; }
.m-status { color: #00f0ff; font-size: 11px; border: 1px solid #00f0ff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }

.token-section { margin-top: 30px; }
.token-section label { color: #94a3b8; font-size: 12px; display: block; margin-bottom: 10px; }
.token-display { background: rgba(0, 0, 0, 0.5); border: 1px solid rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; overflow-x: auto; color: #64748b; font-family: monospace; font-size: 12px; line-height: 1.5; }
.token-tip { font-size: 11px; color: #475569; margin-top: 10px; }
</style>