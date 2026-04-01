<template>
  <div class="user-center-page">
    <div class="page-header">
      <h2>用户管理中心</h2>
      <p class="subtitle">管理您的研究账号信息与系统偏好</p>
    </div>

    <div class="user-dashboard">
      <div class="profile-card">
        <div class="avatar-wrapper">
          <div class="avatar">{{ userInitials }}</div>
        </div>
        <h3 class="username">{{ authStore.username || '未命名研究员' }}</h3>
        <span class="role-badge">系统管理员 / 核心研发</span>
        
        <div class="info-list">
          <div class="info-item">
            <span class="label">邮箱：</span>
            <span class="value">{{ 'admin@traffic-rec.edu.cn' }}</span>
          </div>
          <div class="info-item">
            <span class="label">所属机构：</span>
            <span class="value">安徽师范大学 计算机与信息学院</span>
          </div>
          <div class="info-item">
            <span class="label">注册时间：</span>
            <span class="value">2026-01-15</span>
          </div>
        </div>

        <button class="edit-btn">修改个人信息</button>
      </div>

      <div class="settings-panel">
        <div class="setting-section">
          <h4>📌 当前参与竞赛与项目</h4>
          <ul class="project-list">
            <li>
              <div class="proj-title">“互联网+” 创新创业大赛</div>
              <div class="proj-desc">数字心理健康服务平台 (已归档)</div>
            </li>
            <li>
              <div class="proj-title">中国大学生计算机设计大赛</div>
              <div class="proj-desc">TrafficRec: 多模态交通方式识别系统 (前端负责人)</div>
            </li>
            <li>
              <div class="proj-title">全国大学生统计建模大赛</div>
              <div class="proj-desc">基于时空轨迹的减排核算模型 (进行中)</div>
            </li>
          </ul>
        </div>

        <div class="setting-section">
          <h4>⚙️ 系统偏好设置</h4>
          <div class="pref-item">
            <span>接收实验完成邮件通知</span>
            <input type="checkbox" checked class="toggle-switch">
          </div>
          <div class="pref-item">
            <span>开启大模型深度推理模式 (消耗Token增加)</span>
            <input type="checkbox" checked class="toggle-switch">
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()

const userInitials = computed(() => {
  const name = authStore.username || 'A'
  return name.charAt(0).toUpperCase()
})
</script>

<style scoped>
.user-center-page {
  padding: 30px;
  height: 100%;
  overflow-y: auto;
  box-sizing: border-box;
}

.page-header h2 {
  color: #e2e8f0;
  margin: 0 0 8px 0;
  font-size: 24px;
}
.subtitle {
  color: #94a3b8;
  margin: 0 0 30px 0;
}

.user-dashboard {
  display: flex;
  gap: 30px;
  align-items: flex-start;
}

.profile-card, .settings-panel {
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 30px;
}

.profile-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.avatar-wrapper {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background: linear-gradient(135deg, #0ea5e9, #3b82f6);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20px;
  box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4);
}

.avatar {
  font-size: 40px;
  color: white;
  font-weight: bold;
}

.username {
  color: white;
  font-size: 22px;
  margin: 0 0 10px 0;
}

.role-badge {
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 13px;
  margin-bottom: 30px;
  border: 1px solid rgba(16, 185, 129, 0.4);
}

.info-list {
  width: 100%;
  margin-bottom: 30px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  padding: 12px 0;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.1);
}

.info-item .label { color: #94a3b8; }
.info-item .value { color: #e2e8f0; font-weight: 500; }

.edit-btn {
  width: 100%;
  padding: 10px;
  background: transparent;
  border: 1px solid #0ea5e9;
  color: #0ea5e9;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s;
}
.edit-btn:hover {
  background: #0ea5e9;
  color: white;
}

.settings-panel {
  flex: 2;
}

.setting-section {
  margin-bottom: 40px;
}

.setting-section h4 {
  color: #e2e8f0;
  font-size: 16px;
  margin: 0 0 20px 0;
  border-left: 4px solid #0ea5e9;
  padding-left: 10px;
}

.project-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.project-list li {
  background: rgba(15, 23, 42, 0.5);
  padding: 15px;
  border-radius: 6px;
  margin-bottom: 15px;
  border: 1px solid rgba(255,255,255,0.05);
}

.proj-title {
  color: #38bdf8;
  font-weight: 500;
  margin-bottom: 5px;
}

.proj-desc {
  color: #94a3b8;
  font-size: 14px;
}

.pref-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px;
  background: rgba(15, 23, 42, 0.5);
  border-radius: 6px;
  margin-bottom: 10px;
  color: #cbd5e1;
}

/* 简单的开关样式伪造 */
.toggle-switch {
  width: 40px;
  height: 20px;
  cursor: pointer;
  accent-color: #0ea5e9;
}
</style>