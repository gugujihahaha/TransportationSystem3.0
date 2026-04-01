<template>
  <div class="admin-user-center">
    <aside class="sidebar">
      <div class="user-brief">
        <div class="avatar">{{ authStore.username ? authStore.username.charAt(0).toUpperCase() : 'U' }}</div>
        <div class="name">{{ authStore.username }}</div>
        <div class="role">超级管理员</div>
      </div>
      
      <nav class="menu">
        <a :class="{ active: activeTab === 'profile' }" @click="activeTab = 'profile'">账号概览</a>
        <a :class="{ active: activeTab === 'activity' }" @click="activeTab = 'activity'">近期动态</a>
        <a :class="{ active: activeTab === 'api' }" @click="activeTab = 'api'">API 开发者</a>
        <a :class="{ active: activeTab === 'settings' }" @click="activeTab = 'settings'">安全设置</a>
      </nav>
    </aside>

    <main class="content-area">
      
      <section v-if="activeTab === 'profile'" class="panel">
        <h3>基本信息</h3>
        <div class="info-list">
          <div class="info-item"><label>登录账号：</label> <span>{{ authStore.username }}</span></div>
          <div class="info-item"><label>当前状态：</label> <span class="tag success">正常运转</span></div>
        </div>

        <h3 class="mt-4">模块访问权限</h3>
        <table class="auth-table">
          <thead>
            <tr>
              <th>系统模块</th>
              <th>授权状态</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>轨迹预测核心 (Exp1-Exp4)</td>
              <td><span class="tag success">● 授权有效</span></td>
            </tr>
            <tr>
              <td>底层数据大盘</td>
              <td><span class="tag success">● 授权有效</span></td>
            </tr>
            <tr>
              <td>AI 智能评估报告</td>
              <td><span class="tag success">● 授权有效</span></td>
            </tr>
          </tbody>
        </table>
      </section>

      <section v-if="activeTab === 'activity'" class="panel">
        <h3>系统操作日志</h3>
        <ul class="timeline">
          <li><strong>刚才</strong> - 成功登录交通方式识别系统</li>
          <li><strong>今天</strong> - 验证了 JWT 身份认证模块</li>
          <li><strong>今天</strong> - 访问了底层数据集统计接口</li>
        </ul>
      </section>

      <section v-if="activeTab === 'api'" class="panel">
        <h3>开发者 Token</h3>
        <p>您可以使用此 Bearer Token 在 Python 或 Postman 中直接调用后端 API。</p>
        <div class="token-box">
          <code>{{ authStore.token || '未获取到 Token' }}</code>
        </div>
      </section>

      <section v-if="activeTab === 'settings'" class="panel">
        <h3>危险操作</h3>
        <div class="danger-zone">
          <p>退出当前账号将清除本地的认证 Token，终止所有未完成的预测任务。</p>
          <button @click="handleLogout" class="btn-logout">安全退出系统</button>
        </div>
      </section>

    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

// 控制右侧面板切换的状态
const activeTab = ref('profile')

const handleLogout = () => {
  if (confirm('确定要退出系统吗？')) {
    authStore.logout()
    router.push('/login')
  }
}
</script>

<style scoped>
/* 基础骨架样式，后续可统一美化 */
.admin-user-center {
  display: flex;
  min-height: calc(100vh - 80px); /* 减去顶部导航栏高度 */
  background-color: #f4f7f9;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #e1e4e8;
  margin: 20px;
}

/* 左侧边栏 */
.sidebar {
  width: 250px;
  background-color: #fff;
  border-right: 1px solid #e1e4e8;
  padding: 30px 0;
  display: flex;
  flex-direction: column;
}

.user-brief {
  text-align: center;
  margin-bottom: 30px;
  padding: 0 20px;
}

.avatar {
  width: 64px;
  height: 64px;
  background-color: #2b3a4a;
  color: #fff;
  border-radius: 50%;
  font-size: 28px;
  line-height: 64px;
  margin: 0 auto 10px;
}

.name {
  font-weight: bold;
  font-size: 18px;
  color: #333;
}

.role {
  font-size: 13px;
  color: #888;
  margin-top: 4px;
}

.menu {
  display: flex;
  flex-direction: column;
}

.menu a {
  padding: 15px 30px;
  color: #555;
  cursor: pointer;
  border-left: 3px solid transparent;
  transition: all 0.2s;
}

.menu a:hover {
  background-color: #f8f9fa;
}

.menu a.active {
  background-color: #eef2f5;
  color: #2b3a4a;
  border-left-color: #2b3a4a;
  font-weight: bold;
}

/* 右侧内容区 */
.content-area {
  flex: 1;
  padding: 40px;
  background-color: #fcfcfc;
}

.panel h3 {
  margin-top: 0;
  margin-bottom: 20px;
  font-size: 20px;
  color: #2c3e50;
  border-bottom: 2px solid #eee;
  padding-bottom: 10px;
}

.mt-4 {
  margin-top: 40px;
}

.info-list .info-item {
  margin-bottom: 15px;
  font-size: 15px;
}

.info-item label {
  color: #666;
  display: inline-block;
  width: 100px;
}

.auth-table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border: 1px solid #eee;
}

.auth-table th, .auth-table td {
  padding: 12px 15px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

.auth-table th {
  background-color: #f9fafb;
  color: #555;
}

.tag {
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 4px;
}

.tag.success {
  background-color: #e6f7ff;
  color: #1890ff;
  border: 1px solid #91d5ff;
}

.timeline {
  list-style-type: disc;
  padding-left: 20px;
  line-height: 2;
  color: #444;
}

.token-box {
  background-color: #282c34;
  color: #abb2bf;
  padding: 15px;
  border-radius: 6px;
  overflow-x: auto;
  margin-top: 15px;
  font-family: monospace;
}

.danger-zone {
  border: 1px solid #ffccc7;
  background-color: #fff2f0;
  padding: 20px;
  border-radius: 6px;
}

.danger-zone p {
  color: #ff4d4f;
  margin-bottom: 15px;
}

.btn-logout {
  background-color: #ff4d4f;
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
}

.btn-logout:hover {
  background-color: #cf1322;
}
</style>