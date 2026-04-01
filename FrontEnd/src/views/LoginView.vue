<template>
  <div class="login-container">
    <div class="login-box">
      <div class="header">
        <h2>交通方式识别系统</h2>
        <p>{{ isLogin ? '用户登录' : '新用户注册' }}</p>
      </div>

      <form @submit.prevent="handleSubmit" class="form-content">
        <div class="form-group">
          <label>用户名</label>
          <input 
            v-model="username" 
            type="text" 
            required 
            placeholder="请输入您的账号" 
            :disabled="isLoading"
          />
        </div>

        <div class="form-group">
          <label>密码</label>
          <input 
            v-model="password" 
            type="password" 
            required 
            placeholder="请输入密码" 
            :disabled="isLoading"
          />
        </div>

        <div v-if="errorMessage" class="error-msg">
          ⚠️ {{ errorMessage }}
        </div>

        <button type="submit" class="submit-btn" :disabled="isLoading">
          {{ isLoading ? '请稍候...' : (isLogin ? '登 录' : '注 册') }}
        </button>
      </form>

      <div class="toggle-mode">
        <span @click="toggleMode" class="toggle-text">
          {{ isLogin ? '还没有账号？点击这里注册' : '已有账号？返回登录' }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { authApi } from '@/api/auth'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const isLogin = ref(true)
const username = ref('')
const password = ref('')
const errorMessage = ref('')
const isLoading = ref(false)

// 切换登录/注册模式
const toggleMode = () => {
  isLogin.value = !isLogin.value
  errorMessage.value = ''
}

// 提交表单
const handleSubmit = async () => {
  if (!username.value || !password.value) return

  isLoading.value = true
  errorMessage.value = ''

  try {
    if (isLogin.value) {
      // 1. 执行登录
      const res = await authApi.login(username.value, password.value)
      // 2. 存入 Pinia 状态库
      authStore.setAuth(res.access_token, username.value)
      // 3. 登录成功，跳转到首页面板
      router.push('/') 
    } else {
      // 1. 执行注册
      await authApi.register(username.value, password.value)
      alert('注册成功！请使用新账号登录。')
      // 2. 注册成功后切回登录界面，清空密码
      isLogin.value = true
      password.value = ''
    }
  } catch (error: any) {
    errorMessage.value = error.message || '操作失败，请检查网络或后台服务'
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f0f2f5;
}

.login-box {
  width: 100%;
  max-width: 400px;
  padding: 40px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h2 {
  color: #1a1a1a;
  margin-bottom: 8px;
  font-size: 24px;
}

.header p {
  color: #666;
  font-size: 14px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  color: #333;
  font-size: 14px;
  font-weight: 500;
}

.form-group input {
  width: 100%;
  padding: 12px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 14px;
  transition: all 0.3s;
  box-sizing: border-box;
}

.form-group input:focus {
  border-color: #4A90E2;
  outline: none;
  box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
}

.submit-btn {
  width: 100%;
  padding: 12px;
  background-color: #4A90E2;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.3s;
  margin-top: 10px;
}

.submit-btn:hover:not(:disabled) {
  background-color: #357ABD;
}

.submit-btn:disabled {
  background-color: #a0c4eb;
  cursor: not-allowed;
}

.error-msg {
  color: #ff4d4f;
  font-size: 13px;
  margin-bottom: 15px;
  text-align: center;
}

.toggle-mode {
  margin-top: 20px;
  text-align: center;
}

.toggle-text {
  color: #4A90E2;
  font-size: 14px;
  cursor: pointer;
  transition: color 0.3s;
}

.toggle-text:hover {
  color: #357ABD;
  text-decoration: underline;
}
</style>