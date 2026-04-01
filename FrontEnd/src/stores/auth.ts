import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useAuthStore = defineStore('auth', () => {
  // 从 localStorage 初始化 token，防止刷新丢失
  const token = ref<string | null>(localStorage.getItem('token') || null);
  const username = ref<string | null>(localStorage.getItem('username') || null);

  // 登录成功后保存状态
  const setAuth = (newToken: string, newUsername: string) => {
    token.value = newToken;
    username.value = newUsername;
    localStorage.setItem('token', newToken);
    localStorage.setItem('username', newUsername);
  };

  // 退出登录，清空状态
  const logout = () => {
    token.value = null;
    username.value = null;
    localStorage.removeItem('token');
    localStorage.removeItem('username');
  };

  // 检查是否已登录
  const isAuthenticated = () => {
    return token.value !== null && token.value !== '';
  };

  return { token, username, setAuth, logout, isAuthenticated };
});