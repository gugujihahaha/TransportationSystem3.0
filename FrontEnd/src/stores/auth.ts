import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(localStorage.getItem('token') || null);
  const username = ref<string | null>(localStorage.getItem('username') || null);

  const setAuth = (newToken: string, newUsername: string) => {
    token.value = newToken;
    username.value = newUsername;
    localStorage.setItem('token', newToken);
    localStorage.setItem('username', newUsername);
  };

  const logout = () => {
    token.value = null;
    username.value = null;
    localStorage.removeItem('token');
    localStorage.removeItem('username');
  };

  const isAuthenticated = () => {
    return token.value !== null && token.value !== '';
  };

  return { token, username, setAuth, logout, isAuthenticated };
});