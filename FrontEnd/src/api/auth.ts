// 把原来的 http://localhost:8000 删掉，直接用 /api 触发代理
const BASE_URL = '/api/auth';

export const authApi = {
  // 登录接口
  login: async (username: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch(`${BASE_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || '登录失败');
    }
    return response.json(); // 返回 { access_token, token_type }
  },

  // 注册接口
  register: async (username: string, password: string) => {
    const response = await fetch(`${BASE_URL}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || '注册失败');
    }
    return response.json();
  }
};