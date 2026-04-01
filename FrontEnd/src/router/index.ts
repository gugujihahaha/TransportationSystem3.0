import { createRouter, createWebHistory } from 'vue-router'
import AppLayout from '../components/AppLayout.vue'
import HomeView from '../views/HomeView.vue'
import LoginView from '../views/LoginView.vue' 
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: LoginView,
      meta: { requiresAuth: false } 
    },
    {
      path: '/',
      name: 'home',
      component: HomeView,
      meta: { requiresAuth: true } // 标记需要鉴权
    },
    {
      path: '/',
      component: AppLayout,
      children: [
        {
          path: '',
          name: 'home',
          component: () => import('../views/HomeView.vue'),
          meta: { title: '首页总览' }
        },
        {
          path: 'dashboard',
          name: 'dashboard',
          component: () => import('../views/DataDashboard.vue'),
          meta: { title: '数据与实验驾驶舱' }
        },
        {
          path: 'congestion-analysis',
          name: 'congestion',
          component: () => import('../views/CongestionAnalysis.vue'),
          meta: { title: '应用验证A：拥堵溯源解析' }
        },
        {
          path: 'green-travel',
          name: 'green',
          component: () => import('../views/GreenTravel.vue'),
          meta: { title: '应用验证B：绿色出行与碳普惠' }
        },
        {
          path: 'tech-support',
          name: 'tech',
          component: () => import('../views/TechSupport.vue'),
          meta: { title: '技术支撑与系统架构' }
        },
      ],
    },
  ],
})

router.beforeEach((to, from, next) => {
  const authStore = useAuthStore() 
  
  // 判断该目标路由是否需要鉴权（或者是所有的页面只要不是 /login 就拦截）
  // 这里我们采用最严格的模式：只要去的不是登录页，且没有 Token，就全部打回登录页！
  if (to.path !== '/login' && !authStore.isAuthenticated()) {
    next({ path: '/login' }) // 强行重定向到登录页
  } else {
    next() // 检票通过，放行
  }
})

export default router