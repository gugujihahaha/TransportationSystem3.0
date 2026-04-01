import { createRouter, createWebHistory } from 'vue-router'
import AppLayout from '../components/AppLayout.vue'
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
      component: AppLayout,
      // 只要是在 AppLayout 下面的子路由，都会被加上顶部导航栏
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
        // 👇 修复报错1：新增个人中心的路由注册
        {
          path: 'user-center',
          name: 'userCenter',
          component: () => import('../views/UserCenterView.vue'),
          meta: { title: '个人中心' }
        }
      ],
    },
  ],
})

// 👇 修复报错2：使用 Vue Router 4 推荐的 return 语法替代 next()
router.beforeEach((to, from) => {
  const authStore = useAuthStore() 
  
  // 判断该目标路由是否需要鉴权
  if (to.path !== '/login' && !authStore.isAuthenticated()) {
    return { path: '/login' } // 强行重定向到登录页
  } 
  
  return true // 检票通过，放行
})

export default router