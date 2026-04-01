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
      // 恢复你所有的原有层级和 name，保证 PageHeader 导航栏不崩溃
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
          meta: { title: 'TrafficRec 多模态数据大屏' } // 只改了展示文字
        },
        {
          path: 'congestion-analysis',
          name: 'congestion',
          component: () => import('../views/CongestionAnalysis.vue'),
          meta: { title: '场景验证 A：拥堵溯源' }
        },
        {
          path: 'green-travel',
          name: 'green',
          component: () => import('../views/GreenTravel.vue'),
          meta: { title: '场景验证 B：碳足迹与普惠' }
        },
        {
          path: 'tech-support',
          name: 'tech',
          component: () => import('../views/TechSupport.vue'),
          meta: { title: '核心算法与模型演进' }
        },
        {
          path: 'user-center',
          name: 'userCenter',
          component: () => import('../views/UserCenterView.vue'),
          meta: { title: '个人中心' }
        },
        { path: 'verification/congestion', redirect: '/congestion-analysis' },
        { path: 'verification/green', redirect: '/green-travel' },
        { path: 'tech', redirect: '/tech-support' },
        { path: 'user', redirect: '/user-center' }
      ],
    },
  ],
})

router.beforeEach((to, from) => {
  const authStore = useAuthStore() 
  if (to.path !== '/login' && !authStore.isAuthenticated()) {
    return { path: '/login' } 
  } 
  return true 
})

export default router