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
          meta: { title: 'TrafficRec 多模态数据大屏' }
        },
        {
          path: 'history',
          name: 'history',
          component: () => import('../views/HistoryView.vue'),
          meta: { title: '历史轨迹档案' }
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
          meta: { title: '系统工程架构与防御沙盘' } // 名字顺便改得更霸气
        },
        // 删除了原本在这里的 model-comparison 路由，因为已经融合进了 dashboard
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