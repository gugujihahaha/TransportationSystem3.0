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
        // 1. 首页总览 
        {
          path: '',
          name: 'home',
          component: () => import('../views/HomeView.vue'),
          meta: { title: '首页总览' }
        },
        // 2. 态势感知 
        {
          path: 'dashboard',
          name: 'dashboard',
          component: () => import('../components/HomePage.vue'),
          meta: { title: 'TrafficRec 态势感知大屏' }
        },
        // 3. 数据洞察 
        {
          path: 'insight',
          name: 'insight',
          component: () => import('../views/DataInsight.vue'),
          meta: { title: '数据洞察分析' }
        },
        // 4. 历史记录 
        {
          path: 'history',
          name: 'history',
          component: () => import('../views/HistoryView.vue'),
          meta: { title: '历史轨迹档案' }
        },
        // 5. 拥堵溯源
        {
          path: 'congestion-analysis',
          name: 'congestion',
          component: () => import('../views/CongestionAnalysis.vue'),
          meta: { title: '场景验证 A：拥堵溯源' }
        },
        // 6. 绿色出行
        {
          path: 'green-travel',
          name: 'green',
          component: () => import('../views/GreenTravel.vue'),
          meta: { title: '场景验证 B：碳足迹与普惠' }
        },
        // 7. 技术支撑
        {
          path: 'tech-support',
          name: 'tech',
          component: () => import('../views/TechSupport.vue'),
          meta: { title: '系统工程架构与防御沙盘' } 
        },
        // 8. 个人中心
        {
          path: 'user-center',
          name: 'userCenter',
          component: () => import('../views/UserCenterView.vue'),
          meta: { title: '个人中心' }
        },
        // 9. 区域详情 
        {
          path: 'region/:name',
          name: 'RegionDetail',
          component: () => import('../views/RegionDetail.vue'),
          meta: { title: '区域交通详情' }
        }
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