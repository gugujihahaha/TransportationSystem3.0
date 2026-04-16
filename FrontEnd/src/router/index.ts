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
          component: () => import('../components/HomePage.vue'),
          meta: { title: 'TrafficRec 态势感知大屏' }
        },
        {
          path: 'insight',
          name: 'insight',
          component: () => import('../views/DataInsight.vue'),
          meta: { title: '数据洞察分析' }
        },
        {
          path: 'history',
          name: 'history',
          component: () => import('../views/HistoryView.vue'),
          meta: { title: '历史轨迹档案' }
        },
        {
          path: 'model-compare',
          name: 'model-compare',
          component: () => import('../views/ModelCompare.vue'),
          meta: { title: '场景验证 A：多模型对比' }
        },
        {
          path: 'carbon-heatmap',
          name: 'carbon-heatmap',
          component: () => import('../views/CarbonHeatmap.vue'),
          meta: { title: '场景验证 B：碳普惠热力' }
        },
        {
          path: 'green-travel',
          name: 'green',
          component: () => import('../views/GreenTravel.vue'),
          meta: { title: '场景验证 C：个人碳足迹' }
        },
        {
          path: 'tech-support',
          name: 'tech',
          component: () => import('../views/TechSupport.vue'),
          meta: { title: '系统工程架构与防御沙盘' } 
        },
        {
          path: 'user-center',
          name: 'userCenter',
          component: () => import('../views/UserCenterView.vue'),
          meta: { title: '个人中心' }
        },
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