import { createRouter, createWebHistory } from 'vue-router'
import AppLayout from '../components/AppLayout.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
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

export default router