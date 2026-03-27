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
          name: 'map-analysis',
          component: () => import('../views/MapAnalysis.vue'),
        },
        {
          path: 'model-comparison',
          name: 'model-comparison',
          component: () => import('../views/ModelComparison.vue'),
        },
        {
          path: 'data-overview',
          name: 'data-overview',
          component: () => import('../views/DataOverview.vue'),
        },
        {
          path: 'about',
          name: 'about',
          component: () => import('../views/About.vue'),
        },
      ],
    },
  ],
})

export default router
