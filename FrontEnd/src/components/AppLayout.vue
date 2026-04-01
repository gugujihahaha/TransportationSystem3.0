<template>
  <div class="screen-wrapper">
    <div class="tech-bg"></div>

    <PageHeader />

    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade-slide" mode="out-in">
          <component :is="Component" class="page-component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup lang="ts">
import PageHeader from './PageHeader.vue'
</script>

<style scoped>
/* 整个大屏容器 */
.screen-wrapper {
  display: flex;
  flex-direction: column; /* 纵向排列 Header 和 Content */
  height: 100vh; /* 强制占满全屏，防止地图高度丢失 */
  width: 100vw;
  background-color: #030816;
  color: #fff;
  overflow: hidden; /* 防止出现双滚动条 */
  position: relative;
}

/* 背景装饰 */
.tech-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: 
    radial-gradient(circle at 10% 20%, rgba(0, 112, 243, 0.08) 0%, transparent 20%),
    radial-gradient(circle at 90% 80%, rgba(0, 240, 255, 0.08) 0%, transparent 20%);
  pointer-events: none;
  z-index: 0;
}

/* 核心内容区：这是地图显示的容器 */
.main-content {
  flex: 1; /* 自动撑开占据剩余所有空间 */
  position: relative;
  z-index: 1;
  width: 100%;
  padding: 0; /* 地图通常需要铺满，去除内边距 */
  overflow: hidden;
}

/* 页面组件样式：确保子页面也是满高的 */
.page-component {
  height: 100%;
  width: 100%;
}

/* 页面切换平滑动画 */
.fade-slide-enter-active, .fade-slide-leave-active {
  transition: all 0.3s ease;
}
.fade-slide-enter-from { opacity: 0; transform: translateY(20px); }
.fade-slide-leave-to { opacity: 0; transform: translateY(-20px); }
</style>