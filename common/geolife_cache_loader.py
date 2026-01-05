"""
通用 GeoLife 数据加载器（带缓存） - 适用于所有实验
放置位置: 项目根目录/common/geolife_cache_loader.py

功能：
1. 一次性加载和预处理 GeoLife 数据
2. 缓存到 data/cache/geolife_processed.pkl
3. 自动检测缓存有效性
4. 所有实验共享同一份数据

使用方法：
    from common.geolife_cache_loader import load_geolife_cached

    segments = load_geolife_cached(
        geolife_root="../data/Geolife Trajectories 1.3",
        max_users=None,
        force_reload=False
    )
"""

import os
import pickle
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# 导入本模块内的基础加载器（完全独立）
from data_loader_base import GeoLifeDataLoader, preprocess_segments_base

# ============================================================
# 路径配置（⭐ 核心：与运行目录无关）
# ============================================================
COMMON_DIR = Path(__file__).resolve().parent          # common/
PROJECT_ROOT = COMMON_DIR.parent                      # 项目根目录
DATA_DIR = PROJECT_ROOT / "data"

# GeoLife 原始数据（⚠️ 直接指向 Data）
GEOLIFE_ROOT = DATA_DIR / "Geolife Trajectories 1.3" / "Data"

# cache 目录（与 common 同级的 data/cache）
CACHE_DIR = DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "geolife_processed_v1.pkl"
META_FILE = CACHE_DIR / "geolife_meta_v1.json"

CACHE_VERSION = "v1"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 元数据管理
# ============================================================
def compute_dir_hash(directory: str) -> str:
    """计算目录的哈希值（用于检测数据变化）"""
    hash_md5 = hashlib.md5()

    # 只检查关键文件：labels.txt 和部分 .plt 文件
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            if file == "labels.txt" or (file.endswith(".plt") and files.index(file) < 5):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        hash_md5.update(f.read(1024))  # 只读前1KB加速
                except:
                    pass

    return hash_md5.hexdigest()


def save_cache_metadata(geolife_root: str, num_segments: int,
                        class_names: List[str], class_counts: dict):
    """保存缓存元数据"""
    meta = {
        "version": CACHE_VERSION,
        "created_at": datetime.now().isoformat(),
        "geolife_root": geolife_root,
        "data_hash": compute_dir_hash(geolife_root),
        "num_segments": num_segments,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_counts": class_counts,
        "feature_dim": 9,
        "sequence_length": 100,
        "label_mapping": {
            "taxi": "car & taxi",
            "car": "car & taxi"
        }
    }

    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"✅ 元数据已保存: {META_FILE}")


def validate_cache(geolife_root: str) -> bool:
    """验证缓存是否有效"""
    if not os.path.exists(CACHE_FILE) or not os.path.exists(META_FILE):
        print("⚠️  缓存文件不存在")
        return False

    try:
        with open(META_FILE, 'r') as f:
            meta = json.load(f)

        # 检查版本
        if meta.get('version') != CACHE_VERSION:
            print(f"⚠️  缓存版本不匹配: {meta.get('version')} != {CACHE_VERSION}")
            return False

        # 检查数据哈希
        current_hash = compute_dir_hash(geolife_root)
        if meta.get('data_hash') != current_hash:
            print(f"⚠️  数据已更改，缓存失效")
            return False

        print(f"✅ 缓存验证通过")
        print(f"   创建时间: {meta.get('created_at')}")
        print(f"   轨迹段数: {meta.get('num_segments')}")
        print(f"   类别数: {meta.get('num_classes')}")

        return True

    except Exception as e:
        print(f"⚠️  缓存验证失败: {e}")
        return False


# ============================================================
# 核心加载函数
# ============================================================
def load_geolife_raw(geolife_root: str,
                     max_users: Optional[int] = None) -> List[Tuple[pd.DataFrame, str]]:
    """
    原始加载（不使用缓存）

    Returns:
        List[(pd.DataFrame, label_str)]  # 原始轨迹段
    """
    print("=" * 60)
    print("🔄 从头加载 GeoLife 数据（无缓存）")
    print("=" * 60)

    loader = GeoLifeDataLoader(geolife_root)
    users = loader.get_all_users()

    if max_users and max_users < len(users):
        users = users[:max_users]
        print(f"⚠️  限制用户数: {max_users}/{len(loader.get_all_users())}")

    print(f"找到 {len(users)} 个用户")

    all_segments = []

    for user_id in tqdm(users, desc="📖 读取用户轨迹"):
        labels = loader.load_labels(user_id)
        if labels.empty:
            continue

        traj_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
        if not os.path.exists(traj_dir):
            continue

        for f in os.listdir(traj_dir):
            if not f.endswith(".plt"):
                continue

            try:
                traj = loader.load_trajectory(os.path.join(traj_dir, f))
                if traj.empty:
                    continue

                segments = loader.segment_trajectory(traj, labels)
                all_segments.extend(segments)

            except Exception as e:
                warnings.warn(f"加载失败 {f}: {e}")
                continue

    print(f"\n📊 原始轨迹段数: {len(all_segments)}")

    return all_segments


def preprocess_and_cache(segments: List[Tuple[pd.DataFrame, str]],
                         geolife_root: str) -> List[Tuple[np.ndarray, str]]:
    """
    预处理并缓存数据

    Returns:
        List[(features_array, label_str)]  # 处理后的特征
    """
    print("\n🔧 预处理轨迹段...")

    # 使用标准预处理（9维特征，序列长度100）
    processed = preprocess_segments_base(
        segments,
        min_length=10,
        max_length=200,
        target_length=100
    )

    print(f"✅ 预处理完成: {len(processed)} 个有效段")

    # 统计类别分布
    from collections import Counter
    labels = [label for _, label in processed]
    class_counts = dict(Counter(labels))
    class_names = sorted(list(set(labels)))

    print("\n📈 类别分布:")
    for cls in class_names:
        print(f"   {cls}: {class_counts.get(cls, 0)}")

    # 保存缓存
    print(f"\n💾 保存缓存到: {CACHE_FILE}")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(processed, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 保存元数据
    save_cache_metadata(geolife_root, len(processed), class_names, class_counts)

    cache_size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
    print(f"✅ 缓存保存完成 ({cache_size_mb:.2f} MB)")

    return processed


def load_geolife_cached(geolife_root: str = "../data/Geolife Trajectories 1.3",
                        max_users: Optional[int] = None,
                        force_reload: bool = False) -> List[Tuple[np.ndarray, str]]:
    """
    智能加载 GeoLife 数据（带缓存）

    Args:
        geolife_root: GeoLife 数据根目录
        max_users: 最大用户数（用于快速测试）
        force_reload: 是否强制重新加载

    Returns:
        List[(features_array, label_str)]
        - features_array: (100, 9) 轨迹特征
        - label_str: 标签字符串（已合并 taxi → car & taxi）

    使用示例：
        # Exp1
        segments = load_geolife_cached()

        # Exp2 (需要在此基础上提取 KG 特征)
        segments = load_geolife_cached()
        for seg, label in segments:
            kg_features = kg.extract_kg_features(seg)

        # Exp6 (MASO)
        segments = load_geolife_cached()
        maso_features = organizer.extract_maso_features(seg)
    """

    # 1. 检查缓存
    if not force_reload and validate_cache(geolife_root):
        print("\n" + "=" * 60)
        print("⚡ 从缓存加载数据（秒级加载）")
        print("=" * 60)

        with open(CACHE_FILE, 'rb') as f:
            processed = pickle.load(f)

        print(f"✅ 加载完成: {len(processed)} 个轨迹段")

        # 如果指定了 max_users，需要过滤
        if max_users is not None:
            print(f"⚠️  限制用户数不适用于缓存数据")

        return processed

    # 2. 从头加载
    print("\n⚠️  缓存无效或强制重新加载")

    raw_segments = load_geolife_raw(geolife_root, max_users)
    processed = preprocess_and_cache(raw_segments, geolife_root)

    return processed


# ============================================================
# 辅助函数：过滤指定类别
# ============================================================
def filter_classes(segments: List[Tuple[np.ndarray, str]],
                   target_classes: List[str]) -> List[Tuple[np.ndarray, str]]:
    """
    过滤指定类别

    Args:
        segments: 轨迹段列表
        target_classes: 目标类别列表

    Returns:
        过滤后的轨迹段

    Example:
        # Exp1 & Exp6: 6 类
        segments = filter_classes(segments,
            ['walk', 'bike', 'car & taxi', 'bus', 'train', 'subway'])

        # Exp2: 7 类
        segments = filter_classes(segments,
            ['Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Airplane', 'Other'])
    """
    filtered = [s for s in segments if s[1] in target_classes]

    print(f"\n🔍 过滤类别:")
    print(f"   目标类别: {target_classes}")
    print(f"   过滤前: {len(segments)} 个轨迹段")
    print(f"   过滤后: {len(filtered)} 个轨迹段")

    from collections import Counter
    labels = [label for _, label in filtered]
    for cls in target_classes:
        count = Counter(labels).get(cls, 0)
        print(f"   {cls}: {count}")

    return filtered


# ============================================================
# 清理缓存
# ============================================================
def clear_cache():
    """清空缓存"""
    print("\n🗑️  清空缓存...")

    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"   ✅ 已删除: {CACHE_FILE}")

    if os.path.exists(META_FILE):
        os.remove(META_FILE)
        print(f"   ✅ 已删除: {META_FILE}")

    print("缓存已清空")


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 示例1: 基础使用
    segments = load_geolife_cached(
        geolife_root="../data/Geolife Trajectories 1.3"
    )

    # 示例2: 过滤类别（Exp1/Exp6）
    segments_6class = filter_classes(segments,
                                     ['walk', 'bike', 'car & taxi', 'bus', 'train', 'subway'])

    # 示例3: 强制重新加载
    segments = load_geolife_cached(force_reload=True)

    # 示例4: 清空缓存
    # clear_cache()