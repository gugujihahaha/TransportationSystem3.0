#!/usr/bin/env python3
"""
scripts/prepare_data.py
========================
四个实验共用的数据准备脚本。

流程：
1. 检查 base_segments.pkl 是否存在，不存在则调用 BaseGeoLifePreprocessor 生成
2. 交互询问清洗模式（也支持 --mode 命令行参数）
3. 检查 cleaned_{mode}.pkl 是否存在，存在则打印统计直接返回
4. 不存在则调用 BaseDataAdapter（用 Exp1DataAdapter 即可，它只做格式转换）
   的 process_segments(base_segments, use_cache=True) 生成缓存
5. 打印标签分布和保留率

关键：prepare_data.py 不写任何清洗逻辑，全部复用 common 里已有的
BaseGeoLifePreprocessor、BaseDataAdapter、TrajectoryCleaner。

用法：
    python scripts/prepare_data.py              # 交互模式
    python scripts/prepare_data.py --mode balanced   # 直接指定
"""
import os
import sys
import argparse
import pickle

# 项目根目录
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

BASE_SEGMENTS_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'base_segments.pkl')
PROCESSED_DIR      = os.path.join(PROJECT_DIR, 'data', 'processed')

# 从common导入
from common.base_preprocessor import BaseGeoLifePreprocessor
from common.adapters import Exp1DataAdapter


# ──────────────────────────────────────────────
# Step 1: 确保 base_segments.pkl 存在
# ──────────────────────────────────────────────

def ensure_base_segments(geolife_root: str):
    if os.path.exists(BASE_SEGMENTS_PATH):
        size_mb = os.path.getsize(BASE_SEGMENTS_PATH) / 1024 / 1024
        print(f"✅ base_segments.pkl 已存在 ({size_mb:.1f} MB)，跳过生成")
        return

    print("\n⚠️  base_segments.pkl 不存在，开始从GeoLife原始数据生成...")

    if not os.path.exists(geolife_root):
        print(f"❌ 找不到GeoLife数据目录: {geolife_root}")
        print("请通过 --geolife_root 参数指定正确路径")
        sys.exit(1)

    preprocessor = BaseGeoLifePreprocessor(geolife_root)
    segments = preprocessor.process_all_users()

    if not segments:
        print("❌ 未提取到任何轨迹段，请检查数据目录")
        sys.exit(1)

    preprocessor.save_to_cache(segments, BASE_SEGMENTS_PATH)
    print(f"✅ base_segments.pkl 生成完成，共 {len(segments)} 段")


# ──────────────────────────────────────────────
# Step 2: 选择清洗模式
# ──────────────────────────────────────────────

def select_mode(mode_arg: str) -> str:
    # 从Exp1DataAdapter获取可用的清洗模式
    available_modes = list(Exp1DataAdapter.CLEANING_PRESETS.keys())

    if mode_arg and mode_arg in available_modes:
        print(f"\n清洗模式: {mode_arg}（来自命令行参数）")
        return mode_arg

    print("\n" + "=" * 50)
    print("请选择数据清洗模式：")
    print("  1 = strict   (严格清洗，样本少但质量高)")
    print("  2 = balanced (均衡清洗，推荐)")
    print("  3 = gentle   (宽松清洗，样本多但噪声多)")
    print("=" * 50)

    mapping = {'1': 'strict', '2': 'balanced', '3': 'gentle',
               'strict': 'strict', 'balanced': 'balanced', 'gentle': 'gentle'}

    while True:
        choice = input("输入选项 (1/2/3 或 strict/balanced/gentle) [默认: 2]: ").strip()
        if choice == '':
            choice = '2'
        if choice in mapping:
            mode = mapping[choice]
            print(f"✅ 已选择: {mode}")
            return mode
        print(f"⚠️  无效输入 '{choice}'，请重新输入")


# ──────────────────────────────────────────────
# Step 3: 生成 cleaned_{mode}.pkl
# ──────────────────────────────────────────────

def ensure_cleaned_data(mode: str) -> str:
    # 获取缓存路径（由BaseDataAdapter自动生成）
    adapter = Exp1DataAdapter(enable_cleaning=True, cleaning_mode=mode, cache_dir=PROCESSED_DIR)
    cache_path = adapter._get_cache_path()

    if os.path.exists(cache_path):
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"\n✅ cleaned_{mode}.pkl 已存在 ({size_mb:.1f} MB)，跳过清洗")

        # 加载并打印统计
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        adapter._print_label_distribution(cached_data)

        return cache_path

    print(f"\n🔧 开始生成 cleaned_{mode}.pkl ...")

    # 加载 base_segments
    print(f"\n[1/2] 加载 base_segments.pkl ...")
    with open(BASE_SEGMENTS_PATH, 'rb') as f:
        base_segments = pickle.load(f)
    print(f"   加载完成: {len(base_segments)} 个原始轨迹段")

    # 调用BaseDataAdapter处理（自动使用缓存）
    print(f"\n[2/2] 调用Exp1DataAdapter处理数据...")
    cleaned_data = adapter.process_segments(base_segments, use_cache=True)

    # 返回缓存路径
    return cache_path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='四实验共用数据准备脚本')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['strict', 'balanced', 'gentle'],
                        help='清洗模式，不指定则交互选择')
    parser.add_argument('--geolife_root', type=str,
                        default=os.path.join(PROJECT_DIR, 'data', 'Geolife Trajectories 1.3'),
                        help='GeoLife数据根目录（仅在base_segments.pkl不存在时使用）')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("数据准备 - 四实验共用数据集")
    print("=" * 60)
    print(f"项目目录: {PROJECT_DIR}")
    print(f"数据目录: {PROCESSED_DIR}")

    # Step 1: 确保 base_segments.pkl 存在
    print("\n【Step 1】检查 base_segments.pkl ...")
    ensure_base_segments(args.geolife_root)

    # Step 2: 选择清洗模式
    print("\n【Step 2】选择清洗模式 ...")
    mode = select_mode(args.mode)

    # Step 3: 确保 cleaned_{mode}.pkl 存在
    print(f"\n【Step 3】检查 cleaned_{mode}.pkl ...")
    cleaned_path = ensure_cleaned_data(mode)

    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print("=" * 60)
    print(f"\n生成文件: {cleaned_path}")
    print("\n后续步骤：")
    print(f"  cd exp2 && python train.py")
    print(f"  cd exp1 && python train.py")
    print(f"  cd exp3 && python train.py")
    print(f"  cd exp4 && python train.py")


if __name__ == '__main__':
    main()
