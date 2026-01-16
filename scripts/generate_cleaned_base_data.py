#!/usr/bin/env python3
"""
生成清洗后的基础数据脚本
执行此脚本生成所有实验共用的清洗后数据，避免重复清洗

使用方法:
    python scripts/generate_cleaned_base_data.py
    python scripts/generate_cleaned_base_data.py --cleaning_mode strict
    python scripts/generate_cleaned_base_data.py --max_users 10  # 快速测试
"""
import sys
import os
import pickle
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.trajectory_cleaner import TrajectoryCleaner


def load_base_data(base_data_path: str) -> List[Dict]:
    """加载基础数据"""
    print(f"\n[1/4] 加载基础数据: {base_data_path}")
    
    if not os.path.exists(base_data_path):
        print(f"❌ 错误: 找不到基础数据文件: {base_data_path}")
        print("\n请先运行以下命令生成基础数据:")
        print("  python scripts/generate_base_data.py")
        return None
    
    with open(base_data_path, 'rb') as f:
        base_segments = pickle.load(f)
    
    print(f"   ✓ 加载完成: {len(base_segments)} 个轨迹段")
    return base_segments


def setup_cleaning_params(cleaning_mode: str) -> Dict[str, Any]:
    """根据清洗模式设置参数"""
    if cleaning_mode == 'strict':
        return {
            'max_time_gap': 180.0,
            'max_bearing_change': 120.0,
            'min_segment_length': 15,
            'max_outlier_ratio': 0.15,
            'enable_smoothing': True,
            'smoothing_window': 7
        }
    elif cleaning_mode == 'gentle':
        return {
            'max_time_gap': 600.0,
            'max_bearing_change': 180.0,
            'min_segment_length': 8,
            'max_outlier_ratio': 0.35,
            'enable_smoothing': False,
            'smoothing_window': 3
        }
    else:  # balanced (默认)
        return {
            'max_time_gap': 300.0,
            'max_bearing_change': 150.0,
            'min_segment_length': 10,
            'max_outlier_ratio': 0.25,
            'enable_smoothing': True,
            'smoothing_window': 5
        }


def clean_segments(base_segments: List[Dict], cleaning_mode: str) -> tuple:
    """清洗所有轨迹段"""
    print(f"\n[2/4] 开始数据清洗 (模式: {cleaning_mode})...")
    
    # 设置清洗参数
    params = setup_cleaning_params(cleaning_mode)
    
    # 创建清洗器
    cleaner = TrajectoryCleaner(**params)
    
    # 清洗所有轨迹段
    cleaned_segments = []
    discarded_count = 0
    
    for segment in tqdm(base_segments, desc="[清洗轨迹段]"):
        # 提取轨迹特征 (9维)
        trajectory = segment['raw_points'].values
        
        # 获取标签
        label = segment['label']
        
        # 清洗轨迹段
        cleaned_trajectory, kept = cleaner.clean_segment(trajectory, label)
        
        if kept:
            # 更新轨迹数据
            segment['cleaned_trajectory'] = cleaned_trajectory
            segment['cleaned'] = True
            cleaned_segments.append(segment)
        else:
            discarded_count += 1
    
    # 获取清洗统计
    stats = cleaner.cleaning_stats
    
    print(f"\n   ✓ 清洗完成:")
    print(f"     - 原始轨迹段: {stats['total_segments']}")
    print(f"     - 保留轨迹段: {stats['segments_kept']}")
    print(f"     - 丢弃轨迹段: {stats['segments_discarded']}")
    print(f"     - 剔除异常点: {stats['outliers_removed']}")
    print(f"     - 插值点数: {stats['points_interpolated']}")
    print(f"     - 平滑点数: {stats['points_smoothed']}")
    print(f"     - 保留率: {stats['segments_kept'] / stats['total_segments'] * 100:.2f}%")
    
    return cleaned_segments, stats


def save_cleaned_data(cleaned_segments: List[Dict], 
                     cleaning_stats: Dict,
                     output_path: str,
                     cleaning_mode: str):
    """保存清洗后的数据"""
    print(f"\n[3/4] 保存清洗后的数据: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存数据
    with open(output_path, 'wb') as f:
        pickle.dump((cleaned_segments, cleaning_stats, cleaning_mode), f)
    
    print(f"   ✓ 保存完成")
    print(f"   - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"   - 轨迹段数: {len(cleaned_segments)}")


def print_summary(cleaned_segments: List[Dict],
                cleaning_stats: Dict,
                cleaning_mode: str,
                output_path: str):
    """打印汇总信息"""
    print(f"\n[4/4] 汇总信息")
    
    print("\n" + "=" * 80)
    print("数据清洗完成")
    print("=" * 80)
    
    print(f"\n清洗模式: {cleaning_mode}")
    print(f"输出路径: {output_path}")
    
    print(f"\n数据统计:")
    print(f"  - 保留轨迹段: {len(cleaned_segments)}")
    print(f"  - 保留率: {cleaning_stats['segments_kept'] / cleaning_stats['total_segments'] * 100:.2f}%")
    
    print(f"\n清洗统计:")
    print(f"  - 剔除异常点: {cleaning_stats['outliers_removed']}")
    print(f"  - 插值点数: {cleaning_stats['points_interpolated']}")
    print(f"  - 平滑点数: {cleaning_stats['points_smoothed']}")
    
    print(f"\n丢弃原因:")
    for reason, count in cleaning_stats['discard_reasons'].items():
        if count > 0:
            print(f"  - {reason}: {count}")
    
    print("\n🚀 后续使用:")
    print("  现在可以直接运行各实验的训练脚本，无需重复清洗数据：")
    print("  ")
    print("  # Exp1")
    print("  cd exp1")
    print("  python train.py --use_cleaned_data")
    print("  ")
    print("  # Exp2")
    print("  cd exp2")
    print("  python train.py --use_cleaned_data")
    print("  ")
    print("  # Exp3")
    print("  cd exp3")
    print("  python train.py --use_cleaned_data")
    print("  ")
    print("  # Exp4")
    print("  cd exp4")
    print("  python train.py --use_cleaned_data")
    print("  ")
    print("  # Exp5")
    print("  cd exp5")
    print("  python train.py --use_cleaned_data")
    print("  ")


def main():
    parser = argparse.ArgumentParser(
        description='生成清洗后的基础数据（所有实验共用）'
    )
    parser.add_argument(
        '--base_data',
        type=str,
        default='../data/processed/base_segments.pkl',
        help='基础数据路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/processed/cleaned_segments_balanced.pkl',
        help='输出路径'
    )
    parser.add_argument(
        '--cleaning_mode',
        type=str,
        default='balanced',
        choices=['strict', 'balanced', 'gentle'],
        help='清洗模式'
    )
    parser.add_argument(
        '--max_users',
        type=int,
        default=None,
        help='最大用户数（用于快速测试，留空处理全部）'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("生成清洗后的基础数据（所有实验共用）")
    print("=" * 80)
    
    print(f"\n配置:")
    print(f"  基础数据: {args.base_data}")
    print(f"  输出路径: {args.output}")
    print(f"  清洗模式: {args.cleaning_mode}")
    print(f"  最大用户数: {args.max_users or '全部'}")
    
    # 加载基础数据
    base_segments = load_base_data(args.base_data)
    if base_segments is None:
        return
    
    # 限制用户数（快速测试）
    if args.max_users:
        user_ids = set()
        filtered_segments = []
        for seg in base_segments:
            if seg['user_id'] not in user_ids:
                user_ids.add(seg['user_id'])
                filtered_segments.append(seg)
            if len(user_ids) >= args.max_users:
                break
        base_segments = filtered_segments
        print(f"   ✓ 限制为 {args.max_users} 个用户: {len(base_segments)} 个轨迹段")
    
    # 清洗数据
    cleaned_segments, cleaning_stats = clean_segments(base_segments, args.cleaning_mode)
    
    # 保存清洗后的数据
    save_cleaned_data(cleaned_segments, cleaning_stats, args.output, args.cleaning_mode)
    
    # 打印汇总信息
    print_summary(cleaned_segments, cleaning_stats, args.cleaning_mode, args.output)
    
    print("\n✅ 清洗后的基础数据生成完成！")


if __name__ == "__main__":
    main()
