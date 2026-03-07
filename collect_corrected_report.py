import json
import os
import pickle

def load_evaluation_results(exp_dir):
    """加载评估结果"""
    json_path = os.path.join(exp_dir, 'evaluation_results', 'evaluation_report.json')
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_model_structure():
    """检查模型结构"""
    print("=" * 80)
    print("检查模型结构")
    print("=" * 80)
    
    # 检查exp1的模型配置
    try:
        import torch
        exp1_checkpoint = torch.load('exp1/checkpoints/exp1_model.pth', map_location='cpu', weights_only=False)
        exp1_config = exp1_checkpoint['model_config']
        print(f"\nEXP1模型配置:")
        print(f"  trajectory_feature_dim: {exp1_config.get('trajectory_feature_dim', 'N/A')}")
        print(f"  hidden_dim: {exp1_config.get('hidden_dim', 'N/A')}")
        print(f"  num_layers: {exp1_config.get('num_layers', 'N/A')}")
        print(f"  num_classes: {exp1_config.get('num_classes', 'N/A')}")
    except Exception as e:
        print(f"\nEXP1模型配置加载失败: {e}")
    
    # 检查exp2的模型配置
    try:
        exp2_checkpoint = torch.load('exp2/checkpoints/exp2_model.pth', map_location='cpu', weights_only=False)
        exp2_config = exp2_checkpoint['model_config']
        print(f"\nEXP2模型配置:")
        print(f"  trajectory_feature_dim: {exp2_config.get('trajectory_feature_dim', 'N/A')}")
        print(f"  spatial_feature_dim: {exp2_config.get('spatial_feature_dim', 'N/A')}")
        print(f"  hidden_dim: {exp2_config.get('hidden_dim', 'N/A')}")
        print(f"  num_layers: {exp2_config.get('num_layers', 'N/A')}")
        print(f"  num_classes: {exp2_config.get('num_classes', 'N/A')}")
    except Exception as e:
        print(f"\nEXP2模型配置加载失败: {e}")

def main():
    print("=" * 80)
    print("交通方式识别论文数据收集完整报告（最终修正版）")
    print("=" * 80)
    print("\n本报告包含所有实验的评估结果，所有实验使用相同的数据集（8841个样本）")
    print("数据集划分：70%训练集 / 10%验证集 / 20%测试集")
    print("测试集样本数：1769个")
    
    print(f"\n{'=' * 80}")
    print("第一部分：四个实验的详细评估报告")
    print(f"{'=' * 80}")
    
    exp_dirs = {
        'EXP1': 'exp1',
        'EXP2': 'exp2',
        'EXP3': 'exp3',
        'EXP4': 'exp4'
    }
    
    for exp_name, exp_dir in exp_dirs.items():
        results = load_evaluation_results(exp_dir)
        if results:
            print(f"\n{'=' * 80}")
            print(f"{exp_name} 评估结果")
            print(f"{'=' * 80}")
            
            print(f"\n总体指标:")
            print(f"  准确率 (Accuracy):         {results['accuracy']:.4f}")
            print(f"  宏平均 F1 (Macro F1):     {results['macro avg']['f1-score']:.4f}")
            print(f"  加权 F1 (Weighted F1):    {results['weighted avg']['f1-score']:.4f}")
            
            print(f"\n各类别详细指标:")
            print(f"类别              Precision    Recall       F1-Score     Support")
            print(f"{'-' * 70}")
            
            classes = ['Bike', 'Bus', 'Car & taxi', 'Subway', 'Train', 'Walk']
            for cls in classes:
                if cls in results:
                    p = results[cls]['precision']
                    r = results[cls]['recall']
                    f1 = results[cls]['f1-score']
                    support = int(results[cls]['support'])
                    print(f"{cls:15s} {p:.4f}       {r:.4f}       {f1:.4f}       {support}")
            
            total_support = int(results['weighted avg']['support'])
            accuracy = results['accuracy']
            error_count = int(total_support * (1 - accuracy))
            error_rate = (1 - accuracy) * 100
            
            print(f"\n测试集样本总数: {total_support}")
            print(f"错误样本数: {error_count} ({error_rate:.2f}%)")
    
    print(f"\n{'=' * 80}")
    print("第二部分：测试集一致性验证")
    print(f"{'=' * 80}")
    
    print(f"\n{'实验':<10} {'测试集总数':<15} {'Subway':<10} {'Train':<10}")
    print(f"{'-' * 60}")
    
    all_consistent = True
    test_sizes = []
    subway_counts = []
    train_counts = []
    
    for exp_name, exp_dir in exp_dirs.items():
        results = load_evaluation_results(exp_dir)
        if results:
            total = int(results['weighted avg']['support'])
            subway = int(results.get('Subway', {}).get('support', 0))
            train = int(results.get('Train', {}).get('support', 0))
            print(f"{exp_name:<10} {total:<15} {subway:<10} {train:<10}")
            
            test_sizes.append(total)
            subway_counts.append(subway)
            train_counts.append(train)
    
    if len(set(test_sizes)) == 1 and len(set(subway_counts)) == 1 and len(set(train_counts)) == 1:
        print(f"\n✅ 所有实验测试集完全一致！")
        print(f"   测试集总数: {test_sizes[0]}")
        print(f"   Subway样本数: {subway_counts[0]}")
        print(f"   Train样本数: {train_counts[0]}")
    else:
        print(f"\n❌ 测试集不一致！")
        all_consistent = False
    
    print(f"\n{'=' * 80}")
    print("第三部分：实验对比分析")
    print(f"{'=' * 80}")
    
    if all_consistent:
        print(f"\n实验对比（使用相同数据集，测试集{test_sizes[0]}个样本）:")
        print(f"{'实验':<10} {'准确率':<12} {'宏平均F1':<12} {'加权F1':<12} {'错误数':<10}")
        print(f"{'-' * 70}")
        
        for exp_name, exp_dir in exp_dirs.items():
            results = load_evaluation_results(exp_dir)
            if results:
                accuracy = results['accuracy']
                macro_f1 = results['macro avg']['f1-score']
                weighted_f1 = results['weighted avg']['f1-score']
                support = int(results['weighted avg']['support'])
                error_count = int(support * (1 - accuracy))
                print(f"{exp_name:<10} {accuracy:<12.4f} {macro_f1:<12.4f} {weighted_f1:<12.4f} {error_count:<10}")
    
    print(f"\n{'=' * 80}")
    print("第四部分：消融实验分析")
    print(f"{'=' * 80}")
    
    print(f"\n消融实验设计:")
    print(f"  EXP1: 仅使用轨迹特征（9维）")
    print(f"  EXP2: 轨迹特征 + OSM空间特征（9+12=21维）")
    print(f"  EXP3: 轨迹特征 + OSM空间特征 + 天气特征（21+10=31维）")
    print(f"  EXP4: 与EXP2相同特征，使用标签平滑 + Focal Loss")
    
    if all_consistent:
        exp1_results = load_evaluation_results('exp1')
        exp2_results = load_evaluation_results('exp2')
        exp3_results = load_evaluation_results('exp3')
        exp4_results = load_evaluation_results('exp4')
        
        print(f"\n特征贡献分析:")
        print(f"  EXP1 -> EXP2: 添加OSM空间特征")
        if exp1_results and exp2_results:
            acc_change = exp2_results['accuracy'] - exp1_results['accuracy']
            error_change = int(exp2_results['weighted avg']['support'] * (1 - exp2_results['accuracy'])) - int(exp1_results['weighted avg']['support'] * (1 - exp1_results['accuracy']))
            print(f"    准确率变化: {exp2_results['accuracy']:.4f} - {exp1_results['accuracy']:.4f} = {acc_change:+.4f} ({acc_change*100:+.2f}%)")
            print(f"    错误数变化: {error_change:+d}")
        
        print(f"  EXP2 -> EXP3: 添加天气特征")
        if exp2_results and exp3_results:
            acc_change = exp3_results['accuracy'] - exp2_results['accuracy']
            error_change = int(exp3_results['weighted avg']['support'] * (1 - exp3_results['accuracy'])) - int(exp2_results['weighted avg']['support'] * (1 - exp2_results['accuracy']))
            print(f"    准确率变化: {exp3_results['accuracy']:.4f} - {exp2_results['accuracy']:.4f} = {acc_change:+.4f} ({acc_change*100:+.2f}%)")
            print(f"    错误数变化: {error_change:+d}")
        
        print(f"\n损失函数影响分析:")
        print(f"  EXP2 vs EXP4: 相同特征，不同损失函数")
        if exp2_results and exp4_results:
            print(f"    EXP2 (标准交叉熵): 准确率 = {exp2_results['accuracy']:.4f}")
            print(f"    EXP4 (标签平滑+Focal Loss): 准确率 = {exp4_results['accuracy']:.4f}")
            acc_change = exp4_results['accuracy'] - exp2_results['accuracy']
            print(f"    准确率变化: {acc_change:+.4f} ({acc_change*100:+.2f}%)")
    
    print(f"\n{'=' * 80}")
    print("第五部分：数据集统计信息")
    print(f"{'=' * 80}")
    
    print(f"\n数据集基本信息:")
    print(f"  总样本数: 8841")
    print(f"  训练集样本数: 6189 (70%)")
    print(f"  验证集样本数: 884 (10%)")
    print(f"  测试集样本数: 1768 (20%)")
    print(f"  注意: 由于train_test_split的整数取整，实际划分可能有±1的误差")
    
    if all_consistent:
        exp1_results = load_evaluation_results('exp1')
        if exp1_results:
            print(f"\n类别分布（测试集{test_sizes[0]}个样本）:")
            classes = ['Bike', 'Bus', 'Car & taxi', 'Subway', 'Train', 'Walk']
            for cls in classes:
                if cls in exp1_results:
                    support = int(exp1_results[cls]['support'])
                    percentage = support / test_sizes[0] * 100
                    print(f"  {cls:15s}: {support} ({percentage:.2f}%)")
    
    print(f"\n数据清洗策略:")
    print(f"  - 物理异常修复")
    print(f"  - 时间间隔插值")
    print(f"  - 轨迹平滑优化")
    print(f"  - 方向异常修正")
    print(f"  - NaN/Inf过滤")
    
    print(f"\n{'=' * 80}")
    print("第六部分：模型结构说明")
    print(f"{'=' * 80}")
    
    print(f"\nEXP1模型结构:")
    print(f"  输入: 轨迹特征（9维）")
    print(f"  编码器: 1个Bi-LSTM编码器，输入维度9，隐藏维度128")
    print(f"  分类器: 全连接层 + Softmax")
    
    print(f"\nEXP2模型结构:")
    print(f"  输入: 轨迹特征（9维）+ OSM空间特征（12维）= 融合特征（21维）")
    print(f"  编码器: 1个Bi-LSTM编码器，输入维度21，隐藏维度128")
    print(f"  分类器: 全连接层 + Softmax")
    
    print(f"\nEXP3模型结构:")
    print(f"  输入: 轨迹特征（9维）+ OSM空间特征（12维）+ 天气特征（10维）= 融合特征（31维）")
    print(f"  编码器: 1个Bi-LSTM编码器，输入维度31，隐藏维度128")
    print(f"  分类器: 全连接层 + Softmax")
    
    print(f"\nEXP4模型结构:")
    print(f"  输入: 轨迹特征（9维）+ OSM空间特征（12维）= 融合特征（21维）")
    print(f"  编码器: 1个Bi-LSTM编码器，输入维度21，隐藏维度128")
    print(f"  分类器: 全连接层 + Softmax")
    print(f"  损失函数: 标签平滑（0.1）+ Focal Loss（gamma=2.0）")
    
    print(f"\n{'=' * 80}")
    print("报告完成")
    print(f"{'=' * 80}")

if __name__ == '__main__':
    main()
