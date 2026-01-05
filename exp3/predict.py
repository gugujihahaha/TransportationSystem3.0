import torch
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from src.model import TransportationModeClassifier


class TransportationPredictorExp3:
    def __init__(self, checkpoint_path="checkpoints/exp3_model.pth", kg_cache_path="cache/kg_data_v1.pkl"):
        """
        初始化实验三预测器
        :param checkpoint_path: 训练好的模型权重路径
        :param kg_cache_path: 增强知识图谱缓存路径 (Exp3 默认为 v1 版本)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 加载模型 Checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ 未找到 Exp3 模型权重: {checkpoint_path}")

        # 使用 safe_globals 确保在 PyTorch 2.6+ 下安全加载 LabelEncoder
        with torch.serialization.safe_globals({LabelEncoder: LabelEncoder, np.str_: np.str_}):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.label_encoder = ckpt['label_encoder']
        self.class_names = self.label_encoder.classes_
        config = ckpt['model_config']

        # 2. 初始化模型架构 (自动适配 15 维 KG 特征)
        self.model = TransportationModeClassifier(
            trajectory_feature_dim=config['trajectory_feature_dim'],  # 通常为 9
            kg_feature_dim=config['kg_feature_dim'],  # Exp3 应为 15
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3)
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"✅ Exp3 模型加载成功！")
        print(f"📊 特征配置: 轨迹维度={config['trajectory_feature_dim']}, KG维度={config['kg_feature_dim']}")
        print(f"🏷️  支持类别: {list(self.class_names)}")

    def predict(self, traj_features, kg_features):
        """
        输入参数:
            traj_features: (seq_len, 9) 的轨迹特征矩阵
            kg_features: (15,) 的增强知识图谱特征向量
        返回:
            pred_label: 预测的交通方式字符串
            confidence: 置信度分数
        """
        # 1. 轨迹特征维度处理 -> (1, seq_len, 9)
        if traj_features.ndim == 2:
            traj_features = np.expand_dims(traj_features, axis=0)

        # 2. KG 特征维度处理 -> (1, 1, 15)
        # ⚠️ 针对实验二遇到的 IndexError 进行了预先修复：
        # 实验三的模型 forward 同样预期 KG 具有时间维度 [batch, 1, dim]
        if kg_features.ndim == 1:
            kg_features = np.expand_dims(np.expand_dims(kg_features, axis=0), axis=1)
        elif kg_features.ndim == 2:
            kg_features = np.expand_dims(kg_features, axis=1)

        # 3. 转换为 Tensor
        traj_tensor = torch.FloatTensor(traj_features).to(self.device)
        kg_tensor = torch.FloatTensor(kg_features).to(self.device)

        # 4. 推理
        with torch.no_grad():
            logits = self.model(traj_tensor, kg_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        # 5. 映射回标签
        pred_label = self.label_encoder.inverse_transform(pred_idx.cpu().numpy())[0]
        score = confidence.cpu().item()

        return pred_label, score


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    # 初始化 Exp3 预测器
    predictor = TransportationPredictorExp3()

    # 模拟输入：
    # 轨迹特征 (9维)
    dummy_traj = np.random.randn(50, 9)
    # 实验三增强 KG 特征 (15维：包含更多的 POI 类型和空间关系)
    dummy_kg = np.random.randn(15)

    label, score = predictor.predict(dummy_traj, dummy_kg)

    print("\n" + "=" * 40)
    print(f"【实验三预测结论】")
    print(f"识别模式: {label}")
    print(f"置信水平: {score:.4%}")
    print("=" * 40)