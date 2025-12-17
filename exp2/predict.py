import torch
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from src.model import TransportationModeClassifier


class TransportationPredictorExp2:
    def __init__(self, checkpoint_path="checkpoints/exp2_model.pth", kg_cache_path="cache/kg_data.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 安全加载模型与配置
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ 未找到模型权重文件: {checkpoint_path}")

        # 使用 torch 2.6+ 安全加载逻辑
        with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.label_encoder = ckpt['label_encoder']
        self.class_names = self.label_encoder.classes_
        config = ckpt['model_config']

        # 2. 初始化模型架构并加载权重
        self.model = TransportationModeClassifier(
            trajectory_feature_dim=config['trajectory_feature_dim'],
            kg_feature_dim=config['kg_feature_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3)
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 3. 加载知识图谱（用于推理时的特征补全，可选）
        self.kg = None
        if os.path.exists(kg_cache_path):
            with open(kg_cache_path, 'rb') as f:
                with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
                    self.kg = pickle.load(f)
            print(f"✅ 知识图谱加载成功")

        print(f"✅ Exp2 模型加载成功！类别: {list(self.class_names)}")

    def predict(self, traj_features, kg_features):
        """
        修复维度冲突后的预测方法 (Exp2)
        """
        # 1. 轨迹特征维度处理 (batch_size, seq_len, 9)
        if traj_features.ndim == 2:
            traj_features = np.expand_dims(traj_features, axis=0)

        # 2. KG 特征维度处理 (batch_size, 1, 11)
        # ⚠️ 关键修复：增加一个维度以满足模型内部 kg_out[:, -1, :] 的索引需求
        if kg_features.ndim == 1:
            # (11,) -> (1, 1, 11)
            kg_features = np.expand_dims(np.expand_dims(kg_features, axis=0), axis=1)
        elif kg_features.ndim == 2:
            # (batch, 11) -> (batch, 1, 11)
            kg_features = np.expand_dims(kg_features, axis=1)

        # 3. 转换为 Tensor 并移动到设备
        traj_tensor = torch.FloatTensor(traj_features).to(self.device)
        kg_tensor = torch.FloatTensor(kg_features).to(self.device)

        # 4. 执行推理
        with torch.no_grad():
            logits = self.model(traj_tensor, kg_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        # 5. 获取结果
        pred_label = self.label_encoder.inverse_transform(pred_idx.cpu().numpy())[0]
        score = confidence.cpu().item()

        return pred_label, score


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    # 初始化预测器
    predictor = TransportationPredictorExp2()

    # 模拟输入：
    # 轨迹特征: 长度50, 维度9
    dummy_traj = np.random.randn(50, 9)
    # KG特征: 维度11 (通常包含POI密度、路网密度等)
    dummy_kg = np.random.randn(11)

    label, score = predictor.predict(dummy_traj, dummy_kg)

    print("-" * 30)
    print(f"Exp2 预测结果: {label}")
    print(f"预测置信度: {score:.4f}")