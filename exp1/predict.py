import torch
import numpy as np
import pickle
import os
from src.model import TransportationModeClassifier


class TrajectoryPredictor:
    def __init__(self, checkpoint_path="checkpoints/exp1_model.pth", cache_path="cache/exp1_processed_features.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 加载标签映射器 (LabelEncoder)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"❌ 未找到缓存文件 {cache_path}，请先运行 train.py")

        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
            self.label_encoder = cache["label_encoder"]
            self.class_names = self.label_encoder.classes_

        # 2. 加载模型架构与权重
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ 未找到模型权重 {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = TransportationModeClassifier(**ckpt["model_config"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"✅ 模型加载成功！支持识别的类别: {list(self.class_names)}")

    def predict(self, trajectory_features):
        """
        输入:
            trajectory_features: np.ndarray, 形状为 (seq_len, 9) 或 (batch, seq_len, 9)
        输出:
            预测标签字符串, 置信度分数
        """
        # 转换数据维度
        if isinstance(trajectory_features, list):
            trajectory_features = np.array(trajectory_features)

        # 如果是单条数据 (seq_len, 9)，增加 batch 维度 -> (1, seq_len, 9)
        if len(trajectory_features.shape) == 2:
            trajectory_features = np.expand_dims(trajectory_features, axis=0)

        x = torch.FloatTensor(trajectory_features).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)

        # 映射回标签名称
        pred_labels = self.label_encoder.inverse_transform(preds.cpu().numpy())
        confidences = conf.cpu().numpy()

        return pred_labels, confidences


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    predictor = TrajectoryPredictor()

    # 模拟输入一条 9 维特征的轨迹段 (假设长度为 50)
    # 在实际应用中，这里应该是你从 .plt 文件提取并经过 preprocess_segments 处理后的特征
    dummy_input = np.random.randn(50, 9)

    label, score = predictor.predict(dummy_input)

    print("-" * 30)
    print(f"预测结果: {label[0]}")
    print(f"置信度: {score[0]:.4f}")