"""
Exp1 评估脚本（最终版）
✔ 仅使用训练阶段保存的特征缓存
✔ 与训练、论文结果严格一致
"""

import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import TransportationModeClassifier
from train import TrajectoryDataset


def plot_confusion_matrix(y_true, y_pred, classes, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("evaluation_results", exist_ok=True)

    # ========================================================
    # 加载缓存
    # ========================================================
    with open("cache/exp1_processed_features.pkl", "rb") as f:
        cache = pickle.load(f)

    segments = cache["segments"]
    label_encoder = cache["label_encoder"]

    dataset = TrajectoryDataset(segments, label_encoder)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # ========================================================
    # 加载模型
    # ========================================================
    ckpt = torch.load("checkpoints/exp1_model.pth", map_location=device, weights_only=False)
    config = ckpt["model_config"]

    model = TransportationModeClassifier(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ========================================================
    # 推理
    # ========================================================
    all_preds, all_labels, all_conf = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.squeeze().numpy())
            all_conf.extend(probs.max(1).values.cpu().numpy())

    class_names = label_encoder.classes_

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        zero_division=0,
        output_dict=True
    )

    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        zero_division=0
    ))

    # ========================================================
    # 保存结果
    # ========================================================
    with open("evaluation_results/exp1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        "evaluation_results/exp1_confusion_matrix.png"
    )

    pd.DataFrame({
        "true_label": [class_names[i] for i in all_labels],
        "pred_label": [class_names[i] for i in all_preds],
        "confidence": all_conf
    }).to_csv(
        "evaluation_results/exp1_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("✓ Exp1 评估完成，结果已保存")


if __name__ == "__main__":
    main()
