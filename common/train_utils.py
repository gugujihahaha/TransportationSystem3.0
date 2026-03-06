"""
通用训练工具函数。

提供支持任意数量输入模态的 train_epoch 和 evaluate 函数，
避免在 exp2/3/4/5 的 train.py 中重复实现相同逻辑。

使用约定：
    DataLoader 的每个 batch 必须是 (*features, labels) 的元组，
    其中 features 是若干特征张量，labels 是标签张量。
"""
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import Tuple, List


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """
    执行一个 epoch 的训练。

    参数：
        model:          PyTorch 模型（需支持 *inputs 调用方式）
        dataloader:     训练数据加载器，batch 格式为 (*features, labels)
        criterion:      损失函数
        optimizer:      优化器
        device:         训练设备（'cuda' 或 'cpu'）
        max_grad_norm:  梯度裁剪阈值，默认 1.0

    返回：
        (avg_loss, accuracy): 本 epoch 的平均损失和准确率
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        *features, labels = batch
        features = [f.to(device) for f in features]
        labels   = labels.to(device)

        optimizer.zero_grad()
        logits = model(*features)
        loss   = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct   / max(total, 1)
    return avg_loss, accuracy


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    label_names: List[str],
) -> Tuple[float, dict, List, List]:
    """
    在验证集或测试集上评估模型。

    参数：
        model:        PyTorch 模型
        dataloader:   评估数据加载器，batch 格式为 (*features, labels)
        criterion:    损失函数
        device:       评估设备
        label_names:  类别名称列表（如 label_encoder.classes_）

    返回：
        (avg_loss, report_dict, all_preds, all_labels)
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            *features, labels = batch
            features = [f.to(device) for f in features]
            labels   = labels.to(device)

            logits      = model(*features)
            total_loss += criterion(logits, labels).item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    report   = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return avg_loss, report, all_preds, all_labels
