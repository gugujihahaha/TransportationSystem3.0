import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import pickle
from torch.utils.data import DataLoader, TensorDataset
# 导入你项目中的模块
from MainModel import MASO_MSF
from PrintMetricInformation import getPRF1

# --- 配置参数 (必须与训练时一致) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBJECT_K = 5  # 轨迹分段数
NUM_CLASSES = 5  # 交通方式类别数
NUM_CHANNEL = 9  # 像素特征维度 (速度, 加速度, 转角等)
VALUE_RATIO = 16  # 注意力机制参数

# 路径设置
DATA_PATH = os.path.join('..', 'data', 'processed', 'final_features.pickle')
MODEL_PATH = "maso_msf_final.pth"


# --- 核心逻辑函数 ---

def confusionMatrix(matrix, preds, labels):
    for p, q in zip(labels, preds):
        matrix[int(p), int(q)] += 1
    return matrix


def getFullMaxIndex(probData):
    return np.argmax(probData)


def getFinalPred(lst, prob):
    """
    决策融合逻辑：
    从 5 个阶段的预测结果中，通过投票和概率加权选出最终类别
    """
    countFreq = {}
    for key in lst:
        countFreq[key] = countFreq.get(key, 0) + 1
    highest = max(countFreq.values())

    a = [k for k, v in countFreq.items() if v == highest]
    if len(a) == 1:
        # 存在绝对多数票
        truelabel = max(lst, default='None', key=lambda v: lst.count(v))
    elif len(a) == OBJECT_K:
        # 全都不一样，取概率最大的
        truelabel = lst[getFullMaxIndex(prob)]
    else:
        # 票数相同，计算平均概率
        sumpro = {}
        for p in range(len(a)):
            b = [v for k, v in enumerate(prob) if lst[k] == a[p]]
            sumpro[a[p]] = np.mean(b)
        truelabel = max(sumpro, key=sumpro.get)
    return truelabel


def ObjectDecisionFusion(data):
    """
    将模型输出的 [Batch*5, 5] 转化为 [Batch] 的最终预测
    """
    predictlabel = data.argmax(dim=1).tolist()
    probability = torch.max(data, 1)[0].tolist()
    k = 0
    maxLabel = []
    while k < len(predictlabel):
        tempLabel = predictlabel[k:k + OBJECT_K]
        tempProb = probability[k:k + OBJECT_K]
        maxLabel.append(getFinalPred(tempLabel, tempProb))
        k += OBJECT_K
    return torch.tensor(maxLabel).long()


# --- 主程序入口 ---

if __name__ == "__main__":
    print(f"--- 正在使用 {DEVICE} 进行模型评估 ---")

    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}，请确认训练已完成。")
        exit()

    net = MASO_MSF(NUM_CHANNEL, NUM_CLASSES, VALUE_RATIO).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()
    print("模型加载成功！")

    # 2. 加载数据
    if not os.path.exists(DATA_PATH):
        print(f"错误：找不到数据文件 {DATA_PATH}")
        exit()

    with open(DATA_PATH, 'rb') as f:
        X, y = pickle.load(f)

    # 转换为 Tensor
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    # 这里我们直接对全量数据进行评估（你也可以像 Train 那样 split 出 20%）
    test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=False)

    # 3. 开始测试
    total, correct = 0, 0
    confMatrixs = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    print("开始计算各项指标...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            # 适配模型的多尺度输入接口
            output = net(X_batch, X_batch, X_batch, X_batch)

            # 决策融合 (5段合1)
            maxLabel = ObjectDecisionFusion(output)

            # 统计
            y_cpu = y_batch.cpu()
            correct += (maxLabel == y_cpu).sum().item()
            total += y_batch.size(0)

            # 更新混淆矩阵
            confMatrixs = confusionMatrix(confMatrixs, maxLabel, y_cpu)

    # 4. 打印结果
    print("\n" + "=" * 30)
    print(f"最终测试准确率 (Accuracy): {100.0 * correct / total:.2f}%")
    print("=" * 30)

    # 调用评估指标工具
    # 会输出 Precision, Recall, F1-Score
    getPRF1(confMatrixs, NUM_CLASSES)