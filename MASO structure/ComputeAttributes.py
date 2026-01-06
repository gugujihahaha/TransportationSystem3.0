import os
import numpy as np
import pickle
from geopy.distance import geodesic  # 替换已过时的 vincenty
import math

# --- 路径设置 ---
loadPath = os.path.join('..', 'data', 'processed', 'step3_image.pickle')
savePath = os.path.join('..', 'data', 'processed', 'final_features.pickle')

# 速度限制 (单位: m/s)
SpeedLimit = {0: 7, 1: 12, 2: 33.3, 3: 50.0, 4: 33.3}


def calAttr(pts, label):
    if len(pts) < 2: return [0] * 9
    speeds, angles = [], []
    for m in range(len(pts) - 1):
        dist = geodesic((pts[m][0], pts[m][1]), (pts[m + 1][0], pts[m + 1][1])).meters
        dt = (pts[m + 1][2] - pts[m][2]) * 86400 + 0.1
        v = dist / dt
        if 0 <= v <= SpeedLimit.get(label, 50):
            speeds.append(v)
            # 计算简单的方位角
            angle = math.degrees(math.atan2(pts[m + 1][1] - pts[m][1], pts[m + 1][0] - pts[m][0]))
            angles.append(angle)

    if not speeds: return [0] * 9
    accel = np.abs(np.diff(speeds)) if len(speeds) > 1 else [0]
    angle_diff = np.abs(np.diff(angles)) if len(angles) > 1 else [0]

    res = [np.mean(speeds), np.max(speeds), np.min(speeds),
           np.mean(accel), np.max(accel), np.min(accel),
           np.mean(angle_diff), np.max(angle_diff), np.min(angle_diff)]
    return res


if __name__ == "__main__":
    with open(loadPath, 'rb') as f:
        allSegmentImage, allSegment, allModes = pickle.load(f)

    final_input = []
    print("开始计算 9 维像素特征...")
    for i in range(len(allSegmentImage)):
        sample_features = []
        for stage in range(5):  # 5个阶段
            grid = allSegmentImage[i][stage]
            feature_map = np.zeros((9, 32, 32))
            for r in range(32):
                for c in range(32):
                    if grid[r][c]:
                        attrs = calAttr(grid[r][c], allModes[i])
                        feature_map[:, r, c] = attrs
            sample_features.append(feature_map)
        final_input.append(sample_features)
        if i % 50 == 0: print(f"进度: {i}/{len(allSegmentImage)}")

    with open(savePath, 'wb') as f:
        pickle.dump([np.array(final_input), np.array(allModes)], f)
    print(f"预处理全部完成！最终特征维度: {np.array(final_input).shape}")