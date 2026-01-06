import os
import numpy as np
import pickle

# --- 路径设置 ---
loadPath = os.path.join('..', 'data', 'processed', 'step1_matched.pickle')
savePath = os.path.join('..', 'data', 'processed', 'step2_split.pickle')

if not os.path.exists(loadPath):
    print("错误：找不到第一步生成的 pickle 文件！")
else:
    with open(loadPath, 'rb') as f:
        trajectoryLabelAllUser = pickle.load(f)

    AllSegment = []
    AllModes = []
    minPoints = 20
    min_trip_time = 20 * 60  # 20分钟间隔断开轨迹

    for Data in trajectoryLabelAllUser:
        if len(Data) < minPoints: continue

        # 计算时间差
        times = Data[:, 2] * 24 * 3600
        delta_time = np.diff(times)

        # 寻找切分点
        split_idx = np.where(delta_time > min_trip_time)[0] + 1
        trips = np.split(Data, split_idx)

        for trip in trips:
            if len(trip) < minPoints: continue

            # 按交通模式(Mode)再次切分
            modes = trip[:, 3]
            mode_split_idx = np.where(modes[:-1] != modes[1:])[0] + 1
            segments = np.split(trip, mode_split_idx)

            for seg in segments:
                if len(seg) >= minPoints:
                    AllSegment.append(seg)
                    AllModes.append(int(seg[0, 3]))

    with open(savePath, 'wb') as f:
        pickle.dump([AllSegment, AllModes], f)
    print(f"第二步完成，切分出 {len(AllSegment)} 条有效线段。")