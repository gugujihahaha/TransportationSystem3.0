import numpy as np
import os
from datetime import datetime
import pickle

# --- 路径修复 ---
# 确保路径指向你截图中的位置
pathFile = os.path.join('..', 'data', 'Geolife Trajectories 1.3', 'Data')
savePath = os.path.join('..', 'data', 'processed')

if not os.path.exists(savePath):
    os.makedirs(savePath)

# 过滤掉非用户文件夹（比如 User Guide.pdf）
allFolder = [f for f in os.listdir(pathFile) if os.path.isdir(os.path.join(pathFile, f))]
allFolder.sort()

print(f"找到用户文件夹: {allFolder}")

trajectoryLabelAllUser = []
mode = ['walk', 'bike', 'bus', 'car', 'taxi', 'subway', 'train', 'railway']
modeIndex = {'walk': 0, 'bike': 1, 'bus': 2, 'car': 3, 'taxi': 3, 'subway': 4, 'train': 4, 'railway': 4}


def dateConvert(time_str):
    # 兼容两种常见的日期格式
    try:
        date_format = "%Y/%m/%d %H:%M:%S"
        current = datetime.strptime(time_str, date_format)
    except:
        date_format = "%Y-%m-%d %H:%M:%S"
        current = datetime.strptime(time_str, date_format)

    bench = datetime.strptime('1899/12/30', "%Y/%m/%d")
    no_days = current - bench
    return no_days.days + current.hour / 24.0 + current.minute / (24. * 60.) + current.second / (24. * 3600.)


for folder in allFolder:
    user_path = os.path.join(pathFile, folder)
    # 只有同时拥有轨迹和标签的用户才处理
    if len(os.listdir(user_path)) >= 2:
        print(f"正在处理用户: {folder}")
        trajectoryOneUser = []
        trajectoriesPath = os.path.join(user_path, 'Trajectory')
        allPlt = [f for f in os.listdir(trajectoriesPath) if f.endswith('.plt')]
        allPlt.sort()

        for plt in allPlt:
            with open(os.path.join(trajectoriesPath, plt), 'r', encoding="utf-8") as f:
                lines = f.readlines()[6:]  # 跳过plt文件的前6行车头
                for line in lines:
                    row = line.rstrip().split(',')
                    if len(row) == 7:
                        trajectoryOneUser.append([float(row[0]), float(row[1]), float(row[4])])

        if not trajectoryOneUser: continue
        trajectoryOneUser = np.array(trajectoryOneUser)

        labelPath = os.path.join(user_path, 'labels.txt')
        if not os.path.exists(labelPath): continue

        labelOneUser = []
        with open(labelPath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # 跳过标题行
            for line in lines:
                row = line.rstrip().split('\t')
                if len(row) == 3 and row[2] in mode:
                    labelOneUser.append([dateConvert(row[0]), dateConvert(row[1]), modeIndex[row[2]]])

        if not labelOneUser: continue
        labelOneUser = np.array(labelOneUser)

        # 匹配逻辑
        Dates = trajectoryOneUser[:, 2]
        sec = 1 / (24.0 * 3600.0)
        indexNeed = []
        modeFinal = []
        for row in labelOneUser:
            idx = np.where((Dates >= (row[0] - sec)) & (Dates <= (row[1] + sec)))[0]
            for i in idx:
                indexNeed.append(i)
                modeFinal.append(row[2])

        if indexNeed:
            matched_traj = trajectoryOneUser[indexNeed]
            modeFinal = np.array(modeFinal).reshape(-1, 1)
            trajectoryLabelOneUser = np.hstack((matched_traj, modeFinal))
            trajectoryLabelAllUser.append(trajectoryLabelOneUser)

# 保存第一步结果
with open(os.path.join(savePath, 'step1_matched.pickle'), 'wb') as f:
    pickle.dump(trajectoryLabelAllUser, f)
print("第一步完成，数据已存至 data/processed/step1_matched.pickle")