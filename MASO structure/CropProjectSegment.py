import os
import numpy as np
import pickle
import math
import sys

# --- 路径设置 ---
# 建议使用绝对路径或确保当前工作目录在 MASO structure 文件夹内
loadPath = os.path.join('..', 'data', 'processed', 'step2_split.pickle')
savePath = os.path.join('..', 'data', 'processed', 'step3_image.pickle')

Wp = Hp = 0.2  # 空间范围
Wm = Hm = 32  # 图像尺寸 (32x32)
parts = 5  # 阶段数


def getSubSeg(data):
    length = len(data)
    sublength = length // parts
    # 确保索引不越界
    return [min(int((k + 0.5) * sublength), length - 1) for k in range(parts)]


def convert2Img(segData):
    imgPoints = []
    total = len(segData)
    print(f"开始转换 {total} 条线段为图像网格...")

    for i, oneSample in enumerate(segData):
        oneSample = np.array(oneSample)  # 确保是numpy数组方便切片
        imgPoint = []
        centerIndices = getSubSeg(oneSample)

        # 获取当前样本的经纬度范围
        minLat, minLng = np.min(oneSample[:, :2], axis=0)

        for index in centerIndices:
            # 计算当前阶段中心点
            centerLat, centerLng = oneSample[index][0], oneSample[index][1]

            # 计算偏移量，使中心点尽可能位于 32x32 图像中央
            offsetX = (Wm // 2) - math.floor((centerLng - minLng) * Wm / Wp)
            offsetY = (Hm // 2) - math.floor((centerLat - minLat) * Hm / Hp)

            # 初始化网格
            oneSegmentImage = [[[] for _ in range(Wm)] for _ in range(Hm)]

            for k, point in enumerate(oneSample):
                imageX = math.floor((point[1] - minLng) * Wm / Wp) + offsetX
                imageY = math.floor((point[0] - minLat) * Hm / Hp) + offsetY

                if 0 <= imageX < Wm and 0 <= imageY < Hm:
                    # 适配数组索引 (行是Y，列是X)
                    idxX, idxY = Hm - 1 - imageY, imageX
                    # 存入属性：Lat, Lng, Time, Mode, ClusterID, OriginalIndex
                    oneSegmentImage[idxX][idxY].append([
                        float(point[0]), float(point[1]), float(point[2]),
                        int(point[3]), 0, k
                    ])
            imgPoint.append(oneSegmentImage)

        imgPoints.append(imgPoint)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"进度: {i + 1}/{total} 条已处理", end='\r')
            sys.stdout.flush()

    print("\n转换逻辑完成，准备写入硬盘...")
    return imgPoints


if __name__ == "__main__":
    # 检查上一步文件
    if not os.path.exists(loadPath):
        print(f"错误：找不到文件 {loadPath}，请先运行 Step 2")
        exit()

    with open(loadPath, 'rb') as f:
        allSegment, allModes = pickle.load(f)

    # 执行转换
    allSegmentImage = convert2Img(allSegment)

    # 保存文件
    print(f"正在执行 pickle.dump，由于数据结构复杂，这可能需要 3-10 分钟，请勿关闭程序...")
    try:
        with open(savePath, 'wb') as f:
            # 提高 protocol 版本以加速大数据量写入
            pickle.dump([allSegmentImage, allSegment, allModes], f, protocol=4)
        print(f"成功！文件已保存至: {savePath}")
        print("第三步完成：图像网格已生成。")
    except Exception as e:
        print(f"\n保存失败: {e}")