import pandas as pd
import json
import random
import os


def generate_carbon_data():
    # 1. 路径配置
    csv_path = r'D:\TransportationSystem3.0\FrontEnd\public\predictions_with_geo_exp1.csv'
    grid_path = r'D:\TransportationSystem3.0\FrontEnd\public\heatmap_grid.json'
    output_path = r'D:\TransportationSystem3.0\FrontEnd\public\carbon_heatmap.json'

    if not os.path.exists(csv_path) or not os.path.exists(grid_path):
        print("错误：文件缺失，请检查路径。")
        return

    # 2. 从 CSV 中提取模型识别的“绿色基因”
    df = pd.read_csv(csv_path)
    # 定义哪些模式算“绿色”
    green_modes = ['Walk', 'Bike', 'Bus', 'Subway']
    is_green = df['pred_label'].isin(green_modes)

    # 计算全局平均指标
    base_green_ratio = is_green.mean()  # 绿色出行占比
    avg_confidence = df['confidence'].mean()
    print(f"检测到模型平均绿色占比: {base_green_ratio:.2%}")

    # 3. 读取原始网格坐标并注入数据
    with open(grid_path, 'r') as f:
        grids = json.load(f)

    final_heatmap = []
    for point in grids:
        simulated_count = int(point['value'] * 1000) + random.randint(1, 10)
        grid_green_ratio = base_green_ratio * (0.9 + random.random() * 0.2)
        grid_green_ratio = min(max(grid_green_ratio, 0.1), 0.95)  # 约束在10%-95%

        # 4. 生成前端 Vue 模板所需的特定字段
        final_heatmap.append({
            "lat": point['lat'],
            "lng": point['lng'],
            "count": simulated_count,
            "green_ratio": grid_green_ratio
        })

    # 5. 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_heatmap, f)

    print(f"成功！已生成对齐前端格式的 {output_path}")


if __name__ == "__main__":
    generate_carbon_data()