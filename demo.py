import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

categories = ['Walk', 'Bike', 'Bus', 'Car & taxi', 'Train']
exp1_f1 = [0.84, 0.69, 0.62, 0.62, 0.55]
exp2_f1 = [0.89, 0.81, 0.79, 0.77, 0.85]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x - width/2, exp1_f1, width, label='Exp1', color='#1f77b4')
ax.bar(x + width/2, exp2_f1, width, label='Exp2', color='#ff7f0e')

ax.set_ylabel('F1-score')
ax.set_ylim(0, 1)
ax.set_title('Exp1 vs Exp2 各类别 F1-score 对比')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()
