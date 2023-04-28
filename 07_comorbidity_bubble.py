import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend


def get_num_proportion(data):
    data = np.array(data)
    n, p = [], []
    for line in data:
        n_term, p_term = [], []
        for item in line:
            for idx in range(len(item)):
                if item[idx] == '(':
                    n_term.append(float(item[:idx]))
                    p_term.append(float(item[idx+1:-1]))
        n.append(n_term)
        p.append(p_term)
    return np.array(n), np.array(p)


com = pd.read_csv('./sample/com_iv.csv')
num, proportion = get_num_proportion(com.iloc[:, 1:])

colors = ['#3951a2', "#5c90c2", "#92c5de", "#fdb96b", "#f67948", "#da382a", "#a80326"]
fig, ax = plt.subplots()
for i in range(proportion.shape[0]):
    for j in range(proportion.shape[1]):
        ax.scatter(proportion[i, j], i, c=colors[j], s=num[i][j] / 5, alpha=0.5)


for i in range(7):
    ax.scatter([], [], c=colors[i], s=15, label=f'S{i + 1}')
plt.legend(ncol=4, labelspacing=0.05, columnspacing=0.1)

a = []
for area in [10, 100, 1000]:
    a.append(ax.scatter([], [], c='k', alpha=0.3, s=area / 5, label=str(area)))
leg = Legend(ax, a, ['10', '100', '1000'],
             loc='lower right', ncol=3, labelspacing=0.05, columnspacing=0.5)
ax.add_artist(leg)

plt.xlabel('Proportion (%)')
plt.yticks([i for i in range(com.shape[0])], com.iloc[:, 0])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False, right=False)
plt.savefig('bubble_iv', bbox_inches='tight', dpi=300)
