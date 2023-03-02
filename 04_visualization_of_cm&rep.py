import numpy as np
import pickle
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

num_clusters = 7
h_min, h_max = 3, 13
c = ['#3951a2', '#5c90c2', '#92c5de', '#fdb96b', '#f67948', '#da382a', '#a80326']
sub = pickle.load(open(f'./results/consensus_cluster_{num_clusters}.pkl', 'rb'))
sub = np.array(sub)
m = pickle.load(open(f'./results/m_{num_clusters}.pkl', 'rb'))

s = sorted(sub)
c_num = []
for i in range(1, len(s)):
    if s[i] != s[i - 1]:
        c_num.append(i)
c_num.append(len(s))

# sort the consensus matrix according to the clusters
# both in x-axis and y-axis
m = m[:, sub.argsort()]
m = m[sub.argsort(), :]

# visualization of the consensus matrix
plt.figure()
color = 'b'
# draw the boundaries of clusters
plt.vlines(1, ymin=0, ymax=c_num[0], color=color, linewidths=3)
plt.hlines(1, xmin=0, xmax=c_num[0], color=color, linewidths=3)
plt.vlines(len(s) - 1, ymin=c_num[-2], ymax=len(s) - 1, color=color, linewidths=3)
plt.hlines(len(s) - 1, xmin=c_num[-2], xmax=len(s) - 1, color=color, linewidths=3)
for i in range(len(c_num) - 1):
    if i == 0:
        plt.vlines(c_num[i] - 1, 0, c_num[i + 1] - 1, color=color)
        plt.hlines(c_num[i] - 1, 0, c_num[i + 1] - 1, color=color)
    else:
        plt.vlines(c_num[i] - 1, c_num[i - 1] - 1, c_num[i + 1] - 1, color=color)
        plt.hlines(c_num[i] - 1, c_num[i - 1] - 1, c_num[i + 1] - 1, color=color)
sns.heatmap(m)
plt.xticks([])
plt.yticks([])
plt.xlabel('Patients')
plt.ylabel('Patients')
plt.savefig(f'heap_{num_clusters}.png', dpi=300)

# visualization of representations
for i in range(h_min, h_max):
    r = pickle.load(open(f'./representations/rep_{i}.pkl', 'rb'))
    tsne = TSNE(n_jobs=-1, n_components=2)
    r_tsne = tsne.fit_transform(r)
    plt.figure()
    plt.title(f'Number of hidden dimensions={i}')
    for j in range(num_clusters):
        r_tsne_s = r_tsne[sub == j]
        plt.scatter(r_tsne_s[:, 0], r_tsne_s[:, 1], label=f'S{j + 1}', c=c[j], s=0.5)
    plt.legend()
    plt.savefig(f'tsne_{i}.png', dpi=300)
