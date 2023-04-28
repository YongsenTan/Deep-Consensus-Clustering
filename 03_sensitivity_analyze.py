import numpy as np
import pickle
from utils import correspond
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy import stats
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

k = 7
n_bootstrapping = 1000
bootstrapping_proportion = 0.7

y = pickle.load(open('./dataset/data.pkl', 'rb'))[1]
y = np.array(y)
sub = pickle.load(open(f'./results/cluster_{k}.pkl', 'rb'))
rep = pickle.load(open(f'./results/m_{k}.pkl', 'rb'))

ids = np.array([i for i in range(len(sub))])

heat = np.array([[0 for _ in range(k)] for _ in range(k)], dtype=np.float64)
ari, nmi = [], []

agg = np.array(rep)


for i in tqdm(range(n_bootstrapping), desc='Processing'):
    boot = resample(ids, replace=False, n_samples=round(len(ids) * bootstrapping_proportion), random_state=i)
    s = sub[boot]
    yi = y[boot]

    m = agg[:, boot][boot, :]

    model = KMeans(n_clusters=k, random_state=1, n_jobs=-1)
    model.fit(m)
    res = np.array(model.labels_)
    res = correspond(res, yi)
    heat += confusion_matrix(res, s) / len(s)

    ari.append(adjusted_rand_score(s, res))
    nmi.append(normalized_mutual_info_score(s, res))

heat /= n_bootstrapping
sns.heatmap(heat, xticklabels=[i + 1 for i in range(k)], yticklabels=[i + 1 for i in range(k)])
plt.xlabel('Subphenotypes of Patient Bootstrapping', fontsize=15)
plt.ylabel('Derived Subphenotypes', fontsize=15)
plt.savefig('sensitivity.png', dpi=300)

lo, hi = stats.norm.interval(0.95, np.mean(ari), np.std(ari))
print(f'ARI: %.3f (%.3f-%.3f)' % (np.mean(ari), lo, hi))

lo, hi = stats.norm.interval(0.95, np.mean(nmi), np.std(nmi))
print(f'NMI: %.3f (%.3f-%.3f)' % (np.mean(nmi), lo, hi))
