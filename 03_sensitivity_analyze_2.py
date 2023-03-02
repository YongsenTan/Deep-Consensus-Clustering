from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy import stats
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

num_clusters = 7
h_min, h_max = 3, 13
sample_size = 0.8
bootstrapping = 30

y = pickle.load(open('./dataset/data.pkl', 'rb'))[1]
y = np.array(y)
sub = pickle.load(open(f'./results/consensus_cluster_{num_clusters}.pkl', 'rb'))

heat = np.array([[0 for _ in range(num_clusters)] for _ in range(num_clusters)], dtype=np.float64)
ari, nmi = [], []

func_kmeans = partial(worker_kmeans, num_clusters)
with Pool(multiprocessing.cpu_count()) as p:
    clusters = list(p.map(func_kmeans, [i for i in range(h_min, h_max)]))

clusters = np.array(clusters)

for i in tqdm(range(bootstrapping), desc='Processing'):
    boot = resample(clusters, replace=False, n_samples=round(len(clusters) * sample_size), random_state=i)

    func_m = partial(worker_m, boot)
    with Pool(multiprocessing.cpu_count()) as p:
        m = list(p.map(func_m, [i for i in range(boot.shape[1])]))

    m = np.array(m, dtype=np.float32) / (round(len(clusters) * sample_size))
    model = KMeans(n_clusters=num_clusters, random_state=1, n_jobs=-1)
    model.fit(m)
    res = np.array(model.labels_)
    res = correspond(res, y)
    heat += confusion_matrix(res, sub) / len(sub)

    ari.append(adjusted_rand_score(sub, res))
    nmi.append(normalized_mutual_info_score(sub, res))

heat /= bootstrapping
f = open(f'./results/heat2.pkl', 'wb')
pickle.dump(heat, f)
f.close()

sns.heatmap(heat, xticklabels=[i + 1 for i in range(num_clusters)], yticklabels=[i + 1 for i in range(num_clusters)])
plt.xlabel('Subphenotypes of Bootstrapping')
plt.ylabel('Subphenotypes')
plt.savefig('sensitivity2.png', dpi=300)
plt.show()

lo, hi = stats.norm.interval(0.95, np.mean(ari), np.std(ari))
print(f'ARI: %.3f (%.3f-%.3f)' % (sum(ari) / len(ari), lo, hi))

lo, hi = stats.norm.interval(0.95, np.mean(nmi), np.std(nmi))
print(f'NMI: %.3f (%.3f-%.3f)' % (sum(nmi) / len(nmi), lo, hi))
