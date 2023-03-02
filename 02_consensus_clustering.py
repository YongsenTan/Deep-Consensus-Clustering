import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from utils import *


def main():
    # range of no. clusters
    k_min, k_max = 3, 10
    # range of no. hidden dimensions
    h_min, h_max = 3, 13

    # no. workers for multiprocessing
    num_workers = multiprocessing.cpu_count()

    # mortality labels, for sorting the clusters
    data_file = './dataset/data.pkl'
    data = pickle.load(open(data_file, 'rb'))
    y = np.array(data[1])

    cdf = []
    areas = []
    consensus_bars = []

    plt.figure()
    for k in range(k_min, k_max):
        # generate clusters from multiple dimensions
        func_kmeans = partial(worker_kmeans, k)
        with Pool(num_workers) as p:
            clusters = list(tqdm(p.map(func_kmeans, [i for i in range(h_min, h_max)]), total=k_max - k_min))
        # [no. models, no. patients]
        clusters = np.array(clusters)

        # calculate the consensus matrix
        func_m = partial(worker_m, clusters)
        with Pool(num_workers) as p:
            m = list(tqdm(p.map(func_m, [i for i in range(clusters.shape[1])]), total=clusters.shape[1]))
        # [no. patients, no. patients]
        m = np.array(m, dtype=np.float32) / (k_max - k_min)

        f = open(f'./results/m_{k}.pkl', 'wb')
        pickle.dump(m, f)
        f.close()

        model = KMeans(n_clusters=k, random_state=42, n_jobs=-1)
        model.fit(m)
        res = np.array(model.labels_)
        # sort derived subphenotypes by mortality
        res = correspond(res, y)

        s = sorted(res)
        f = open(f'./results/consensus_cluster_{k}.pkl', 'wb')
        pickle.dump(res, f)
        f.close()

        c_num = []
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                c_num.append(i)
        c_num.append(len(s))

        # sort the consensus matrix according to the clusters
        # both in x-axis and y-axis
        m = m[:, res.argsort()]
        m = m[res.argsort(), :]
        #
        # # visualization of the consensus matrix
        # plt.figure()
        # color = 'b'
        # # draw the boundaries of clusters
        #
        # plt.vlines(1, ymin=0, ymax=c_num[0], color=color, linewidths=3)
        # plt.hlines(1, xmin=0, xmax=c_num[0], color=color, linewidths=3)
        # plt.vlines(len(s) - 1, ymin=c_num[-2], ymax=len(s) - 1, color=color, linewidths=3)
        # plt.hlines(len(s) - 1, xmin=c_num[-2], xmax=len(s) - 1, color=color, linewidths=3)
        # for i in range(len(c_num) - 1):
        #     if i == 0:
        #         plt.vlines(c_num[i] - 1, 0, c_num[i + 1] - 1, color=color)
        #         plt.hlines(c_num[i] - 1, 0, c_num[i + 1] - 1, color=color)
        #     else:
        #         plt.vlines(c_num[i] - 1, c_num[i - 1] - 1, c_num[i + 1] - 1, color=color)
        #         plt.hlines(c_num[i] - 1, c_num[i - 1] - 1, c_num[i + 1] - 1, color=color)
        # sns.heatmap(m)
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlabel('Patients')
        # plt.ylabel('Patients')
        # plt.savefig(f'heap_{k}.png', dpi=300)

        # calculate consensus value
        b = []
        for i in range(len(c_num)):
            if i == 0:
                b.append(m[:c_num[0], :c_num[0]].sum() / (c_num[0] * c_num[0]))
            else:
                b.append(m[c_num[i - 1]:c_num[i], c_num[i - 1]:c_num[i]].sum() / ((-c_num[i - 1] + c_num[i]) ** 2))
        consensus_bars.append(b)

        # calculate cdf and area under cdf
        consensus_value = m.ravel()
        hist, bin_edges = np.histogram(consensus_value)
        hist[-1] -= m.shape[0]

        c = np.cumsum(hist / sum(hist))
        cdf.append(c)

        width = (bin_edges[1] - bin_edges[0])
        plt.plot(bin_edges[1:] - width / 2, cdf[k - k_min], label=f'k={k}')

        delta_a = [h * (b - a) for b, a, h in zip(bin_edges[1:], bin_edges[:-1], cdf)]
        a = np.sum(delta_a)
        areas.append(a)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel('Consensus Index')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function of Consensus Values')
    plt.savefig(f'cdf.png', dpi=300)

    delta_k = []
    for i in range(k_min, k_max):
        if i == k_min:
            delta_k.append(areas[0])
        else:
            delta_k.append((areas[i - k_min] - areas[i - k_min - 1]) / areas[i - k_min])

    plt.figure()
    plt.plot([i for i in range(k_min, k_max)], delta_k)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Relative Change')
    plt.title('Relative Change in Area Under CDF Curve')
    plt.xticks([i for i in range(k_min, k_max)])
    plt.savefig(f'delta.png', dpi=300)

    plt.figure()
    index = 0
    i_s = []
    for i in range(k_min, k_max):
        b = consensus_bars[i - k_min]
        x = [j for j in range(index, index + i)]
        plt.bar(x, b, label=f'k={i}')
        i_s.append(index + i * 0.35)
        index += i + 2

    plt.xticks(ticks=i_s, labels=[f"k={i}" for i in range(k_min, k_max)])
    plt.xlabel('Subphenotypes of Different k')
    plt.ylabel('Average Consensus Value')
    plt.title('Average Consensus Value of Subphenotypes')
    plt.savefig(f'aver_con.png', dpi=300)


if __name__ == '__main__':
    main()
