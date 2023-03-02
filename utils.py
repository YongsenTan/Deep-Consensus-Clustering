import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
from collections import defaultdict
from sklearn.cluster import KMeans


# customize dataset
class AKIData(Dataset):
    def __init__(self, path):
        infile = open(path, 'rb')
        data_x, data_y = pickle.load(infile)

        self.C = torch.tensor([0] * len(data_x)).float()
        self.data_x = torch.tensor(data_x).float()
        self.data_y = torch.tensor(data_y).float()
        self.rep = None
        self.outcome_ratio_dict = None

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx, :], self.data_y[idx].float()


# get data representation
def get_embedding(model, dataloader_train):
    embedding = []
    model.eval()
    for batch_idx, (data_x, target) in enumerate(dataloader_train):
        data_x = data_x.to(model.device)
        enc, _ = model.get_rep(data_x)
        embed = enc.data.cpu()[:, 0, :]
        embedding.append(embed)
    embedding = torch.cat(embedding, dim=0)
    return embedding.numpy()


# get data prediction
def get_outcome(model, dataloader):
    out, y_true = [], []
    model.eval()
    for batch_idx, (data_x, target) in enumerate(dataloader):
        data_x = data_x.to(model.device)
        _, _, y = model(data_x)
        out.append(y.data.cpu())
        y_true.append(target)
    out, y_true = torch.cat(out, dim=0), torch.cat(y_true, dim=0)
    return out.numpy(), y_true.numpy()


# evaluate on validation set
def evaluate(model, dataloader_test):
    model.eval()
    error_ae = []
    error_outcome_likelihood = []
    outcome_true_y = []
    outcome_prob = []
    for batch_idx, (data_x, target) in enumerate(dataloader_test):
        data_x = data_x.to(model.device)
        target = target.to(model.device)

        _, decoded_x, output_outcome = model(data_x)

        loss_ae = model.MSELoss(data_x, decoded_x)
        loss_outcome = model.BCELoss(output_outcome, target.float())

        error_ae.append(loss_ae.data.cpu().numpy())
        error_outcome_likelihood.append(loss_outcome.data.cpu().numpy())

        outcome_true_y.append(target.data.cpu())
        outcome_prob.append(output_outcome.data.cpu())

    test_ae_loss = np.mean(error_ae)
    test_outcome_loss = np.mean(error_outcome_likelihood)
    outcome_auc_score = roc_auc_score(np.concatenate(outcome_true_y, 0), np.concatenate(outcome_prob, 0))

    return test_ae_loss, test_outcome_loss, outcome_auc_score


def correspond(unsorted_c, y):
    dict_c_count = defaultdict(int)
    dict_outcome_in_c_count = defaultdict(int)

    for i in range(len(unsorted_c)):
        dict_c_count[unsorted_c[i]] += 1
        if y[i] == 1:
            dict_outcome_in_c_count[unsorted_c[i]] += 1

    dict_outcome_ratio = {}
    for c in dict_c_count:
        dict_outcome_ratio[c] = dict_outcome_in_c_count[c] / dict_c_count[c]

    sorted_dict_outcome_ratio = dict(sorted(dict_outcome_ratio.items(), key=lambda x: x[1]))
    order = list(sorted_dict_outcome_ratio.keys())
    order_c_map = {}
    for i in range(len(order)):
        order_c_map[order[i]] = i

    sorted_c = []
    for i in range(len(unsorted_c)):
        sorted_c.append(order_c_map[unsorted_c[i]])
    sorted_c = np.array(sorted_c)
    return sorted_c


def worker_m(clusters, i):
    template = np.array([clusters[:, i] for _ in range(clusters.shape[1])]).T
    consensus = np.array(template == clusters, dtype=int)
    return np.sum(consensus, axis=0)


def worker_kmeans(k, i):
    r = pickle.load(open(f'./representations/rep_{i}.pkl', 'rb'))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(r)
    return kmeans.labels_
