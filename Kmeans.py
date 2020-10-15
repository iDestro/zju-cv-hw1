import torch
import numpy as np


class KMeans:
    def __init__(self, n_clusters, device, tol=1e-4):
        self.n_clusters = n_clusters
        self.device = device
        self.tol = tol
        self._labels = None
        self._cluster_centers = None

    def _initial_state(self, data):
        n, c = data.shape
        dis = torch.zeros((n, self.n_clusters), device=self.device)
        initial_state = torch.zeros((self.n_clusters, c), device=self.device)
        idx = np.random.randint(0, n)
        initial_state[0, :] = data[idx]

        for k in range(1, self.n_clusters):
            for center_idx in range(self.n_clusters):
                dis[:, center_idx] = torch.sum((data - initial_state[center_idx, :]) ** 2, dim=1)
            min_dist, _ = torch.min(dis, dim=1)
            p = min_dist / torch.sum(min_dist)
            initial_state[k, :] = data[np.random.choice(np.arange(n), 1, p=p.to('cpu').numpy())]

        return initial_state

    @staticmethod
    def pairwise_distance(x, y):
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=0)
        dis = (x-y)**2
        dis = dis.sum(dim=-1).squeeze()
        return dis

    def fit(self, data):
        print("yes")
        data = data.to(self.device)
        cluster_centers = self._initial_state(data)
        dis = torch.zeros((len(data), self.n_clusters))
        while True:
            print("no")
            for i in range(self.n_clusters):
                dis[:, i] = torch.norm(data - cluster_centers[i], dim=1)
            labels = torch.argmin(dis, dim=1)
            cluster_centers_pre = cluster_centers.clone()
            for i in range(self.n_clusters):
                cluster_centers[i, :] = torch.mean(data[labels == i], dim=0)
            center_shift = torch.sum(torch.sqrt(torch.sum((cluster_centers - cluster_centers_pre) ** 2, dim=1)))
            print(center_shift)
            if center_shift ** 2 < self.tol:
                break

        self._cluster_centers = cluster_centers
        self._labels = labels

    def predict(self, x):
        dis = torch.zeros([x.size(0), self.n_clusters])

        for i in range(self.n_clusters):
            dis[:, i] = torch.sum((x-self._cluster_centers[i, :])**2, dim=1)

        pred = torch.argmin(dis, dim=1)
        return pred

    def silhouette_score(self, x, labels):
        x = x.to(self.device)
        labels = labels.to(self.device)
        length = len(x)
        indexes = torch.arange(length)
        sampled_indexes = np.random.choice(indexes, 10000)
        ones_vector = torch.ones(length).to(self.device)
        total = torch.Tensor([0]).to(self.device)
        for index, i in enumerate(x[sampled_indexes]):
            if index % 100 == 0:
                print(index)
            matched = labels == labels[index]
            dis = torch.sqrt(torch.sum((x-i)**2, dim=1))
            cnt = torch.sum(ones_vector[matched])
            a = torch.sum(dis[matched]) / cnt
            matched = labels != labels[index]
            cnt = torch.sum(ones_vector[matched])
            b = torch.sum(dis[matched]) / cnt
            total += (b-a) / (torch.max(a, b))
        return total / 10000

    @property
    def labels_(self):
        return self._labels

    @property
    def cluster_centers_(self):
        return self._cluster_centers
