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
        # 初始化第一个点的选择概率，此时概率均匀
        pr = np.repeat(1 / n, n)
        initial_state[0, :] = data[np.random.choice(np.arange(n), p=pr)]
        # 计算所有点离第一个centers的距离
        dis[:, 0] = torch.sum((data - initial_state[0, :]) ** 2, dim=1)

        for k in range(1, self.n_clusters):
            pr = torch.sum(dis, dim=1) / torch.sum(dis)
            initial_state[k, :] = data[np.random.choice(np.arange(n), 1, p=pr.to('cpu').numpy())]
            for center_idx in range(k+1):
                dis[:, center_idx] = torch.sum((data - initial_state[center_idx, :]) ** 2, dim=1)


        return initial_state
