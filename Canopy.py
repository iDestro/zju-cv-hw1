import torch
import random
import numpy as np


class Canopy:
    def __init__(self, t1, t2, device):
        self.t1 = t1
        self.t2 = t2
        self.device = device
        self._labels = None

    def fit(self, x):
        x = x.to(self.device)
        length = len(x)
        labels = torch.zeros(length).to(self.device)
        unvisited_indexes = torch.ones(length).to(self.device)
        indexes = torch.arange(length)
        while torch.sum(unvisited_indexes) > 0:
            canopy_center = torch.LongTensor(np.random.choice(indexes[unvisited_indexes == 1], 1)).to(self.device)
            labels[canopy_center] = canopy_center.float()
            unvisited_indexes[canopy_center] = 0
            dis = torch.sqrt(torch.sum((x-x[canopy_center])**2, dim=1))
            a = torch.zeros(length).to(self.device)
            b = torch.zeros(length).to(self.device)
            a[dis < self.t1] = 1
            b[unvisited_indexes == 1] = 1
            matched = a * b
            print(torch.sum(matched))
            # print(matched)
            labels[matched == 1] = canopy_center.float()
            a = torch.zeros(length).to(self.device)
            a[dis < self.t2] = 1
            deleted = a * b
            unvisited_indexes[deleted == 1] = 0

        self._labels = labels
        print(len(set(labels.to('cpu').numpy().tolist())))

    @property
    def labels_(self):
        return self._labels

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


