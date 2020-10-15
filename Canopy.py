import torch
import numpy as np
import random


class Canopy:
    def __init__(self, x, t1, t2):
        self.x = x
        self.t1 = t1
        self.t2 = t2
        self._labels = None
        self._cluster_centers = None

    def fit(self):
        labels = torch.zeros([len(self.x), 1])
        unvisited_indexes = list(range(len(self.x)))
        canopies = {}
        while len(unvisited_indexes) != 0:
            canopy_center = random.choice(unvisited_indexes)
            canopy = [canopy_center]
            delete_indexes = []
            unvisited_indexes.remove(canopy_center)
            for index in unvisited_indexes:
                dis = torch.sqrt(torch.sum((self.x[index]-self.x[canopy_center])**2))
                if dis < self.t1:
                    canopy.append(index)
                if dis < self.t2:
                    delete_indexes.append(index)
            unvisited_indexes = [i for i in unvisited_indexes if i not in delete_indexes]
            canopies[canopy_center] = canopy
        type = {index: i for i, index in enumerate(canopies.keys())}
        cluster_centers = []
        for key, value in canopies.items():
            labels[value, :] = type[key]
            cluster_centers.append(self.x[key])

        self._labels = labels
        self._cluster_centers = torch.Tensor([np.array(i) for i in cluster_centers])

    @property
    def labels_(self):
        return self._labels

    @property
    def cluster_centers_(self):
        return self._cluster_centers

