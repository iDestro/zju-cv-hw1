from Canopy import Canopy
import torch
import numpy as np
import matplotlib.pyplot as plt

a = np.random.rand(20, 2)*10 + np.array([0, 0])
b = np.random.rand(30, 2)*5 + np.array([0, 8])
c = np.random.rand(20, 2)*8 + np.array([8, 0])

data = np.concatenate([a, b, c], axis=0)

data = torch.Tensor(data)

canopy = Canopy(data, 10, 6)
canopy.fit()
labels = canopy.labels_
centers = canopy.cluster_centers_
print("types:", len(centers))

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.show()