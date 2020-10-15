from Canopy import Canopy
import torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = np.random.rand(20, 2)*10 + np.array([0, 0])
b = np.random.rand(30, 2)*5 + np.array([0, 8])
c = np.random.rand(20, 2)*8 + np.array([8, 0])

data = np.concatenate([a, b, c], axis=0)

data = torch.Tensor(data)

canopy = Canopy(10, 6, device=device)
canopy.fit(data)
labels = canopy.labels_
print(labels)
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.show()