# import numpy as np
#
# a = np.array([1, 2, 3]).reshape(1, 3)
# b = np.array([4, 5, 6]).reshape(1, 3)
#
# print(np.concatenate((a, b), axis=0))

import torch

a = torch.arange(3).view(1, 3)
b = torch.arange(3).view(1, 3)

print(torch.cat((a, b), dim=0))