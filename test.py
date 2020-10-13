import torch
import numpy as np
from Kmeans import KMeans


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    import numpy as np

    data_1 = np.random.randn(200, 2) + [1, 1]
    data_2 = np.random.randn(200, 2) + [4, 4]
    data_3 = np.random.randn(200, 2) + [7, 1]
    data = np.concatenate((data_1, data_2, data_3), axis=0)

    data = torch.Tensor(data)

    kmeans = KMeans(n_clusters=3, device=device)
    kmeans.fit(data)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', c='k')
    plt.show()

    test_1 = np.random.randn(200, 2) + [1, 1]
    test_2 = np.random.randn(200, 2) + [4, 4]
    test_3 = np.random.randn(200, 2) + [7, 1]
    test_data = np.concatenate((test_1, test_2, test_3), axis=0)

    test_data = torch.Tensor(test_data)

    clus_pred = kmeans.predict(test_data)
    plt.clf()
    plt.scatter(test_data[:, 0], test_data[:, 1], alpha=0.5, c=clus_pred)
    plt.show()


