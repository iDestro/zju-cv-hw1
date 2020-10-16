from sklearn.cluster import KMeans
from sklearn import metrics
import torch
import numpy as np
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def silhouette_score(x, labels):
    x = x.to(device)
    labels = labels.to(device)
    length = len(x)
    indexes = torch.arange(length)
    sampled_indexes = np.random.choice(indexes, 10000)
    ones_vector = torch.ones(length).to(device)
    total = torch.Tensor([0]).to(device)
    for index, i in enumerate(x[sampled_indexes]):
        if index % 100 == 0:
            print(index)
        matched = labels == labels[index]
        dis = torch.sqrt(torch.sum((x - i) ** 2, dim=1))
        cnt = torch.sum(ones_vector[matched])
        a = torch.sum(dis[matched]) / cnt
        matched = labels != labels[index]
        cnt = torch.sum(ones_vector[matched])
        b = torch.sum(dis[matched]) / cnt
        total += (b - a) / (torch.max(a, b))
    return total / 10000


descriptors = pickle.load(open('./total_descriptors.plk', 'rb'))
kmeans = KMeans(n_clusters=200)
kmeans.fit(descriptors)
labels = kmeans.labels_
print(metrics.calinski_harabasz_score(descriptors, labels))