from Kmeans import KMeans
import pickle
import os
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BagOfVisualWords:
    def __init__(self, images, kmeans):
        self.images = images
        if os.path.exists('./kmeans_'+str(kmeans.n_clusters)+'.pkl'):
            self.kmeans = pickle.load(open('./kmeans_'+str(kmeans.n_clusters)+'.pkl', 'rb'))
            self.visual_words, self.labels = self.kmeans.cluster_centers_, self.kmeans.labels_
        else:
            self.total_descriptors = self.__load_total_descriptors()
            self.kmeans = kmeans
            self.visual_words, self.labels = self.__generate_visual_words()
        self.inverted_file_table = self.__generate_inverted_file_table()

    def __load_total_descriptors(self):
        total_descriptors = None
        for image in self.images:
            if total_descriptors is None:
                total_descriptors = image.descriptors
            else:
                total_descriptors = np.concatenate((total_descriptors, image.descriptors), axis=0)
        return torch.Tensor(total_descriptors)

    def __generate_visual_words(self):
        self.kmeans.fit(self.total_descriptors)
        pickle.dump(self.kmeans, open('./kmeans_'+str(self.kmeans.n_clusters)+'.pkl', 'wb'))
        centers = self.kmeans.cluster_centers_
        labels = self.kmeans.labels_
        return centers, labels

    def __generate_inverted_file_table(self):
        labels = self.labels.numpy()
        inverted_file_table = {i: [] for i in range(self.kmeans.n_clusters)}
        for i in inverted_file_table.keys():
            print("inverted_file_table: ", i)
        descriptor_cursor = 0
        for image in self.images:
            descriptors_size = image.descriptors_size
            cur_image_labels = labels[descriptor_cursor: descriptor_cursor+descriptors_size]
            self.generate_image_histogram(image, cur_image_labels)
            for label in cur_image_labels:
                inverted_file_table[label].append(image.filename)
            descriptor_cursor += descriptors_size
        return inverted_file_table

    def generate_image_histogram(self, image, labels):
        histogram = torch.zeros([1, self.kmeans.n_clusters])
        for label in labels:
            histogram[:, label] += 1
        image.set_histogram(histogram)
