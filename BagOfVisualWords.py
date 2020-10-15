from Kmeans import KMeans
import numpy as np
import torch
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')


class BagOfVisualWords:
    def __init__(self, images, kmeans):
        self.images = images
        self.kmeans = kmeans
        self.total_descriptors = self.__load_total_descriptors()
        self.visual_words = self.__generate_visual_words()
        self.inverted_file_table = self.__generate_inverted_file_table()

    def __load_total_descriptors(self):
        total_descriptors = None
        for image in self.images:
            if total_descriptors is None:
                total_descriptors = image.descriptors
            else:
                np.concatenate((total_descriptors, image.descriptors), axis=0)
        return total_descriptors

    def __generate_visual_words(self):
        k = 1000
        kmeans = KMeans(n_clusters=k, device=device)
        kmeans.fit(self.total_descriptors)
        centers = kmeans.cluster_centers_
        return centers

    def __generate_inverted_file_table(self):
        labels = self.kmeans.labels_
        inverted_file_table = {i: [] for i in range(self.kmeans.n_clusters)}
        descriptor_cursor = 0
        for image in self.images:
            descriptors_size = image.descriptors_size
            cur_image_labels = labels[descriptor_cursor: descriptor_cursor+descriptors_size]
            self.__generate_image_histogram(image, cur_image_labels)
            for label in cur_image_labels:
                inverted_file_table[label].append(image.filename)
        return inverted_file_table

    def __generate_image_histogram(self, image, labels):
        histogram = torch.zeros(len(self.kmeans.n_clusters))
        for label in labels:
            histogram[label] += 1
        image.set_histogram(histogram)
