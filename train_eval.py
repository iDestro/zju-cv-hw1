import numpy as np
from sklearn import metrics
from Kmeans import KMeans
from BagOfVisualWords import BagOfVisualWords
from ImageRetriever import ImageRetriever
import torch
from utils import load_images
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")


def test(train_path, test_path):
    test_acc, test_report, test_confusion = evaluate(train_path, test_path)
    print("Test Acc: " + str(test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def evaluate(train_path, test_path):
    images = load_images(test_path)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    classes = ['bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall']
    classes2idx = {key: val for val, key in enumerate(classes)}
    kmeans = KMeans(n_clusters=70, device=device)
    image_retriever = ImageRetriever(BagOfVisualWords(images=load_images(train_path), kmeans=kmeans))
    for image in images:
        labels_all = np.append(labels_all, classes2idx[image.label])
        predict_all = np.append(predict_all, classes2idx[image_retriever.retrieve(image).label])
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=classes, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return acc, report, confusion


if __name__ == '__main__':
    test('./dataset/train', './dataset/test')