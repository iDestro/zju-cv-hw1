import torch
from Kmeans import KMeans
from BagOfVisualWords import BagOfVisualWords
from ImageRetriever import ImageRetriever
from utils import load_images, load_image, imshow
import argparse

parser = argparse.ArgumentParser(description='Image retrieve base on bag of visual words')
parser.add_argument('image_path', type=str, help='choose a image to retrieve')
parser.add_argument('image_database_path', default="./dataset", type=str, help='choose a image database(just a path)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    kmeans = KMeans(n_clusters=1000, device=device)
    target_image = load_image(args.image_path)
    images = load_images(args.image_database_path)
    image_retriever = ImageRetriever(BagOfVisualWords(images=images, kmeans=kmeans))
    result = image_retriever.retrieve(target_image)
    imshow(result)