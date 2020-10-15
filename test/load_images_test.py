from utils import load_images

images = load_images('../dataset')

for img in images:
    print(img.filename, img.label, img.descriptors)
