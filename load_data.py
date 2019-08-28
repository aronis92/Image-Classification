import numpy as np
import random
import cv2
import os
from IPython.display import display
from PIL import Image, ImageDraw

def load_dataset():
    images_loc = './processed_dataset/'
    folders = os.listdir(images_loc)
    n_classes = len(folders)

    size_of_classes = []
    images = []
    labels = []

    for i in range(n_classes):
        image_loc = images_loc + folders[i] + '/' # Specify the location of the images of the class.
        image_names = os.listdir(image_loc)
        size_of_classes.append(len(image_names))
        labels.extend([i]*size_of_classes[i])

        for img_name in image_names:
            # Load the image and append it to the images list.
            img = cv2.imread(image_loc + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    random.seed(0)
    temp = list(zip(images, labels))
    random.shuffle(temp)
    images, labels = zip(*temp)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def show_img(img):
    img = Image.fromarray(img)
    display(img)

images, labels = load_dataset()






