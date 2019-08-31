import cv2
import os
import matplotlib.pyplot as plt

loc = './natural_images/' # Location of the dataset.
folders = os.listdir(loc) # Get all subfolders names (same for Annotation and Images folders).
n_classes = len(folders) # Get number of classes.
          
class_count = []
for folder_name in folders:
    class_size = len(os.listdir(loc + folder_name))
    class_count.append(class_size)
    
min_count = min(class_count)
max_count = max(class_count)
print('Minimum Class Count : ', min_count)
print('Maximum Class Count : ', max_count)
    
cl = [i for i in range(1, n_classes + 1)]
plt.bar(cl, height=class_count)
plt.show()
#plt.savefig('class_count.png')

max_h = 0
max_w = 0
min_h = 3000
min_w = 3000

for i in range(n_classes): # Iterate through the classes
    
    image_loc = loc + folders[i] + '/' # Specify the location of the images of the class.
    image_names = os.listdir(image_loc) # Get the file name of each image.
    
    for img_name in image_names: # Iterate through the names of the images.
        img = cv2.imread(image_loc + img_name) # Load the image.
        h = img.shape[0] # Get the image's height
        w = img.shape[1] # Get the image's width
        if h > max_h:
            max_h = h
        elif h < min_h:
            min_h = h
        if w > max_w:
            max_w = w
        elif w < min_w:
            min_w = w

print('Minimum Image Width : ', min_w)           
print('Maximum Image Width : ', max_w)
print('Minimum Image Height : ', min_h)
print('Maximum Image Height : ', max_h)
