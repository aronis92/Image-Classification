from PIL import Image
import numpy as np
import cv2
import os

# Function that creates a folder if it does not exist.
def create_dir(path):
    try:
        os.mkdir(path)
    except:
        pass
    
def process_dataset(new_size, output_folder_name):
    loc = './natural_images/' # Location of the dataset.
    folders = os.listdir(loc) # Get all subfolders names (same for Annotation and Images folders).
    n_classes = len(folders) # Get number of classes.
    
    # Create the folders to store the new dataset (Resized images and annotations).
    create_dir(output_folder_name + '/')
    
    for i in range(n_classes): # Iterate through the classes
        # Some Initializations.
        image_loc = loc + folders[i] + '/' # Specify the location of the images of the class.
        processed_image_loc = output_folder_name + '/' + folders[i] + '/' # Location to store processed images.
        image_names = os.listdir(image_loc) # Get the file name of each image.
        create_dir(processed_image_loc) # Create the folder.
        
        for img_name in image_names: # Iterate through the names of the images.
            
            img = cv2.imread(image_loc + img_name) # Load the image.
            h = img.shape[0] # Get the image's height
            w = img.shape[1] # Get the image's width
            
            # Resize the Image using the following:
            # We keep the biggest value of h or w and resize that to the new_size.
            # The second value is resized as a percentage of the origial resizing in order to maintain the height/width ratio.
            # The rest of the image that is needed to reach size (new_size, new_size) is covered in black pixels 
            if h > w: # If the height is greater that the width.
                new_w = int(np.round(new_size*w/h) ) # Calculate the new width of the image.
                img = cv2.resize(img, (new_w, new_size) ) # Resize the image.
                w_diff = new_size - new_w # Calculate the width that is needed to make the image square.
                left = int(w_diff/2) # Calulate how much will apply on the left side.
                right = int(w_diff/2) + w_diff%2 # Calulate how much will apply on the right side (add one more pixel for odd w_diff).
                
                # If there is blank space to be filled on the left create a black image and concatenate with the resized image.
                if left > 0: 
                    temp = Image.fromarray(np.zeros((new_size, left, 3), np.uint8))
                    img = np.concatenate((temp, img), axis=1)
                # If there is blank space to be filled on the right create a black image and concatenate with the resized image.
                if right > 0:
                    temp = Image.fromarray(np.zeros((new_size, right, 3), np.uint8))
                    img = np.concatenate((img, temp), axis=1)
            elif w > h: # If the width is greater that the height.
                new_h = int(np.round(new_size*h/w)) # Calculate the new height of the image.
                img = cv2.resize(img,(new_size, new_h)) # Calculate the new width of the image.
                h_diff = new_size - new_h # Calculate the height that is needed to make the image square.
                upper = int(h_diff/2) # Calulate how much will apply on the upper side.
                lower = int(h_diff/2) + h_diff%2 # Calulate how much will apply on the right side (add one more pixel for odd w_diff).
                    
                # If there is blank space to be filled on the upper side create a black image and concatenate with the resized image.
                if upper > 0:
                    temp = Image.fromarray(np.zeros((upper, new_size, 3), np.uint8))
                    img = np.concatenate((temp, img), axis=0)
                # If there is blank space to be filled below create a black image and concatenate with the resized image.
                if lower > 0:
                    temp = Image.fromarray(np.zeros((lower, new_size, 3), np.uint8))
                    img = np.concatenate((img, temp), axis=0)
            else: # If the image is already square.
                img = cv2.resize(img,(new_size, new_size)) # Resize image.
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.save(processed_image_loc + img_name)

def equalize_classes():
    loc = './processed_dataset/' # Location of the dataset.
    folders = os.listdir(loc) # Get all subfolders names (same for Annotation and Images folders).
    n_classes = len(folders) # Get number of classes.
    
    for i in range(n_classes): # Iterate through the classes
        image_loc = loc + folders[i] + '/' # Specify the location of the images of the class.
        image_names = os.listdir(image_loc) # Get the file name of each image.
        needed_images = 1000 - len(image_names)
        class_name = image_names[0][:-8]
        
        for j in range(needed_images):
            img_name = image_names[j]
            img = cv2.imread(image_loc + img_name) # Load the image.
            new_img = cv2.bilateralFilter(img, 9, 80, 80)
    
            number = len(image_names) + j + 1
            if number < 1000:
                number = '0' + str(number)
            else:
                number = '1000'
    
            new_img_name = class_name + number + '.jpg'
            new_img = Image.fromarray(new_img)
            new_img.save(image_loc + new_img_name)

#new_image_size = 256 # Each image will be resized to (new_size, new_size, 3)
#output_folder_name = 'processed_dataset'
#process_dataset(new_image_size, output_folder_name)
#equalize_classes()


