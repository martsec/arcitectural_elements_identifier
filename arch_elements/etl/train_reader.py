import os 
import numpy as np
import matplotlib.image as img 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class TrainReader:
    """
    This class reads the training folders containing the categories and datasets.
    
    It expects the following structure
    ```
    .
    ├── training_data 
        ├── category1
        |   ├── image1.png
        |   ├── ...
        |   └── imageN.jpg
        └── category2
            ├── image1b.png
            ├── ...
            └── imageNb.jpg
    ```
    """
    ImageShape = tuple
    OutputType = tuple

    def __init__(self, training_folder:str):
        self.dataset_dir = training_folder
        self.categories = os.listdir(self.dataset_dir)


    def load(self, image_size:ImageShape, color_mode='rgb') -> OutputType:
        """
        Reads the directory and returns the images, resizing them as needed

        :param shape_restriction: Target shape of the images as (height, width, channels)
        :type shape_restriction: tuple
        :param color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb". The desired image format
        :type color_mode: str
        :return: Tuple with np.array(images) and np.array(labels)
        :rtype: tuple
        """
        images = []
        labels = []
        for c in self.categories:
            path = os.path.join(self.dataset_dir, c) 
            if os.path.isdir(path):
                imgs = self.__list_images(os.listdir(path))
                for i in imgs:
                    images.append(self.__load_image(os.path.join(self.dataset_dir, c, i), image_size, color_mode))
                    labels.append(c)
        x = np.array(images)
        y = np.array(labels)
        del images
        del labels
        return (x, y)

    def __list_images(self, files: list) -> list:
        return [ i for i in files 
                if i.lower().endswith('.jpg') or i.lower().endswith('.png')]
     
    def __load_image(self, path:str, target_shape:tuple, color_mode:'rgb') -> np.array:
        image = img_to_array(load_img(path, color_mode=color_mode))
        shape = image.shape
        if len(shape) != 3:
            print("WARNING: image '" + path + "' does not fit the correct shape. Found: " + str(shape))
        elif shape != target_shape:
            image = tf.image.resize_with_pad(image, 
                target_height=target_shape[0], target_width=target_shape[1])

        return image
