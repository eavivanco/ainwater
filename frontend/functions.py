import streamlit_option_menu as sm #library for option menu
from PIL import Image
import cv2 as cv
import numpy as np
import yaml
# import tensorflow 
# from keras import models, layers

# --- STYLE ---
def read_schema_page():
    """
    > The function `read_schema_page()` reads the schema page from the file `files/schema_page.yaml` and
    returns the schema page as a Python dictionary
    :return: A dictionary of the schema_page.yaml file
    """
    yaml_path= 'files/schema_page.yaml' 
    schema_page=yaml.load(open(yaml_path),Loader=yaml.FullLoader)
    return schema_page

def menu_bar(select_color="#0d63ba",orientation="horizontal"):
    """
    `menu_bar()` creates a menu bar with the menu items and icons defined in the schema page
    
    :param select_color: the color of the selected menu item, defaults to #0d63ba (optional)
    :param orientation: "horizontal" or "vertical", defaults to horizontal (optional)
    :return: A list of the menu items.
    """
    schema = read_schema_page()
    menu_list = [i for i in schema.keys()]
    menu_icon = [schema[i] for i in menu_list] #https://icons.getbootstrap.com
    page = sm.option_menu(None, menu_list, 
        icons=menu_icon, 
        menu_icon="cast", default_index=0, orientation=orientation,
        styles={"nav-link-selected": {"background-color": select_color}})
    return page

# --- ETL ---
def resize_image(original_image):
    """
    > The function takes an image as an input and returns a resized version of the image
    
    :param original_image: The image that you want to resize
    :return: The resized image.
    """
    width = 720
    height = 540
    dim = (width, height)
    resized_image = original_image.resize(size=dim,
                                          resample=Image.LANCZOS )
    return resized_image

def load_image(image_file):
    """
    It loads an image file and returns the image
    
    :param image_file: The path to the image file
    :return: The image is being returned.
    """
    image = Image.open(image_file)
    return image

def get_Xtf(image):
    """
    It takes an image, adds a fast true to it, and returns the image as an array.
    
    :param image: the image to be processed
    """
    i = 0
    arr_images = np.ndarray(
        shape=(1, 540, 720, 3),
        dtype=np.uint8
        )
    pre_img = np.asarray(image)
    img = add_fast_true(pre_img) 
    arr_images[i] = img
    return arr_images

def add_fast_true(pre_img):
    """
    > The function `add_fast_true` takes an image as input and returns the same image with FAST features
    drawn on it
    
    :param pre_img: The image to be processed
    :return: The image with the keypoints drawn on it.
    """
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(pre_img, None)
    img = cv.drawKeypoints(pre_img, kp, None, color=(255,0,0))
    return img

# --- ML MODEL ---
# def create_sequential(IMG_SIZE, NB_OUTPUTS, LR, NAME):
#     """
#     It creates a convolutional neural network with 3 convolutional layers, 2 dense layers, and a final
#     output layer
    
#     :param IMG_SIZE: the size of the images that will be fed into the model
#     :param NB_OUTPUTS: number of outputs (classes)
#     :param LR: Learning rate
#     :param NAME: the name of the model, used to save the model and to name the layers
#     :return: A model with the following architecture:
#     """
#     h, w = IMG_SIZE[0], IMG_SIZE[1]
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, name=f'dense1_{NAME}'))
#     model.add(layers.Dense(2 * NB_OUTPUTS, name=f'dense2_{NAME}'))

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
#     return model

# def predict(X):
#     """
#     It loads the weights of the four models, then predicts the class of the input image
    
#     :param X: the image you want to predict on
#     :return: The predictions are being returned.
#     """
#     IMG_SIZE = [540, 720]
#     LR = 0.001
#     fi_model = create_sequential(IMG_SIZE, 6, LR, NAME='FI')
#     fi_model.load_weights('./weights/FilamentousIndex_ghmodelweights.h5')
#     fl1_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL1')
#     fl1_model.load_weights('./weights/Floculo1_ghmodelweights.h5')
#     fl2_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL2')
#     fl2_model.load_weights('./weights/Floculo2_ghmodelweights.h5')
#     fl3_model = create_sequential(IMG_SIZE, 2, LR, NAME='FL3')
#     fl3_model.load_weights('./weights/Floculo3_ghmodelweights.h5')
    
#     fi_pre_prediction = fi_model.predict(X)
#     prediction1 = np.argmax(fi_pre_prediction,axis=1)[0]
#     fl1_pre_prediction = fl1_model.predict(X)
#     prediction2 = np.argmax(fl1_pre_prediction,axis=1)[0]
#     fl2_pre_prediction = fl2_model.predict(X)
#     prediction3 = np.argmax(fl2_pre_prediction,axis=1)[0]
#     fl3_pre_prediction = fl3_model.predict(X)
#     prediction4 = np.argmax(fl3_pre_prediction,axis=1)[0]

#     predictions = [prediction1, prediction2, prediction3, prediction4]
#     return predictions