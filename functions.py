import cv2 as cv
import numpy as np
import tensorflow as tf
from keras import models, layers

def get_Xtf(path, nb_image):
    '''
    Transform images with true fast contours to an array from image information
    '''
    i = 0
    arr_images = np.ndarray(
        shape=(1, 540, 720, 3),
        dtype=np.uint8
        )
    
    img_file = f'{path}/{nb_image}.png'
    pre_img = cv.imread(img_file)
    
    if pre_img.shape[0] > 99:
        img = add_fast_true(pre_img) 
        arr_images[i] = img / 255
        i += 1

    return arr_images

def add_fast_true(pre_img):
  fast = cv.FastFeatureDetector_create()
  kp = fast.detect(pre_img, None)
  img = cv.drawKeypoints(pre_img, kp, None, color=(255,0,0))

  return img

def create_sequential(IMG_SIZE, NB_OUTPUTS, LR, NAME):
    h, w = IMG_SIZE[0], IMG_SIZE[1]
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, name=f'dense1_{NAME}'))
    model.add(layers.Dense(2 * NB_OUTPUTS, name=f'dense2_{NAME}'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model