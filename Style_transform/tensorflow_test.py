from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import cv2
from keras.applications import vgg19
from keras import backend as K

base_image_path = "images/input.jpg"
style_reference_image_path = "images/style.jpg"
result_prefix = "output"
iterations = 5

# 原圖與風格圖佔output比重
content_weight = 0.025
style_weight = 1.0
# 損失總差異預設值
total_variation_weight = 1.0

# output 圖的寬高
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# 轉換成 VGG 19 input 格式
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    print(img.shape)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


base_image = K.variable(preprocess_image(base_image_path))
#print(type(base_image))
#cv2.imshow(base_image)