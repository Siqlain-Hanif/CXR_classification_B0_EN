import pandas as pd
#import tensorflow as tf
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#import efficientnet.tfkeras as efn
#from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
#from glob import glob
from tqdm.auto import tqdm
import numpy as np
#from matplotlib import pyplot as plt
import cv2
#from sklearn.decomposition import PCA
#import pickle
#import scipy.io as scio
#from keras.utils.np_utils import to_categorical
from tqdm.contrib.concurrent import thread_map
import shutil
import random
import tensorflow as tf
from glob import glob
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from keras.applications.efficientnet_v2 import *
from sklearn.utils.extmath import softmax
#import pydicom
#from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.metrics import f1_score


#Checking out the segmentation from a specific model
"""
my_model = tf.keras.models.load_model('C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/u_net.h5')
my_model.load_weights('C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m2-45.h5')

test_dir = 'C:/Users/Asad/Desktop/BRAX_Validation/Images/14. SUPPORT DEVICES'
output_dir = 'C:/Users/Asad/Desktop/BRAX_Validation/Masks/unet/14. SUPPORT DEVICES'
ground_truths_dir = 'C:/Users/Asad/Desktop/segmentation_datasets/all_data/Testing Data/ground_truths/'

image_size = (512,512,1)
images = os.listdir(test_dir)

output_masks = []

for image in tqdm(images):
    my_image = cv2.resize(cv2.imread(os.path.join(test_dir, image), 0)/255.,(image_size[0], image_size[1])).reshape(image_size)
    my_predictions = my_model.predict(np.expand_dims(my_image, axis=0))
    #my_predictions = my_predictions[-1]

    output_mask = np.asarray(my_predictions[0]*255).astype(np.uint8).reshape(image_size[0],image_size[1])
    output_mask = cv2.threshold(output_mask, 0, 1, cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, image),np.array(output_mask[1]*255, dtype=np.uint8))
    #output_masks.append(output_mask[1][:,:])


# output_masks = np.asarray(output_masks)
# ground_truths_masks = []
#
# ground_truths = os.listdir(ground_truths_dir)
#
# for ground_truth in tqdm(ground_truths):
#     ground_truths_masks.append(cv2.resize(cv2.imread(os.path.join(ground_truths_dir, ground_truth), 0)/255.,(image_size[0], image_size[1])))
#
#
# ground_truths_masks = np.asarray(ground_truths_masks, dtype=np.uint8)
# output_masks = output_masks.reshape(output_masks.shape[0] * output_masks.shape[1] * output_masks.shape[2],1)
# ground_truths_masks = ground_truths_masks.reshape(ground_truths_masks.shape[0] * ground_truths_masks.shape[1]
#                                                   * ground_truths_masks.shape[2],1)
# validation_accuracy = f1_score(ground_truths_masks, output_masks, average='binary')
#
# print('F1 score: ', validation_accuracy)
print('Done')
"""

#Computing F1 score on the validation dataset
ground_truths_dirs_list = ['C:/Users/Asad/Desktop/Brax_Masks/Masks/1. ATELECTASIS/',
                           'C:/Users/Asad/Desktop/Brax_Masks/Masks/2. CARDIOMEGALY/',
                           'C:/Users/Asad/Desktop/Brax_Masks/Masks/3. CONSOLIDATION/',
                           'C:/Users/Asad/Desktop/Brax_Masks/Masks/4. EDEMA/',
                           #'C:/Users/Asad/Desktop/Brax_Masks/Masks/6. FRACTURE/',
                           'C:/Users/Asad/Desktop/Brax_Masks/Masks_1/7. LUNG LESION/',
                           ]
masks_dirs_list = ['C:/Users/Asad/Desktop/BRAX_Validation/Masks/lwnet/1. ATELECTASIS',
                   'C:/Users/Asad/Desktop/BRAX_Validation/Masks/lwnet/2. CARDIOMEGALY',
                   'C:/Users/Asad/Desktop/BRAX_Validation/Masks/lwnet/3. CONSOLIDATION',
                   'C:/Users/Asad/Desktop/BRAX_Validation/Masks/lwnet/4. EDEMA/',
                   #'C:/Users/Asad/Desktop/BRAX_Validation/Masks/unet/6. FRACTURE/',
                   'C:/Users/Asad/Desktop/BRAX_Validation/Masks/lwnet/7. LUNG LESION/',
                   ]


image_size = (512,512,1)

ground_truths_masks = []
model_masks = []



for ground_truths_dir in tqdm(ground_truths_dirs_list):
    images = os.listdir(ground_truths_dir)
    for image in images:
        ground_truths_masks.append(cv2.resize(cv2.imread(os.path.join(ground_truths_dir, image), 0)/255.,(image_size[0], image_size[1]))[:,:])

for model_masks_dir in tqdm(masks_dirs_list):
    images = os.listdir(model_masks_dir)
    for image in images:
        #image = image.split('.')[0]+'.png'
        model_masks.append(cv2.resize(cv2.imread(os.path.join(model_masks_dir, image), 0)/255.,(image_size[0], image_size[1]))[:,:])


ground_truths_masks = np.asarray(ground_truths_masks, dtype=np.uint8)
model_masks = np.asarray(model_masks, dtype=np.uint8)

model_masks = model_masks.reshape(model_masks.shape[0] * model_masks.shape[1] * model_masks.shape[2],1)
ground_truths_masks = ground_truths_masks.reshape(ground_truths_masks.shape[0] * ground_truths_masks.shape[1]
                                                  * ground_truths_masks.shape[2],1)
validation_accuracy = f1_score(ground_truths_masks, model_masks, average='binary')

print('F1 score: ', validation_accuracy)
print('Done')

