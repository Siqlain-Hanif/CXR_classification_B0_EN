import pandas as pd
#import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
#from keras.applications.efficientnet_v2 import *
from sklearn.utils.extmath import softmax
#import pydicom
#from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from itertools import cycle
from my_utils import training_strategy

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score


num_gpus, strategy = training_strategy(set_manually=True, which_gpu=1)

classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'No Finding', 'Pleural Effusion']
input_dir_1 = 'C:/Users/Asad/Desktop/Folds/Fold_2/Validation/1. ATELECTASIS/*.*'
input_dir_2 = 'C:/Users/Asad/Desktop/Folds/Fold_2/Validation/2. CARDIOMEGALY/*.*'
input_dir_3 = 'C:/Users/Asad/Desktop/Folds/Fold_2/Validation/3. CONSOLIDATION/*.*'
input_dir_4 = 'C:/Users/Asad/Desktop/Folds/Fold_2/Validation/4. EDEMA/*.*'
input_dir_5 = 'C:/Users/Asad/Desktop/Folds/Fold_2/Validation/9. NO FINDING/*.*'
input_dir_6 = 'C:/Users/Asad/Desktop/Folds/Fold_2/Validation/10. PLEURAL EFFUSION/*.*'

labels_from_folders = False
enable_ensemble = True

input_dir_1_images = glob(input_dir_1)
input_dir_2_images = glob(input_dir_2)
input_dir_3_images = glob(input_dir_3)
input_dir_4_images = glob(input_dir_4)
input_dir_5_images = glob(input_dir_5)
input_dir_6_images = glob(input_dir_6)

total_images = []
total_images.extend(input_dir_1_images)
total_images.extend(input_dir_2_images)
total_images.extend(input_dir_3_images)
total_images.extend(input_dir_4_images)
total_images.extend(input_dir_5_images)
total_images.extend(input_dir_6_images)

labels = []

if labels_from_folders:


    #labels = [0]*len(input_dir_1_images)
    labels = [[1, 0, 0, 0, 0, 0]]*len(input_dir_1_images)

    #labels.extend([1]*len(input_dir_2_images))
    labels.extend([[0, 1, 0, 0, 0, 0]]*len(input_dir_2_images))

    #labels.extend([2]*len(input_dir_3_images))
    labels.extend([[0, 0, 1, 0, 0, 0]]*len(input_dir_3_images))

    #labels.extend([3]*len(input_dir_4_images))
    labels.extend([[0, 0, 0, 1, 0, 0]]*len(input_dir_4_images))

    #labels.extend([4]*len(input_dir_5_images))
    labels.extend([[0, 0, 0, 0, 1, 0]]*len(input_dir_5_images))

    #labels.extend([5]*len(input_dir_6_images))
    labels.extend([[0, 0, 0, 0, 0, 1]]*len(input_dir_6_images))


else:
    my_csv = pd.read_csv('C:/Users/Asad/Desktop/master_spreadsheet_update.csv')
    my_csv['PngPath'] = my_csv['PngPath'].apply(lambda x: x[7:].replace('/','_')[0:187])
    for image in total_images:
        name = image.split('\\')[-1].replace('.jpg','')
        my_row = my_csv.loc[my_csv['PngPath'] == name]
        my_row = list(my_row[classes].fillna(0).values[0])
        labels.append(my_row)


predictions = []

if not enable_ensemble:
    model_path = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-28.h5'
    #model_path = 'C:/Users/Asad/Desktop/New folder (3)/m1-102.h5'
    with strategy.scope():
        my_model = tf.keras.models.load_model(model_path)
    # my_model.load_weights(weight_path)
    print('Model loaded and weights added...')

if enable_ensemble:
    model_path_1 = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-25.h5'
    model_path_2 = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-103.h5'
    model_path_3 = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-104.h5'
    model_path_4 = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-105.h5'

    with strategy.scope():
        my_model_1 = tf.keras.models.load_model(model_path_1)
        my_model_2 = tf.keras.models.load_model(model_path_2)
        my_model_3 = tf.keras.models.load_model(model_path_3)
        my_model_4 = tf.keras.models.load_model(model_path_4)

    print('Models loaded...')

TTA = False
use_test_time_temperature = False


# classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
#            'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
#            'Pneumothorax']


image_size = (768, 768, 3)


if use_test_time_temperature:
    def temp_relu(x):
        x = tf.clip_by_value(x, -1, 88.65)
        return (tf.math.divide(tf.math.exp(tf.math.divide(x, 2)),
                               tf.math.reduce_sum(tf.math.exp(tf.math.divide(x, 2)))))

    predictions_test_time_temperature = []
    x = tf.keras.layers.Lambda(temp_relu)(my_model.get_layer('max_pool').output)
    x = my_model.get_layer('Output')(x)
    my_model_test_time_temperature = tf.keras.Model(inputs=my_model.inputs, outputs=x)
    my_model_test_time_temperature.summary()

for image in tqdm(total_images):
        my_image = cv2.resize(cv2.imread(image, -1), image_size[0:2])#/255.0
        if not TTA and not enable_ensemble:
            my_predictions = my_model.predict(np.expand_dims(my_image, axis=0))
            my_predictions = list(my_predictions[0, :])
        if TTA:
            my_image_expanded = np.zeros((4, *image_size))
            my_image_expanded[0, ...] = my_image
            my_image_expanded[1, ...] = np.fliplr(my_image)
            my_image_expanded[2, ...] = np.roll(my_image, 5, axis=1)
            my_image_expanded[3, ...] = np.flipud(my_image)

            my_predictions = my_model.predict(my_image_expanded)

            my_predictions = np.reshape(np.mean(my_predictions, axis=0), (1, 6))
            my_predictions = list(my_predictions[0, :])

        if enable_ensemble:
            my_predictions_1 = my_model_1.predict(np.expand_dims(cv2.resize(my_image, (256,256)), axis=0))
            my_predictions_1 = list(my_predictions_1[0, :])
            my_predictions_2 = my_model_2.predict(np.expand_dims(cv2.resize(my_image, (384,384)), axis=0))
            my_predictions_2 = list(my_predictions_2[0, :])
            my_predictions_3 = my_model_3.predict(np.expand_dims(cv2.resize(my_image, (512,512)), axis=0))
            my_predictions_3 = list(my_predictions_3[0, :])
            my_predictions_4 = my_model_4.predict(np.expand_dims(cv2.resize(my_image, (768,768)), axis=0))
            my_predictions_4 = list(my_predictions_4[0, :])

            my_predictions = [my_predictions_1[i] + my_predictions_2[i] + my_predictions_3[i] + my_predictions_4[i] for i in
                              range(6)]
            my_predictios = [my_predictions[i] / 4 for i in range(6)]

        my_predictions = [my_predictions]
        predictions.append(list(my_predictions[0]))

        if use_test_time_temperature:
            my_predictions_test_time_temperature = my_model_test_time_temperature.predict(np.expand_dims(my_image, axis=0))
            my_predictions_test_time_temperature = list(my_predictions_test_time_temperature[0, :])
            predictions_test_time_temperature.append(my_predictions_test_time_temperature)

print(roc_auc_score(np.array(labels), np.array(predictions), average=None))
#print(roc_auc_score(np.array(labels), np.array(predictions_test_time_temperature), average=None))
#cmd = ConfusionMatrixDisplay((confusion_matrix(list(labels), list(my_predictions))),
#                                 display_labels=classes)
#cmd.plot()
#plt.show()

print('Done')

labels, predictions = np.array(labels), np.array(predictions)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#
#
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#
# # Finally average it and compute AUC
# mean_tpr /= n_classes
#
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# # Plot all ROC curves
# plt.figure()
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )
#
# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),
    ('densely dashed',(0, (5, 1))),
    #('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),

    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
('loosely dashed',(0, (5, 10)))]  # Same as '-.'

colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkgreen", "darkred", "darkblue"])
for i, color, linestyle_str_tuple in zip(range(6), colors, linestyle_str):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=1.5,
        linestyle = linestyle_str_tuple[1],
        label="ROC curve of {0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], linestyle=linestyle_str[-1][-1], lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fold 2 Ensemble ROC")
plt.legend(loc="lower right")
plt.show()