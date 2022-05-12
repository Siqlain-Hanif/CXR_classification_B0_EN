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
from my_utils import training_strategy
#from keras.applications.efficientnet_v2 import *
from sklearn.utils.extmath import softmax
#import pydicom
#from pydicom.pixel_data_handlers.util import apply_voi_lut



#Extracting from DICOM and resizing the BRAX dataset to make it more managable
"""
input_dir = 'H:/Datasets/physionet.org/files/brax/1.0.0/Anonymized_DICOMs/*/*/*/*.dcm'
output_dir = 'H:/Datasets/BRAX/Training Data/'


def image_extraction_and_saving(path):

    my_dicom = pydicom.dcmread(path)

    image_data = apply_voi_lut(my_dicom.pixel_array, my_dicom)

    if my_dicom.PhotometricInterpretation == 'MONOCHROME1':
        image_data = np.amax(image_data) - image_data

    image_data = image_data - np.min(image_data)
    image_data = image_data / np.max(image_data)
    image_data = (image_data*255).astype(np.uint8)

    my_image = cv2.resize(np.stack((image_data,) * 3, axis=-1), (1024, 1024))

    parts = path.split('\\')


    new_name = '_'.join(parts[1:]).replace('.dcm', '.jpg')

    cv2.imwrite(os.path.join(output_dir, new_name), my_image)



input_dir_files = glob(input_dir, recursive=True)
input_dir_files = [x for x in input_dir_files if '.html' not in x]

_ = thread_map(image_extraction_and_saving, input_dir_files, max_workers=7)
print('Done')
"""

#Renaming files to .dcm so that they can be read
"""
input_dir = 'H:/Datasets/physionet.org/files/brax/1.0.0/Anonymized_DICOMs/*/*/*/*'
input_dir_files = glob(input_dir, recursive=True)
input_dir_files = [x for x in input_dir_files if '.html' not in x]

for file in tqdm(input_dir_files):
    os.rename(file, file+'.dcm')
"""


#Distributing the BRAX images into classes and resizing them to (768, 768)
"""
my_csv = pd.read_csv('H:/Datasets/BRAX/master_spreadsheet_update.csv')
classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
           'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
           'Support Devices']
classes_output_folders = ['1. ATELECTASIS', '2. CARDIOMEGALY', '3. CONSOLIDATION', '4. EDEMA', '5. ENLARGED CARDIOMEDIASTINUM',
                          '6. FRACTURE', '7. LUNG LESION', '8. LUNG OPACITY', '9. NO FINDING', '10. PLEURAL EFFUSION',
                          '11. PLEURAL OTHER', '12. PNEUMONIA', '13. PNEUMOTHORAX', '14. SUPPORT DEVICES']

input_dir = 'H:/Datasets/BRAX/Training Data/'
output_dir = 'H:/Datasets/BRAX/Classes/'

my_samples = my_csv[['DicomPath', 'ViewPosition', *classes]]
my_samples = my_samples[my_samples['ViewPosition'] != 'L']
my_samples = my_samples[my_samples['ViewPosition'] != 'RL']


for sample in tqdm(range(len(my_samples))):
    my_sample = my_samples.iloc[sample]
    image_path = '_'.join(my_sample['DicomPath'].split('/')[1:])[:-18] + '.jpg'

    #my_sample = list(my_sample)
    for counter, my_class in enumerate(my_sample[2:]):
        if my_class == 1.0:
            if os.path.isfile(os.path.join(input_dir, image_path)):
                shutil.copy(os.path.join(input_dir, image_path),
                        os.path.join(output_dir, classes_output_folders[counter]))
            else:
                print(image_path)



print('Done')
"""

#Spliting the 6 classes of BRAX dataset into folds
"""
input_dir_list = ['C:/Users/Asad/Desktop/BRAX/Classes/1. ATELECTASIS/',
                  'C:/Users/Asad/Desktop/BRAX/Classes/2. CARDIOMEGALY/',
                  'C:/Users/Asad/Desktop/BRAX/Classes/3. CONSOLIDATION/',
                  'C:/Users/Asad/Desktop/BRAX/Classes/4. EDEMA/',
                  'C:/Users/Asad/Desktop/BRAX/Classes/9. NO FINDING/',
                  'C:/Users/Asad/Desktop/BRAX/Classes/10. PLEURAL EFFUSION/']

output_fold_1_list = ['C:/Users/Asad/Desktop/Folds/Fold_1/','C:/Users/Asad/Desktop/Folds/Fold_2/',
                      'C:/Users/Asad/Desktop/Folds/Fold_3/','C:/Users/Asad/Desktop/Folds/Fold_4/',
                      'C:/Users/Asad/Desktop/Folds/Fold_5/']
split = ['Training', 'Validation']
classes = ['1. ATELECTASIS', '2. CARDIOMEGALY', '3. CONSOLIDATION', '4. EDEMA', '9. NO FINDING', '10. PLEURAL EFFUSION']

for my_class, input_dir in tqdm(enumerate(input_dir_list)):
    image_list = os.listdir(input_dir)
    if classes[my_class] == '9. NO FINDING':
        image_list = image_list[1:2000]

    random.shuffle(image_list)
    total_images = len(image_list)
    images_in_each_split = int(total_images/5)

    start_stop_indices = []
    for i in range(5):
        start_stop_indices.append([i*images_in_each_split, (i+1)*images_in_each_split])

    start_stop_indices[-1] = [start_stop_indices[-1][0], total_images]

    for fold, index_pair in enumerate(start_stop_indices):
        current_image_list = image_list[index_pair[0]:index_pair[1]]
        remaining_folds = [x for x in start_stop_indices if start_stop_indices.index(x) != fold]

        for image in current_image_list:
            shutil.copy(os.path.join(input_dir, image), os.path.join(output_fold_1_list[fold],split[1],classes[my_class],
                                                                     image))

        for remaining_fold in remaining_folds:
            current_image_list = image_list[remaining_fold[0]:remaining_fold[1]]
            for image in current_image_list:
                shutil.copy(os.path.join(input_dir, image), os.path.join(output_fold_1_list[fold], split[0],
                                                                         classes[my_class], image))

print('Done')
"""

# Running trained models on test dataset

num_gpus, strategy = training_strategy(set_manually=True, which_gpu=0)

classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'No Finding', 'Pleural Effusion']
input_dir_1 = 'C:/Users/Asad/Desktop/Folds/Fold_3/Validation/1. ATELECTASIS/*.*'
input_dir_2 = 'C:/Users/Asad/Desktop/Folds/Fold_3/Validation/2. CARDIOMEGALY/*.*'
input_dir_3 = 'C:/Users/Asad/Desktop/Folds/Fold_3/Validation/3. CONSOLIDATION/*.*'
input_dir_4 = 'C:/Users/Asad/Desktop/Folds/Fold_3/Validation/4. EDEMA/*.*'
input_dir_5 = 'C:/Users/Asad/Desktop/Folds/Fold_3/Validation/9. NO FINDING/*.*'
input_dir_6 = 'C:/Users/Asad/Desktop/Folds/Fold_3/Validation/10. PLEURAL EFFUSION/*.*'

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
    model_path = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-32.h5'
    #model_path = 'C:/Users/Asad/Desktop/New folder (3)/m1-102.h5'
    with strategy.scope():
        my_model = tf.keras.models.load_model(model_path)
    # my_model.load_weights(weight_path)
    print('Model loaded and weights added...')

if enable_ensemble:
    model_path_1 = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-29.h5'
    model_path_2 = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-30.h5'
    model_path_3 = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-31.h5'
    model_path_4 = 'C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/m1-32.h5'

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


image_size = (384, 384, 3)


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


#Seprating a set number of lateral images from all the classes that will be used as validation dataset for segmentation and opacity models
"""
input_dir = 'C:/Users/Asad/Desktop/BRAX_Classes/Original/14. SUPPORT DEVICES/'
output_dir = 'C:/Users/Asad/Desktop/BRAX_Classes/Validation/14. SUPPORT DEVICES/'

input_dir_images = os.listdir(input_dir)

random_indices = random.sample(range(0, len(input_dir_images)), 25)

for index in tqdm(random_indices):
    shutil.copy(os.path.join(input_dir, input_dir_images[index]), os.path.join(output_dir, input_dir_images[index]))
    os.remove(os.path.join(input_dir, input_dir_images[index]))
"""

#Finding and removing the overlapped images among different classes so that we have 200 distinct images
"""
check_dir = 'C:/Users/Asad/Desktop/BRAX_Classes/Validation/14. SUPPORT DEVICES/'
all_other_dirs_list = ['C:/Users/Asad/Desktop/BRAX_Classes/Validation/1. ATELECTASIS/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/2. CARDIOMEGALY/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/3. CONSOLIDATION/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/4. EDEMA/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/5. ENLARGED CARDIOMEDIASTINUM/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/6. FRACTURE/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/7. LUNG LESION/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/8. LUNG OPACITY/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/9. NO FINDING/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/10. PLEURAL EFFUSION/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/11. PLEURAL OTHER/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/12. PNEUMONIA/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/13. PNEUMOTHORAX/',
                       'C:/Users/Asad/Desktop/BRAX_Classes/Validation/14. SUPPORT DEVICES/']

all_images = set(os.listdir(check_dir))


for dir in tqdm(all_other_dirs_list):
    if dir.split('/')[-2] != check_dir.split('/')[-2]:
        dir_images = set(os.listdir(dir))
        if len(list(all_images.intersection(dir_images))) > 0:
            print('Found overlapping images in ' + dir)
            print('Total images: ', len(list(all_images.intersection(dir_images))))
            print(list(all_images.intersection(dir_images)))

print('Done')
"""