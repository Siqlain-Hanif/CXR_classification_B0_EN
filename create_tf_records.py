import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import cv2
from tensorflow.keras.utils import to_categorical

# target_mappings = {'1. ATELECTASIS':0, '2. CARDIOMEGALY':1, '3. CONSOLIDATION':2, '4. EDEMA':3,
#                    '5. ENLARGED CARDIOMEDIASTINUM':4, '6. FRACTURE':5, '7. LUNG LESION':6, '8. LUNG OPACITY':7,
#                    '9. NO FINDING':8, '10. PLEURAL EFFUSION':9, '11. PLEURAL OTHER':10, '12. PNEUMONIA':11,
#                    '13. PNEUMOTHORAX':12, '14. SUPPORT DEVICES':13}

# target_mappings = {'1. ATELECTASIS':0, '2. CARDIOMEGALY':1, '3. CONSOLIDATION':2, '4. EDEMA':3,
#                    '9. NO FINDING':4, '10. PLEURAL EFFUSION':5 }
target_mappings = {'1. ATELECTASIS':0, '2. CARDIOMEGALY':1, '3. CONSOLIDATION':2, '4. EDEMA':3,
                   '9. NO FINDING':4, '10. PLEURAL EFFUSION':5 }



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature_modified(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def serialize_example(feature0, feature1, feature3, feature4):
    feature = {
        'image': _bytes_feature(feature0),
        'image_name': _bytes_feature(feature1),
        'target': _int64_feature(feature3),
        'one_hot_target': _float_feature_modified(feature4)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# input_dirs = ['C:/Users/Asad/Desktop/Classes/Training Data/1. ATELECTASIS/*.jpg',
#               'C:/Users/Asad/Desktop/Classes/Training Data/2. CARDIOMEGALY/*.jpg',
#               'C:/Users/Asad/Desktop/Classes/Training Data/3. CONSOLIDATION/*.jpg',
#               'C:/Users/Asad/Desktop/Classes/Training Data/4. EDEMA/*.jpg',
#               #'C:/Users/Asad/Desktop/Classes/Training Data/5. ENLARGED CARDIOMEDIASTINUM/*.jpg',
#               #'C:/Users/Asad/Desktop/Classes/Training Data/6. FRACTURE/*.jpg',
#               #'C:/Users/Asad/Desktop/Classes/Training Data/7. LUNG LESION/*.jpg',
#               #'C:/Users/Asad/Desktop/Classes/Training Data/8. LUNG OPACITY/*.jpg',
#               'C:/Users/Asad/Desktop/Classes/Training Data/9. NO FINDING/*.jpg',
#               'C:/Users/Asad/Desktop/Classes/Training Data/10. PLEURAL EFFUSION/*.jpg']#,
#               #'C:/Users/Asad/Desktop/Classes/Training Data/11. PLEURAL OTHER/*.jpg',
#               #'C:/Users/Asad/Desktop/Classes/Training Data/12. PNEUMONIA/*.jpg',
#               #'C:/Users/Asad/Desktop/Classes/Training Data/13. PNEUMOTHORAX/*.jpg']#,
#               #'C:/Users/Asad/Desktop/Classes/Validation Data/14. SUPPORT DEVICES/*.jpg']

input_dirs = ['C:/Users/Asad/Desktop/Folds/Fold_5/Validation/1. ATELECTASIS/*.jpg',
              'C:/Users/Asad/Desktop/Folds/Fold_5/Validation/2. CARDIOMEGALY/*.jpg',
              'C:/Users/Asad/Desktop/Folds/Fold_5/Validation/3. CONSOLIDATION/*.jpg',
              'C:/Users/Asad/Desktop/Folds/Fold_5/Validation/4. EDEMA/*.jpg',
              'C:/Users/Asad/Desktop/Folds/Fold_5/Validation/9. NO FINDING/*.jpg',
              'C:/Users/Asad/Desktop/Folds/Fold_5/Validation/10. PLEURAL EFFUSION/*.jpg',
              ]


all_images = []

for dir in tqdm(input_dirs):
    all_images.extend(glob(dir))


output_dir = 'C:/Users/Asad/Desktop/brax_6_fold_5/'

image_size = (768,768)
num_classes = 6

files_per_tf_record = 500

total_tf_record_files = (len(all_images)//files_per_tf_record)+ int(len(all_images)%files_per_tf_record!=0)

split = 'valid'
counter = 0

for file in tqdm(range(total_tf_record_files)):

    files_in_current_tf_record = min(files_per_tf_record, len(all_images)-file*files_per_tf_record)

    file_name = '%s%s_%ix%i_%.3i_%i.tfrec' %(output_dir,split, image_size[0], image_size[1], counter, files_in_current_tf_record)

    with tf.io.TFRecordWriter(file_name) as writer:
        for value in range(files_in_current_tf_record):
            image = all_images[value + (file*files_per_tf_record)]
            img = cv2.resize(cv2.imread(image, -1), image_size)
            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            name = image.split('\\')[-1]
            target = target_mappings[image.split('/')[-1].split('\\')[0]]
            example = serialize_example(img, str.encode(name), target, list(to_categorical(target, num_classes=num_classes)))
            writer.write(example)

    counter += 1