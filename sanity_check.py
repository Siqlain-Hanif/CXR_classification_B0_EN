import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tqdm import tqdm
import os

#Checking model weights
"""
my_model_1 = tf.keras.models.load_model('C:/Users/Asad/My Drive/PhD/Code/CXR/chexpert_pre_training/Models/effnet_v1B0_384_6_classes_crossentropy_adam_val_accuracy_softmax_imagenet_same_augmentation.h5')
#my_model_1.load_weights('C:/Users/Asad/My Drive/PhD/Code/CXR/chexpert_pre_training/Models/m4-16_weights.h5')
my_model_2 = tf.keras.models.load_model('C:/Users/Asad/My Drive/PhD/Code/CXR/chexpert_pre_training/Models/m4-16.h5')

for layer_1, layer_2 in zip(my_model_1.layers, my_model_2.layers):
    print(layer_1.name)
    weights_1 = layer_1.get_weights()
    weights_2 = layer_2.get_weights()
    if len(weights_1) > 0:
        comparison = [np.array_equal(w1, w2) for w1, w2 in zip(weights_1, weights_2)]
        comparison = bool(sum(comparison))
    else:
        comparison = weights_1 == weights_2
    print('Layer weights are same? ',comparison)

print('Done')
"""

#Temperature Scaling
"""
def temp_relu(x):
    divider = 2
    #x = tf.clip_by_value(x, -1, 88.65)
    print((x/divider).numpy())
    print(K.exp(x/divider).numpy())
    print(K.sum(K.exp(x/divider)).numpy())
    #print(tf.math.divide(x, divider).numpy())
    #print(tf.math.exp(tf.math.divide(x, divider)).numpy())
    #return (tf.math.divide(tf.math.exp(tf.math.divide(x, divider)),
    #                       tf.math.reduce_sum(tf.math.exp(tf.math.divide(x, divider)))))
    return(K.exp(x/divider)/(K.sum(K.exp(x/divider))))

x = tf.constant([1, -0.566, 0.5, 4], dtype=tf.float32)
y = temp_relu(x).numpy()
print(y)
print('Done')
"""

#Creating models with custom activation
"""
def temp_relu(x):
    divider = 2
    return(K.exp(x/divider)/(K.sum(K.exp(x/divider))))

my_classes = 6

my_base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(256,256,3), classes=my_classes, pooling='max')

x = tf.keras.layers.Dropout(0.1)(my_base_model.output)
x = tf.keras.layers.Activation(temp_relu)(x)
x = tf.keras.layers.Dense(my_classes, activation='softmax', name='Output', dtype='float32')(x)
my_model = tf.keras.Model(inputs=my_base_model.inputs, outputs=x)

config = my_model.get_config()


my_optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)

my_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=my_optimiser,
                                     metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])
"""

#Making sure that the k-fold splits is correct
"""
input_dir = 'C:/Users/Asad/Desktop/Folds/Fold_1/Validation/10. PLEURAL EFFUSION/'
check_dirs_list = ['C:/Users/Asad/Desktop/Folds/Fold_2/Validation/10. PLEURAL EFFUSION/',
                   'C:/Users/Asad/Desktop/Folds/Fold_3/Validation/10. PLEURAL EFFUSION/',
                   'C:/Users/Asad/Desktop/Folds/Fold_4/Validation/10. PLEURAL EFFUSION/',
                   'C:/Users/Asad/Desktop/Folds/Fold_5/Validation/10. PLEURAL EFFUSION/']


classes = ['1. ATELECTASIS', '2. CARDIOMEGALY', '3. CONSOLIDATION', '4. EDEMA', '9. NO FINDING', '10. PLEURAL EFFUSION']


image_list = os.listdir(input_dir)
print('Total Images: ', len(image_list))
for check_dir in check_dirs_list:
    check_list = os.listdir(check_dir)
    print('Overlapping Images: ', len(set(image_list).intersection(set(check_list))))

print('Done')
"""