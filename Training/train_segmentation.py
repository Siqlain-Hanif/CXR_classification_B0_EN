"""
m2-45: unet trained wth 762 grayscale normalised images.
m2-46: unet trained wth 762 pseudo-colored normalised images.
m2-47: lunet trained wth 762 grayscale normalised images.
m2-48: lunet trained wth 762 pseudo-colored normalised images.
m2-49: lwnet trained wth 762 grayscale normalised images.
m2-50: lwnet trained wth 762 pseudo-colored normalised images.
"""
import math
import pandas as pd
import tensorflow as tf
import albumentations as albu
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.utils.np_utils import to_categorical
from my_utils import training_strategy, generate_image_paths_and_labels, DataGenerator, CustomCallback,\
    check_improvement, learning_rate_scheduler, cosine_annealing_scheduler, decaying_cosine_annealing_scheduler
#from keras_unet.models import vanilla_unet, custom_unet



def main():
    image_dimensions = (512, 512, 3)
    input_channles = image_dimensions[-1]
    # Training Parameters:
    epochs, initial_epoch = 500, 0
    network_type, batch_size, n_classes, model_name, build_new, patience = "segmentation", 2, 1, "LWnet", \
                                                                           True, 35
    train_on_cloud = False
    major_revision, minor_revision = 2, 51
    no_improvement = True
    multihead = True

    # Data Generator Parameters
    train_using_generator, shuffle_data, augment_data, return_labels, visualise = True, True, True, True, False
    training_testing_split = 0.9
    read_from_folder = True
    normalise_images = False

    #Callback Parameters
    num_heads = 2

    p = 0.25
    augmentations = [albu.Flip(p=p),
                     albu.HorizontalFlip(p=p),
                     albu.Cutout(always_apply=False, p=1.0, num_holes=25, max_h_size=9, max_w_size=9),

                     albu.NoOp(p=p), 0.65]
    #augmentations = None

    #Callback parameters
    use_validation_data = True
    save_sample_split, power_redundancy = False, False

    num_gpus, strategy = training_strategy(set_manually=False, which_gpu=0)
    num_gpus_multiplier = 2

    if train_on_cloud:
        #Vestigial code from previous generation. Not used.
        # For running on COLAB
        input_dir = ""
        model_save_path = ""
        log_save_path = ""
    else:
        # Input/Output Directories Parameters:
        input_dir = "C:/Users/Asad/Desktop/segmentation_datasets/all_data_resized/Training Data/images/"
        model_save_path = "C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/"
        log_save_path = "C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Logs/"

    training_data, training_labels, testing_data, testing_labels = None, None, None, None
    file_name = 'segmentation_images.csv'


    while True:
        if train_using_generator:
            training_data, training_labels, testing_data, testing_labels = generate_image_paths_and_labels(
                network_type=network_type, input_dir=input_dir, split=training_testing_split,
                read_from_folders=read_from_folder, file_name=file_name, classes=None)

        if save_sample_split:
            df = pd.DataFrame(index=None, data=[training_data, training_labels, testing_data, testing_labels]).T
            df.columns = ['training_data', 'training_labels', 'testing_data', 'testing_labels']
            df.to_csv('m%i-%i_%s.csv' % (major_revision, minor_revision, 'sample_split'), sep=',')

        if power_redundancy:
            print('Reading pre-saved training/validation split...')
            df = pd.read_csv('m%i-%i_%s.csv' % (major_revision, minor_revision, 'sample_split'), sep=',')

            training_data, training_labels, testing_data, testing_labels = df['training_data'].values,\
                                                                           df['training_labels'].values,\
                                                                           df['testing_data'].values,\
                                                                           df['testing_labels'].values

            testing_data = [x for x in testing_data if type(x) == type('str')]
            testing_labels = [x for x in testing_labels if not math.isnan(x)]

        # Building the model
        my_model = tf.keras.models.load_model('C:/Users/Asad/My Drive/PhD/Code/CXR/brax/Models/lwnet_3channels.h5')
        #my_model = custom_unet(input_shape=(512, 512, 1))
        #my_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['binary_accuracy'])

        #my_model.summary()


        # Obtaining Training and testing data from the training and testing generator
        generated_train_data = DataGenerator(list_ids=training_data, labels=training_labels,
                                             return_labels=return_labels, batch_size=batch_size, dim=image_dimensions,
                                             n_classes=None, shuffle_data=shuffle_data, augment_data=augment_data,
                                             network_type=network_type, visualise=visualise,
                                             major_revision=major_revision, minor_revision=minor_revision,
                                             train_on_cloud=None,
                                             model_log_path=log_save_path,
                                             augmentations=augmentations, multihead=multihead, input_channels=input_channles,
                                             normalise=normalise_images)

        generated_validation_data = DataGenerator(list_ids=testing_data, labels=testing_labels,
                                                  return_labels=return_labels, batch_size=batch_size,
                                                  dim=image_dimensions,
                                                  n_classes=None, shuffle_data=False,
                                                  augment_data=False,
                                                  network_type=network_type, visualise=visualise,
                                                  major_revision=major_revision,
                                                  minor_revision=minor_revision, model_log_path=log_save_path,
                                                  augmentations=augmentations, multihead=multihead,
                                                  input_channels=input_channles, normalise=normalise_images)


        # Creating Callbacks
        my_custom_callback = CustomCallback(model=my_model, network_type=network_type,
                                            validation_data=generated_validation_data,
                                            validation_labels=None,
                                            major_revision=major_revision, minor_revision=minor_revision,
                                            model_save_path=model_save_path, model_log_path=log_save_path,
                                            model_name=model_name, use_validataion_accuracy=True,
                                            patience=patience, map_iou_threshold=None,
                                            classes=None, multihead=multihead, num_heads=num_heads)

        my_callbacks = [my_custom_callback,
                        tf.keras.callbacks.LearningRateScheduler(decaying_cosine_annealing_scheduler, verbose=True)]

        print('Scheduler selected: ', decaying_cosine_annealing_scheduler)

        print('Training Testing Split is: ', training_testing_split)

        # Training
        my_model_history = my_model.fit(generated_train_data, epochs=epochs,
                                        initial_epoch=initial_epoch, callbacks=my_callbacks,
                                        verbose=1, max_queue_size=(num_gpus * num_gpus_multiplier) * 64, workers=128)

        if 'loss' in my_model_history.history.keys():
            if len(my_model_history.history['loss']) > 1:
                build_new = False
            else:
                build_new = True

        if 'categorical_accuracy' in my_model_history.history.keys():
            no_improvement = check_improvement(model=network_type,
                                               history=my_model_history.history['categorical_accuracy'],
                                               patience=patience, major_revision=major_revision,
                                               minor_revision=minor_revision, log_save_path=log_save_path)

        if no_improvement:
            break


main()