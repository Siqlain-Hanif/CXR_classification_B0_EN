import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, models, optimizers
import cv2
import os
import copy
from tensorflow.keras.utils import to_categorical
import albumentations as albu
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import random
from matplotlib.pyplot import plot as plt
import efficientnet.tfkeras as efn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error as mae
from glob import glob
import json
import operator
import multiprocessing
from skimage import morphology
from scipy import ndimage
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2



def alternate_learning_rate(epoch):
    """
        An alternate scheduler for each epoch.
        :param epoch: current epoch
        :return: learning rate for current epoch
        """
    if epoch % 2 == 0:
        return 1e-5
    else:
        return 1e-4


def check_improvement(model, history, patience, major_revision, minor_revision, log_save_path):
    """
    The function checks whether the model has imporved in the last patience number of iterations. If it hasn't
    then the training process is halted.
    The function checks validation accuracy for classification models and validation loss for segmentation models.
    :param model: network type. classification or segmentation
    :param history: validation loss or validation accuracy history
    :param patience: number of epochs that should have passed during which no improvement is observed.
    :param major_revision: major revision of the model.
    :param minor_revision: minor revision of the model.
    :param log_save_path: path where the model is saved.
    :return:
    """
    if model == 'segmentation':
        if len(history) >= 1:
            file = open(os.path.join(log_save_path, 'log-%i-%i.txt' % (major_revision, minor_revision)),
                        'a+')
            temp = np.argsort(history)
            if temp[0] == (len(history) - patience - 1):
                print('#### Validation accuracy no increase for %i epochs: EARLY STOPPING' % patience)
            return True

    elif model == 'classification':
        file = open(os.path.join(log_save_path, 'log-%i-%i.txt' % (major_revision, minor_revision)),
                    'a+')
        if np.argsort(-1 * np.asarray(history))[0] == (len(history) - patience - 1):
            file.write('#### Validation accuracy no increase for %i epochs: EARLY STOPPING' % patience)
        return True

    elif model == 'regression':
        file = open(os.path.join(log_save_path, 'log-%i-%i.txt' % (major_revision, minor_revision)),
                    'a+')
        if np.argsort(-1*np.asarray(history))[0] == (len(history) - patience - 1):
            file.write('#### Validation accuracy no increase for %i epochs: EARLY STOPPING' % patience)
        return True


def combination_loss():
    """
        Implements elo loss as mentioned here: https://www.kaggle.com/c/siim-covid19-detection/discussion/246783
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
        :return:
    """

    def combination_loss_fixed(y_true, y_pred):

        weight_fl = 0.5
        weight_bce = 0.25
        weight_cce = 0.25
        #weight_lsf = 0.7

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        bce_loss = bce(y_true, y_pred)

        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        cce_loss = cce(y_true, y_pred)

        #y_true_ordinal = tf.math.argmax(y_true, axis=1)
        #lsf_loss = lovasz_softmax_flat(y_pred, y_true_ordinal)

        f_loss = focal_loss_fixed(y_true, y_pred)
        weighted_bce = tf.math.multiply(weight_bce, bce_loss)
        weighted_cce = tf.math.multiply(weight_cce, cce_loss)
        #weighted_lsf = tf.math.multiply(weight_lsf, lsf_loss)
        weighted_fl = tf.math.multiply(weight_fl, f_loss)

        return tf.reduce_sum(tf.stack([weighted_bce, weighted_cce, weighted_fl]))

    return combination_loss_fixed


def cosine_annealing_scheduler(epoch):
    """
        A step rate scheduler for setting the learning rate for each epoch.
        :param epoch: current epoch
        :return: learning rate for current epoch
        """
    learning_rate_min = 1e-6
    learning_rate_max = 1e-3
    epochs_per_cycle = 20

    return learning_rate_min + (learning_rate_max - learning_rate_min) * \
           (1 + math.cos(math.pi * (epoch % epochs_per_cycle) / epochs_per_cycle)) / 2


def constant_learning_scheduler(epoch):
    """
    Returns a constant learning rate for every epoch
    :param epoch: current epoch
    :return: learning rate
    """
    return 1e-5



def elo_loss(reduction='sum'):
    """
    Implements elo loss as mentioned here: https://www.kaggle.com/c/siim-covid19-detection/discussion/246783
    y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
    y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
    :return:
    """

    def elo_loss_fixed(y_true, y_pred):

        y_true = tf.clip_by_value(tf.convert_to_tensor(y_true, tf.float32), -1e12, 1e12)
        y_pred = tf.clip_by_value(tf.convert_to_tensor(y_pred, tf.float32), -1e12, 1e12)

        y_true_ordinal = tf.math.argmax(y_true, axis=1)
        class_ids = tf.unique(y_true_ordinal)[0]
        losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for i in class_ids:
            class_predictions = y_pred[:, i]
            class_targets = tf.cast((y_true_ordinal == i), tf.float32)


            mask = tf.multiply(tf.expand_dims(class_targets, 1), (1 - tf.expand_dims(class_targets, 0)))
            class_loss = tf.reduce_mean(tf.boolean_mask(
                -1*tf.sigmoid(tf.expand_dims(class_predictions, 1) - tf.expand_dims(class_predictions, 0)), mask))

            if not(tf.math.is_nan(class_loss)):
                losses = losses.write(losses.size(), class_loss)

        if reduction == 'sum':
            return tf.math.abs(tf.reduce_sum(losses.stack()))
        if reduction == 'mean':
            return tf.math.abs(tf.reduce_mean(losses.stack()))

    return elo_loss_fixed


def focal_loss(gamma=2., alpha=1.):
    """Implements focal loss for the deep learning model"""

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, - tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


def focal_loss_fixed(y_true, y_pred):
    """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, - tf.math.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), 2.))
    fl = tf.multiply(1. , tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)

def generate_image_paths_and_labels(network_type, input_dir, split, read_from_folders, file_name, classes):
    """
    For Classification:
        Function to read images names with paths from multiple folders and create their labels according to
        folder name and then store them in a readable file that is used the next time around.
        The function first checks for a single .csv file with the name file_name that is created on the first pass.
        If that file exists, then that file is directly loaded. If the file doesn't exist, then all the images are read.

    For Segmentation:
        Function looks for a single .txt file with the name file_name containing the paths to all the iamges and their
        corresponding bounding boxes and labels. If no such file is found, then an exception is raised.
        If .txt file exists, then it is read and total samples are split into training and testiong samples according
        to the given split.
    return: Training and testing samples names with paths and labels.
    """
    my_dataframe = None
    if network_type == 'classification':
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
        images = []
        labels = []

        if read_from_folders:
            print('Reading image names from folders...')
            directories = [x[0] for x in os.walk(input_dir)]
            for directory in tqdm(directories, desc='Directories Done'):
                for each_class in classes:
                    if each_class in directory:
                        files = os.listdir(directory)
                        files = [file for file in files if
                                 any(image_extension in file for image_extension in image_extensions)]
                        images.extend([os.path.join(directory, file) for file in files])
                        labels.extend([classes.index(each_class)] * len(files))

            # Saving files as npz array for future reads
            my_dict = {}
            my_dict['id'], my_dict['label'] = images, labels
            my_dataframe = pd.DataFrame(data=my_dict, index=None)
            my_dataframe.to_csv(os.path.join(input_dir, file_name), sep=',', index=False)
            unique_labels = np.unique(labels)
            total_images = 0
            for unique_label in unique_labels:
                total_images += np.count_nonzero(np.where(np.array(labels) == unique_label))
                print('Total Images of Class ' + str(unique_label + 1) + ': ',
                      np.count_nonzero(np.where(np.array(labels) == unique_label)))
            print('Total Images Read: ', total_images)

        else:
            path_and_label_file = os.path.exists(os.path.join(input_dir, file_name))

            if path_and_label_file:
                print('Reading the .csv file')

                my_dataframe = pd.read_csv(os.path.join(input_dir, file_name), sep=',')
                images, labels = my_dataframe['id'].to_list(), my_dataframe['label'].to_list()
                unique_labels = np.unique(labels)
                total_images = 0
                for unique_label in unique_labels:
                    total_images += np.count_nonzero(np.where(np.array(labels) == unique_label))
                    print('Total Images of Class ' + str(unique_label + 1) + ': ',
                          np.count_nonzero(np.where(np.array(labels) == unique_label)))
                print('Total Images Read: ', total_images)

            else:
                print('Relevant .csv files do not exist. Data set not loaded.')


    elif network_type == 'segmentation':
        text_file = os.path.exists(os.path.join(input_dir, file_name))

        if text_file:
            print('Reading the .txt file')
            my_dataframe = pd.read_csv(os.path.join(input_dir, file_name), sep='\n', header=None)
            print('Total Examples: ', len(my_dataframe))
            examples = my_dataframe[0].to_list()

            examples = shuffle(examples)

            training, testing, training_labels, testing_labels = train_test_split(examples, [0] * len(examples),
                                                                                  train_size=split)
            return training, training_labels, testing, testing_labels

        else:
            raise Exception('Text file with segmentation examples does not exist')

    elif network_type == 'regression':
        path_and_label_file = os.path.exists(os.path.join(input_dir, file_name))

        if path_and_label_file:
            print('Reading the .csv file')

            my_dataframe = pd.read_csv(os.path.join(input_dir, file_name), sep=',')
            images, labels = my_dataframe['id'].to_list(), my_dataframe['label'].to_list()
            total_images = len(images)
            print('Total Images Read: ', total_images)

        else:
            print('Relevant .csv files do not exist. Data set not loaded.')

    else:
        raise Exception('No valid network type selected')

    images, labels = shuffle(images, labels)

    training, testing, training_labels, testing_labels = train_test_split(images, labels, train_size=split)

    return training, training_labels, testing, testing_labels


def generate_training_testing_split_mutihead(input_file_path=None, split=0.9):
    """
    Training/Testing split for multihead classifiers
    """
    images, labels = [], []
    my_csv = pd.read_csv(input_file_path)

    for i in range(len(my_csv)):
        my_row = list(my_csv.iloc[i].values)
        images.append(my_row[0])
        labels.append(my_row[1:])

    images, labels = shuffle(images, labels)

    training, testing, training_labels, testing_labels = train_test_split(images, labels, train_size=split)

    return training, training_labels, testing, testing_labels



def learning_rate_scheduler(epoch):
    """
    For training classification models we first use a step rate scheduler for initial 50 epochs and then switch
    to cosine annealing scheduler for the remaining epochs.
    This has the benefit of starting the learning rate slowly and then having the benefits of a cylic scheduler.
    :param epoch: current epoch
    :return: learning rate
    """
    if epoch < 10:
        return 0.000001
    elif epoch < 50:
        return step_rate_scheduler(epoch)
    else:
        return cosine_annealing_scheduler(epoch)


def multihead_labels_to_categorical(labels, multihead_classes):
    """
    Converts the multihead labels to one hot array so that they can be used by a multihead model.
    """
    labels = np.array(labels)
    total_classes = len(multihead_classes)
    one_hot_encoding = []

    for i in range(total_classes):
        one_hot_encoding.append(to_categorical(labels[:,i], num_classes=multihead_classes[i]))

    return one_hot_encoding


def read_annotation_lines(annotation_path, test_size=None, random_seed=5566):
    """
    Function reads the input file containing the path to images and their bounding boxes.
    :param annotation_path: Path to file
    :param test_size: train/test split
    :param random_seed: random seed for recreation
    :return: a list containing the training paths
    """
    with open(annotation_path) as f:
        lines = f.readlines()
    if test_size:
        return train_test_split(lines, test_size=test_size, random_state=random_seed)
    else:
        return lines


def read_data(input_dir, split, visualise, read_from_folders):
    # TODO: Function only works for two classes. Make it generic.
    """
    Function to read images from multiple folders and create their labels according to
    folder name.
    The function first checks for a single .npz file that is created on the first pass. If that file exists, then
    that file is directly loaded. If the file doesn't exist, then all the images are read.
    :return: Training and testing data split according to the parameter passed.
    """

    good_images = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    bad_images = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
    total_good_images = 0
    total_bad_images = 0

    if read_from_folders:
        print('Reading Images from folders...')
        directories = [x[0] for x in os.walk(input_dir)]
        for directory in tqdm(directories, desc='Directories Done'):
            print('Total Images of Class 0 Read: ', total_good_images)
            print('Total Images of Class 1 Read: ', total_bad_images)
            if 'Good' in directory:
                print("Current Directory: ", directory)
                files = os.listdir(directory)
                files = [file for file in files if any(image_extension in file for image_extension in image_extensions)]
                total_good_images += len(files)
                data = np.zeros((len(files), 256, 256, 3), dtype=np.uint8)
                counter = 0
                for file in tqdm(files, desc='Images done'):
                    my_image = cv2.imread(os.path.join(directory, file), -1)
                    data[counter, ...] = cv2.resize(my_image, (256, 256))
                    counter += 1

                good_images = np.concatenate((good_images, data), axis=0)

            elif 'Bad' in directory:
                print("Current Directory: ", directory)
                files = os.listdir(directory)
                files = [file for file in files if any(image_extension in file for image_extension in image_extensions)]
                total_bad_images += len(files)
                data = np.zeros((len(files), 256, 256, 3), dtype=np.uint8)
                counter = 0
                for file in tqdm(files, desc='Images done'):
                    my_image = cv2.imread(os.path.join(directory, file), -1)
                    data[counter, ...] = cv2.resize(my_image, (256, 256))
                    counter += 1

                bad_images = np.concatenate((bad_images, data), axis=0)

        # Saving files as npz array for future reads
        good_images = good_images[1:, ...]
        np.savez_compressed(os.path.join(input_dir, 'good_images.npz'), data=good_images)
        bad_images = bad_images[1:, ...]
        np.savez_compressed(os.path.join(input_dir, 'bad_images.npz'), data=bad_images)
        print('All Images Read and Saved as .npz file.')
        print('Total Images Read: ', np.shape(good_images)[0] + np.shape(bad_images)[0])
        print('Total Images of Class 1: ', np.shape(good_images)[0])
        print('Total Images of Class 2: ', np.shape(bad_images)[0])

    else:
        class_1 = os.path.exists(os.path.join(input_dir, 'good_images.npz'))
        class_2 = os.path.exists(os.path.join(input_dir, 'bad_images.npz'))

        if class_1 and class_2:
            print('Reading Images from .npz files...')
            good_images = np.load(os.path.join(input_dir, 'good_images.npz'))['data']
            bad_images = np.load(os.path.join(input_dir, 'bad_images.npz'))['data']
            print('Total Images Read: ', np.shape(good_images)[0] + np.shape(bad_images)[0])
            print('Total Images of Class 1: ', np.shape(good_images)[0])
            print('Total Images of Class 2: ', np.shape(bad_images)[0])
            total_good_images = np.shape(good_images)[0]
            total_bad_images = np.shape(bad_images)[0]

        else:
            print('Relevant .npz files do not exist. Data set not loaded.')

    total_images = total_good_images + total_bad_images

    good_images = shuffle(good_images)
    bad_images = shuffle(bad_images)

    good_images_training, good_images_testing, _, _ = train_test_split(good_images, [0] * total_good_images,
                                                                       train_size=split)
    bad_images_training, bad_images_testing, _, _ = train_test_split(bad_images, [0] * total_bad_images,
                                                                     train_size=split)

    training_data = np.concatenate((good_images_training, bad_images_training), axis=0)
    training_labels = np.zeros((np.shape(training_data)[0], 1), dtype=np.uint8)
    training_labels[np.shape(good_images_training)[0]:, ...] = 1

    testing_data = np.concatenate((good_images_testing, bad_images_testing), axis=0)
    testing_labels = np.zeros((np.shape(testing_data)[0], 1), dtype=np.uint8)
    testing_labels[np.shape(good_images_testing)[0]:, ...] = 1

    training_data, training_labels = shuffle(training_data, training_labels)
    testing_data, testing_labels = shuffle(testing_data, testing_labels)

    if visualise:
        row, col = 3, 4
        plt.figure(figsize=(20, 15))
        for value in range(row * col):
            plt.subplot(row, col, value + 1)
            index = value + random.randint(0, np.shape(training_data)[0])
            plt.imshow(cv2.cvtColor(training_data[index, ...],
                                    cv2.COLOR_BGR2RGB))
            plt.title('Label: ' + str(training_labels[index]))
        plt.suptitle('12 Randomly Selected Images from Training Data')
        plt.show()

        plt.figure(figsize=(20, 15))
        for value in range(row * col):
            plt.subplot(row, col, value + 1)
            index = value + random.randint(0, np.shape(testing_data)[0])
            plt.imshow(cv2.cvtColor(testing_data[index, ...],
                                    cv2.COLOR_BGR2RGB))
            plt.title('Label: ' + str(testing_labels[index]))
        plt.suptitle('12 Randomly Selected Images from Testing Data')
        plt.show()

    training_labels = to_categorical(training_labels, num_classes=2)
    testing_labels = to_categorical(testing_labels, num_classes=2)

    return training_data, training_labels, testing_data, testing_labels


def step_rate_scheduler(epoch):
    """
    A step rate scheduler for setting the learning rate for each epoch.
    :param epoch: current epoch
    :return: learning rate for current epoch
    """
    learning_rate_start = 1e-5
    learning_rate_max = 1e-3
    learning_rate_rampup_epochs = 5
    Learning_rate_sustain_epoch = 0
    learning_rate_step_decay = 0.75

    if epoch < learning_rate_rampup_epochs:
        lr = (learning_rate_max - learning_rate_start) / learning_rate_rampup_epochs * epoch + learning_rate_start
    elif epoch < learning_rate_rampup_epochs + Learning_rate_sustain_epoch:
        lr = learning_rate_max
    else:
        lr = learning_rate_max * \
             learning_rate_step_decay ** ((epoch - learning_rate_rampup_epochs - Learning_rate_sustain_epoch) // 10)

    return lr


def training_strategy(set_manually=False, which_gpu=0, set_memory_limit = False, memory_limit=7168):
    """
    Determines the number of GPUs in the system and sets the training strategy accordingly.
    If the strategy is to be set manually, then the num_gpus is used for training.
    :param set_manually: manually use a particular GPU
    :param which_gpu: Gpu to use if set manually
    :return: strategy
    """
    num_gpus = None
    if not set_manually:
        num_gpus = tf.config.list_physical_devices('GPU')
        print('Total Available GPUs: ', len(num_gpus))

        if len(num_gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
        else:
            cross_device_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=len(num_gpus))
            strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)

    if set_manually:
        num_gpus = [1]
        # device = '/gpu:%i' % which_gpu
        # device = '/job:localhost/replica:0/task:0/device:GPU:%i' % which_gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[which_gpu], 'GPU')

        if set_memory_limit:
            tf.config.experimental.set_virtual_device_configuration(gpus[which_gpu],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

        strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')


    return len(num_gpus), strategy


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, network_type, validation_data, validation_labels, major_revision, minor_revision,
                 model_save_path,
                 model_log_path, model_name, use_validataion_accuracy, patience=10, map_iou_threshold = 0.5,
                 image_size = (608, 608), classes = [], multihead=False, multihead_classes = None):
        self.model = model
        self.network_type = network_type
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.patience = patience
        self.major_revision = major_revision
        self.minor_revision = minor_revision
        self.model_save_path = model_save_path
        self.model_log_path = model_log_path
        self.model_name = model_name
        self.use_validation_accuracy = use_validataion_accuracy
        self.accuracies = []
        self.map_iou_threshold = map_iou_threshold
        self.image_size = image_size
        self.my_classes = classes
        self.multihead_status = multihead
        self.multihead_classes = multihead_classes

    def on_train_begin(self, logs=None):
        file = open(os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                    'a+')
        file.write('#' * 25)
        file.write('\n')
        file.write('####MODEL NAME %s\n' % (self.model_name))

        #Populating accuracires list with the validation accuracies of previous epochs
        file = open(os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                    'r')
        data = file.readlines()
        data = [x for x in data if 'VALIDATION ACCURACY' in x]
        data = [x.split('=')[-1] for x in data]
        data = [x.split(' ')[0] for x in data]
        data = [float(x) for x in data]
        self.accuracies = data



    def on_epoch_end(self, epoch, logs={}):
        """
        We will calculate the validation accuracy at the end of each epoch and then we will log that accuracy to a file.
        :param epoch: current epoch
        :param logs:
        :return: None
        """
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Epoch %d: Invalid loss, terminating training' % epoch)
                file = open(
                    os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                    'a+')
                file.write('Epoch %d: Invalid loss, terminating training' % epoch)
                self.model.stop_training = True

        if self.network_type == 'classification':
            validation_accuracy, my_confusion_matrix = 0, None
            my_confusion_matrices = []
            if self.use_validation_accuracy:
                predictions = self.model.predict(self.validation_data)

                if not self.multihead_status:
                    predictions = np.argmax(predictions, axis=1)
                    validation_accuracy = accuracy_score(np.argmax(self.validation_labels[0:len(predictions), :], axis=1),
                                                         predictions, normalize=True)
                    my_confusion_matrix = confusion_matrix(np.argmax(self.validation_labels[0:len(predictions), :], axis=1),
                                                           predictions)

                elif self.multihead_status:

                    for i in range(len(self.multihead_classes)):
                        temp_validation_accuracy = accuracy_score(
                            np.argmax(self.validation_labels[i][0:len(predictions[0]), :], axis=1),
                            np.argmax(predictions[i], axis=1),
                            normalize=True)
                        my_confusion_matrices.append(confusion_matrix(
                            np.argmax(self.validation_labels[len(self.multihead_classes)-1-i][0:len(predictions[0]), :], axis=1),
                            np.argmax(predictions[i][0:len(predictions[0]), :],
                                                                                   axis=1)))
                        validation_accuracy += temp_validation_accuracy

                    validation_accuracy = validation_accuracy/len(self.multihead_classes)

                # Logging to file
                file = open(
                    os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                    'a+')
                file.write('#' * 25)
                file.write('\n')
                file.write('####EPOCH %i\n' % (epoch + 1))
                file.write('#### VALIDATION ACCURACY =%.5f \n' % validation_accuracy)
                file.write('####CONFUSION MATRIX####')
                file.write('\n')
                if not self.multihead_status:
                    file.write(str(my_confusion_matrix))
                elif self.multihead_status:
                    for i in range(len(my_confusion_matrices)):
                        file.write(str(my_confusion_matrices[i]))

                print('\n')
                print('#' * 25)
                print('#### EPOCH %i' % (epoch + 1))
                print('#### VALIDATION ACCURACY =%.5f' % validation_accuracy)
                print('####CONFUSION MATRIX####')
                if not self.multihead_status:
                    print(str(my_confusion_matrix))
                elif self.multihead_status:
                    for i in range(len(my_confusion_matrices)):
                        print(str(my_confusion_matrices[i]))
                print('#' * 25)

                self.accuracies.append(validation_accuracy)
                x = np.asarray(self.accuracies)
                if np.argsort(-x)[0] == (len(x) - self.patience - 1):
                    print('#### Validation accuracy no increase for %i epochs: EARLY STOPPING' % self.patience)
                    file.write('#### Validation accuracy no increase for %i epochs: EARLY STOPPING\n' % self.patience)
                    self.model.stop_training = True

                if (validation_accuracy > 0.000) & (validation_accuracy >= np.nanmax(self.accuracies)):
                    print('#### Saving new best...')
                    file.write('#### Saving new best...\n')
                    self.model.save_weights(
                        os.path.join(self.model_save_path, 'm%i-%i.h5' % (self.major_revision, self.minor_revision)))

                file.close()

            else:
                # Logging to file
                file = open(
                    os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                    'a+')
                file.write('#' * 25)
                file.write('\n')
                file.write('####EPOCH %i\n' % (epoch + 1))
                file.write('#### TRAINING ACCURACY =%.5f \n' % logs.get('categorical_accuracy'))

                self.accuracies.append(logs.get('categorical_accuracy'))
                x = np.asarray(self.accuracies)
                if np.argsort(-x)[0] == (len(x) - self.patience - 1):
                    print('#### Validation accuracy no increase for %i epochs: EARLY STOPPING' % self.patience)
                    file.write('#### Validation accuracy no increase for %i epochs: EARLY STOPPING\n' % self.patience)
                    self.model.stop_training = True

                if (logs.get('categorical_accuracy') > 0.000) & (logs.get('categorical_accuracy') >
                                                                 np.nanmax(self.accuracies)):
                    print('#### Saving new best...')
                    file.write('#### Saving new best...\n')
                    self.model.save_weights(
                        os.path.join(self.model_save_path, 'm%i-%i.h5' % (self.major_revision, self.minor_revision)))

                file.close()

        elif self.network_type == 'regression':
            if self.use_validation_accuracy:
                predictions = self.model.predict(self.validation_data)[:,0]

                mean_absolute_error = mae(self.validation_labels[0:len(predictions)], predictions)

                # Logging to file
                file = open(
                    os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                    'a+')
                file.write('#' * 25)
                file.write('\n')
                file.write('####EPOCH %i\n' % (epoch + 1))
                file.write('#### MEAN ABSOLUTE ERROR =%.5f \n' % mean_absolute_error)

                print('\n')
                print('#' * 25)
                print('#### EPOCH %i' % (epoch + 1))
                print('#### MEAN ABSOLUTE ERROR =%.5f' % mean_absolute_error)
                print('#' * 25)

                self.accuracies.append(mean_absolute_error)
                x = np.asarray(self.accuracies)
                if np.argsort(-x)[0] == (len(x) - self.patience - 1):
                    print('#### Validation accuracy no increase for %i epochs: EARLY STOPPING' % self.patience)
                    file.write('#### Validation accuracy no increase for %i epochs: EARLY STOPPING\n' % self.patience)
                    self.model.stop_training = True

                if (mean_absolute_error > 0.000) & (mean_absolute_error <= np.nanmin(self.accuracies)):
                    print('#### Saving new best...')
                    file.write('#### Saving new best...\n')
                    self.model.save_weights(
                        os.path.join(self.model_save_path, 'm%i-%i.h5' % (self.major_revision, self.minor_revision)))

                file.close()

            else:
                # Logging to file
                file = open(
                    os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                    'a+')
                file.write('#' * 25)
                file.write('\n')
                file.write('####EPOCH %i\n' % (epoch + 1))
                file.write('#### TRAINING ACCURACY =%.5f \n' % logs.get('categorical_accuracy'))

                self.accuracies.append(logs.get('mae'))
                x = np.asarray(self.accuracies)
                if np.argsort(-x)[0] == (len(x) - self.patience - 1):
                    print('#### Validation accuracy no increase for %i epochs: EARLY STOPPING' % self.patience)
                    file.write('#### Validation accuracy no increase for %i epochs: EARLY STOPPING\n' % self.patience)
                    self.model.stop_training = True

                if (logs.get('mae') > 0.000) & (logs.get('mae') <=
                                                                 np.nanmin(self.accuracies)):
                    print('#### Saving new best...')
                    file.write('#### Saving new best...\n')
                    self.model.save_weights(
                        os.path.join(self.model_save_path, 'm%i-%i.h5' % (self.major_revision, self.minor_revision)))

                file.close()



        elif self.network_type == 'segmentation':
            my_mean_average_precision = calculate_map(validation_samples_list = self.validation_data,
                                                      yolo_model =WorkAround.current_model , iou_threshold=self.map_iou_threshold,
                                                      model_image_size = self.image_size,
                                                      model_classes = self.my_classes)
            file = open(os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                        'a+')
            file.write('#' * 25)
            file.write('\n')
            file.write('####EPOCH %i\n' % (epoch + 1))
            file.write('#### TRAINING LOSS =%.5f \n' % logs.get('loss'))
            file.write('#### VALIDATION LOSS =%.5f \n' % logs.get('val_loss'))
            file.write('#### mAP =%.5f \n' % my_mean_average_precision)

            print()
            print('mAP at ', self.map_iou_threshold, ' is: ', my_mean_average_precision)
            temp_monitored_value = logs.get('val_loss')
            if not math.isnan(temp_monitored_value):
                self.accuracies.append(temp_monitored_value)
            else:
                self.accuracies.append(1000000)

            x = np.asarray(self.accuracies)
            if len(x) > 1:
                if np.argsort(x)[0] == (len(x) - self.patience - 1):
                    print('#### Validation accuracy no increase for %i epochs: EARLY STOPPING' % self.patience)
                    file.write('#### Validation accuracy no increase for %i epochs: EARLY STOPPING\n' % self.patience)
                    self.model.stop_training = True

                if (temp_monitored_value > 0.000) & (temp_monitored_value < np.min(self.accuracies[:-1])):
                    print('#### Saving new best...')
                    file.write('#### Saving new best...\n')
                    WorkAround.save_inference_model(path=self.model_save_path, major_revision=self.major_revision,
                                                    minor_revision=self.minor_revision)

            file.close()

        else:
            raise Exception('No valid network type selected')


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data Generator for generating a batch in real-time to be fed to a model.
    Ideally, it should work both for classification networks and segmentation/localisation networks.
    """

    def __init__(self, list_ids, labels, return_labels=True, batch_size=32, dim=(256, 256, 3), n_classes=2,
                 shuffle_data=True, augment_data=False, visualise=None, network_type='classification',
                 anchors=None, max_boxes=100, train_on_cloud=False, major_revision=0,
                 minor_revision=0, model_log_path='./', augmentations=None, use_vasculature=False,
                 vasculature_directory=None, vasculature_extension='.png', multihead=False,
                 multihead_classes = []):
        self.list_ids = list_ids  # same as annotations line. Basically, the list of the images that we will read.
        self.labels = labels
        self.return_labels = return_labels
        self.batch_size = batch_size
        self.dim = dim  # image size
        self.n_classes = n_classes
        self.augment_data = augment_data
        self.shuffle_data = shuffle_data
        self.visualise = visualise
        self.network_type = network_type
        self.on_epoch_end()
        self.anchors = np.array(yolo_config['anchors']).reshape((9, 2))  # YOLO anchors
        self.max_boxes = max_boxes
        self.train_on_cloud = train_on_cloud
        self.major_revision = major_revision
        self.minor_revision = minor_revision
        self.model_log_path = model_log_path
        self.valid_augmentations = augmentations
        self.use_vasculature = use_vasculature
        self.vasculature_directory = vasculature_directory
        self.vascuture_extension = vasculature_extension
        self.multihead = multihead
        self.multihead_classes = multihead_classes

    def on_epoch_end(self):
        """
        This function will shuffle the list of images on each epoch end
        :return:
        """
        self.indices = np.arange(len(self.list_ids))
        if self.shuffle_data:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_ids_temp, list_labels_temp=None):
        """
        Generates data for a single batch containing batch_size samples as X : (n_samples, *dim) and y as (n_samples,
        num_classes) in case of classification network
        Generates data for a single batch containing batch_size samples as X: (n_samples, *dim) and y as (n_samples,
        max_boxes, 5) for segmentation network.
        """
        if not self.multihead:
            if self.network_type == 'classification':
                # Initialization
                X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2]), dtype=np.uint8)
                y = np.empty(self.batch_size, dtype=np.uint8)

                # Generate data
                for i, ID in enumerate(list_ids_temp):
                    # Store sample
                    #X[i, ...] = cv2.resize(cv2.imread(ID, -1), self.dim[:2])
                    #print(ID)
                    my_image = cv2.resize(cv2.imread(ID, -1), (self.dim[0], self.dim[1]))
                    my_shape = np.shape(my_image)
                    if len(my_shape) == 2:
                        X[i, :,:,0] = my_image
                        X[i, :,:,1] = X[i, :,:,0]
                        X[i, :,:,2] = X[i, :,:,0]
                    elif len(my_shape) == 3:
                        X[i, ...] = my_image

                    # Store class
                    y[i] = list_labels_temp[i]

                if self.augment_data:
                    X = self.__augment_batch(X, y)

                return X, to_categorical(y, num_classes=self.n_classes)
                #return X, y

            elif self.network_type == 'regression':
                # Initialization
                X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2]), dtype=np.uint8)
                y = np.empty(self.batch_size, dtype=np.float32)

                # Generate data
                for i, ID in enumerate(list_ids_temp):
                    # Store sample
                    X[i, ...] = cv2.resize(cv2.imread(ID, -1), self.dim[:2])

                    # Store class
                    y[i] = list_labels_temp[i]

                if self.augment_data:
                    X = self.__augment_batch(X, y)

                return X, y

            elif self.network_type == 'segmentation':
                X = np.empty((len(list_ids_temp), self.dim[0], self.dim[1], self.dim[2]), dtype=np.float32)
                y_bbox = np.empty((len(list_ids_temp), self.max_boxes, 5), dtype=np.float32)

                valid_sample = 0
                for i, ID in enumerate(list_ids_temp):
                    valid, img_data, box_data = self.get_data(ID)
                    if not valid:
                        X[valid_sample] = img_data
                        y_bbox[valid_sample] = box_data
                        valid_sample += 1

                X = X[:valid_sample]
                y_bbox = y_bbox[:valid_sample]

                if self.augment_data:
                    X, y_bbox = self.__augment_batch(X, y_bbox)

                y_tensor, y_true_boxes_xywh = self.preprocess_true_boxes(y_bbox, self.dim[:2], self.anchors,
                                                                         self.n_classes)

                return X, y_tensor, y_true_boxes_xywh

            else:
                raise Exception('Neither type of network selected')

        elif self.multihead:
            if self.network_type == 'classification':
                # Initialization
                X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2]), dtype=np.uint8)
                y_grades = np.empty(self.batch_size, dtype=np.uint8)
                y_artifacts = np.empty(self.batch_size, dtype=np.uint8)
                y_exposure = np.empty(self.batch_size, dtype=np.uint8)
                y_focus = np.empty(self.batch_size, dtype=np.uint8)

                # Generate data
                for i, ID in enumerate(list_ids_temp):
                    # Store sample
                    #X[i, ...] = cv2.resize(cv2.imread(ID, -1), self.dim[:2])
                    #print(ID)
                    my_image = cv2.resize(cv2.imread(ID, -1), (self.dim[0], self.dim[1]))
                    my_shape = np.shape(my_image)
                    if len(my_shape) == 2:
                        X[i, :,:,0] = my_image
                        X[i, :,:,1] = X[i, :,:,0]
                        X[i, :,:,2] = X[i, :,:,0]
                    elif len(my_shape) == 3:
                        X[i, ...] = my_image

                    # Store class
                    y_grades[i] = list_labels_temp[i][-1]
                    y_artifacts[i] = list_labels_temp[i][0]
                    y_exposure[i] = list_labels_temp[i][1]
                    y_focus[i] = list_labels_temp[i][2]

                if self.augment_data:
                    X = self.__augment_batch(X, None)

                return X, to_categorical(y_grades, num_classes=self.multihead_classes[-1]),\
                       to_categorical(y_artifacts, num_classes=self.multihead_classes[0]),\
                       to_categorical(y_exposure, num_classes=self.multihead_classes[1]),\
                       to_categorical(y_exposure, num_classes=self.multihead_classes[2])
                #return X, y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indices]

        if self.train_on_cloud:
            grdive_path = '/content/drive/My Drive/Yolo_Data/'
            list_ids_temp = [(grdive_path + i.split('/')[-1]).replace('\\', '/') for i in list_ids_temp]

        if self.network_type == 'classification':
            list_labels_temp = [self.labels[k] for k in indices]
            X, y, y_1, y_2, y_3, y_4  = None, None, None, None, None, None
            # Generate data
            if self.multihead:
                X, y_1, y_2, y_3, y_4 = self.__data_generation(list_ids_temp, list_labels_temp)
            elif not self.multihead:
                X, y = self.__data_generation(list_ids_temp, list_labels_temp)

            if self.multihead:
                if self.return_labels:
                    return X, (y_1, y_2, y_3, y_4)
                else:
                    return X
            elif not self.multihead:
                if self.return_labels:
                    return X, y
                else:
                    return X

        elif self.network_type == 'regression':
            list_labels_temp = [self.labels[k] for k in indices]
            # Generate data
            X, y = self.__data_generation(list_ids_temp, list_labels_temp)

            if self.return_labels:
                return X, y
            else:
                return X

        elif self.network_type == 'segmentation':
            X, y_tensor, y_bbox = self.__data_generation(list_ids_temp=list_ids_temp)

            return [X, *y_tensor, y_bbox], np.zeros(len(list_ids_temp))

        else:
            raise Exception('Neither type of network selected.')

    def get_data(self, annotation_line):
        line = annotation_line.split()
        if self.train_on_cloud:
            # print(line[0] + ' ' + line[1])
            img_path = line[0] + ' ' + line[1]
        else:
            img_path = line[0]

        if self.use_vasculature:
            img = cv2.imread(img_path, -1)
            img_name = img_path.split('/')[-1]
            # print(img_name)
            img_extension = '.' + img_name.split('.')[-1]
            # print(img_extension)
            vascultaure_name = img_name.replace(img_extension, self.vascuture_extension)
            # print(vascultaure_name)
            vasculture_path = os.path.join(self.vasculature_directory, vascultaure_name)
            # print(vasculture_path)
            vasculture = cv2.imread(vasculture_path, -1)
            img[:, :, 0] = vasculture
            img = img[:, :, ::-1]
        else:
            #print(img_path)
            img = cv2.imread(img_path, -1)
            my_shape = np.shape(img)
            if len(my_shape) < 3:
                img = np.stack((img,) * 3, axis=-1)

            if img is None:
                print(img_path)
            else:
                img = img[:, :, ::-1]

        ih, iw = img.shape[:2]
        h, w, c = self.dim
        if self.train_on_cloud:
            boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[2:]],
                             dtype=np.float32)  # x1y1x2y2
        else:
            boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]],
                             dtype=np.float32)  # x1y1x2y2
        scale_w, scale_h = w / iw, h / ih
        img = cv2.resize(img, (w, h))
        image_data = np.array(img) / 255.

        # correct boxes coordinates
        box_data = np.zeros((self.max_boxes, 5))
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes = boxes[:self.max_boxes]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w  # + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h  # + dy
            # boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            # boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            box_data[:len(boxes)] = boxes

        if np.isnan(image_data).any():
            file = open(os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                        'a+')
            file.write('\n')
            file.write('NAN value encountered in image data. Sample will be discarded.')

        if np.isnan(box_data).any():
            file = open(os.path.join(self.model_log_path, 'log-%i-%i.txt' % (self.major_revision, self.minor_revision)),
                        'a+')
            file.write('\n')
            file.write('NAN value encountered in annotation data. Sample will be discarded.')

        return (np.isnan(image_data).any() or np.isnan(box_data).any()), \
               image_data, box_data
        #return (np.isnan(image_data).any() or np.isnan(box_data).any()), \
        #       np.stack((image_data,)*3, axis=-1), box_data

    def __random_transform(self, img, annotation):
        """
        We generate the augmentations that can be applied to a set of images in a dataset
        The valid_augmentations list depends on the data that is being supplied.
        """
        if self.network_type == 'classification':

            composition = albu.Compose(transforms=self.valid_augmentations[:-1], p=self.valid_augmentations[-1])

            transformed_data = composition(image=img)

            if self.visualise:
                temp_image = np.float32(transformed_data['image'])
                img = np.float32(img)
                horizontally_stacked = np.hstack((img, temp_image))
                cv2.imshow('Transformed Image', np.uint8(horizontally_stacked))
                cv2.waitKey()

            return transformed_data['image']

        elif self.network_type == 'regression':
            composition = albu.Compose(transforms=self.valid_augmentations[:-1], p=self.valid_augmentations[-1])

            transformed_data = composition(image=img)

            if self.visualise:
                temp_image = np.float32(transformed_data['image'])
                img = np.float32(img)
                horizontally_stacked = np.hstack((img, temp_image))
                cv2.imshow('Transformed Image', np.uint8(horizontally_stacked))
                cv2.waitKey()

            return transformed_data['image']


        elif self.network_type == 'segmentation':

            temp, class_labels = [], []
            for i in range(np.shape(annotation)[0]):
                if sum(annotation[i, ...]) > 0:
                    temp.append(list(annotation[i, ...][0:-1]))
                    class_labels.append(self.labels[int(annotation[i, -1])])

            composition = albu.Compose(transforms=self.valid_augmentations[:-1],
                                       bbox_params=albu.BboxParams(format='pascal_voc', min_visibility=0.7,
                                                                   label_fields=['class_labels']),
                                       p=self.valid_augmentations[-1])
            #print(composition.transforms)
            transformed_data = composition(image=np.uint8(img * 255), bboxes=temp, class_labels=class_labels)

            if self.visualise:
                temp_image = np.float32(transformed_data['image'][:, :, ::-1])
                img = np.float32(img * 255)

                for i in range(len(transformed_data['class_labels'])):
                    start_point = (int(transformed_data['bboxes'][i][0]), int(transformed_data['bboxes'][i][1]))
                    end_point = (int(transformed_data['bboxes'][i][2]), int(transformed_data['bboxes'][i][3]))
                    temp_image = cv2.rectangle(temp_image, start_point, end_point, (0, 0, 255), 2)

                    start_point = (int(annotation[i][0]), int(annotation[i][1]))
                    end_point = (int(annotation[i][2]), int(annotation[i][3]))
                    img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)

                horizontally_stacked = np.hstack((img[:, :, ::-1], temp_image))
                cv2.imshow('Transformed Image', np.uint8(horizontally_stacked))
                cv2.waitKey()

            temp = transformed_data['bboxes']
            class_labels = transformed_data['class_labels']
            for i in range(len(temp)):
                annotation[i, 0:4] = temp[i]
                annotation[i, -1] = float(self.labels.index(class_labels[i]))

            return np.array(transformed_data['image']) / 255., annotation

        else:
            raise Exception('No valid type of network selected')

    def __augment_batch(self, img_batch, label_batch):

        if self.network_type == 'classification':
            for i in range(img_batch.shape[0]):
                img_batch[i, ...] = self.__random_transform(img_batch[i, ...], self.valid_augmentations)
            return img_batch

        elif self.network_type == 'regression':
            for i in range(img_batch.shape[0]):
                img_batch[i, ...] = self.__random_transform(img_batch[i, ...], self.valid_augmentations)
            return img_batch

        elif self.network_type == 'segmentation':
            for i in range(img_batch.shape[0]):
                img_batch[i, ...], label_batch[i, ...] = self.__random_transform(img_batch[i, ...], label_batch[i, ...])
            return img_batch, label_batch
        else:
            raise Exception('No valid type of network selected')

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        # ToDo: Dry run this code at least once.
        '''Preprocess true boxes to training input format

        Parameters
        ----------
        true_boxes: array, shape=(batch_size, max boxes per img, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), (9, wh)
        num_classes: int

        Returns
        -------
        y_true: list of array, shape like yolo_outputs, xywh are relative value

        '''

        num_stages = 3  # default setting for yolo, tiny yolo will be 2
        anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        bbox_per_grid = 3
        true_boxes = np.array(true_boxes, dtype='float32')
        true_boxes_abs = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        true_boxes_xy = (true_boxes_abs[..., 0:2] + true_boxes_abs[..., 2:4]) // 2  # (100, 2)
        true_boxes_wh = true_boxes_abs[..., 2:4] - true_boxes_abs[..., 0:2]  # (100, 2)

        # Normalize x,y,w, h, relative to img size -> (0~1)
        true_boxes[..., 0:2] = true_boxes_xy / input_shape[::-1]  # xy
        true_boxes[..., 2:4] = true_boxes_wh / input_shape[::-1]  # wh

        bs = true_boxes.shape[0]
        grid_sizes = [input_shape // {0: 8, 1: 16, 2: 32}[stage] for stage in range(num_stages)]
        y_true = [np.zeros((bs,
                            grid_sizes[s][0],
                            grid_sizes[s][1],
                            bbox_per_grid,
                            5 + num_classes), dtype='float32')
                  for s in range(num_stages)]
        # [(?, 52, 52, 3, 5+num_classes) (?, 26, 26, 3, 5+num_classes)  (?, 13, 13, 3, 5+num_classes) ]
        y_true_boxes_xywh = np.concatenate((true_boxes_xy, true_boxes_wh), axis=-1)
        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)  # (1, 9 , 2)
        anchor_maxes = anchors / 2.  # (1, 9 , 2)
        anchor_mins = -anchor_maxes  # (1, 9 , 2)
        valid_mask = true_boxes_wh[..., 0] > 0  # (1, 100)

        for batch_idx in range(bs):
            # Discard zero rows.
            wh = true_boxes_wh[batch_idx, valid_mask[batch_idx]]  # (# of bbox, 2)
            num_boxes = len(wh)
            if num_boxes == 0: continue
            wh = np.expand_dims(wh, -2)  # (# of bbox, 1, 2)
            box_maxes = wh / 2.  # (# of bbox, 1, 2)
            box_mins = -box_maxes  # (# of bbox, 1, 2)

            # Compute IoU between each anchors and true boxes for responsibility assignment
            intersect_mins = np.maximum(box_mins, anchor_mins)  # (# of bbox, 9, 2)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = np.prod(intersect_wh, axis=-1)  # (9,)
            box_area = wh[..., 0] * wh[..., 1]  # (# of bbox, 1)
            anchor_area = anchors[..., 0] * anchors[..., 1]  # (1, 9)
            iou = intersect_area / (box_area + anchor_area - intersect_area)  # (# of bbox, 9)

            # Find best anchor for each true box
            best_anchors = np.argmax(iou, axis=-1)  # (# of bbox,)
            for box_idx in range(num_boxes):
                best_anchor = best_anchors[box_idx]
                for stage in range(num_stages):
                    if best_anchor in anchor_mask[stage]:
                        x_offset = true_boxes[batch_idx, box_idx, 0] * grid_sizes[stage][1]
                        y_offset = true_boxes[batch_idx, box_idx, 1] * grid_sizes[stage][0]
                        # Grid Index
                        grid_col = np.floor(x_offset).astype('int32')
                        grid_row = np.floor(y_offset).astype('int32')
                        anchor_idx = anchor_mask[stage].index(best_anchor)
                        class_idx = true_boxes[batch_idx, box_idx, 4].astype('int32')
                        y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, :2] = true_boxes_xy[batch_idx, box_idx,
                                                                                       :]  # abs xy
                        y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 2:4] = true_boxes_wh[batch_idx,
                                                                                        box_idx,
                                                                                        :]  # abs wh
                        y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 4] = 1  # confidence

                        y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 5 + class_idx] = 1  # one-hot encoding

        return y_true, y_true_boxes_xywh

