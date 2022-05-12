import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras



#Creating full size U-net model with input size (512,512,1) and output size (512, 512, 1)
"""
inputs = Input((512,512,3))
#Size = (256,256,64)
conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv1)
conv1= BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#Size = (128,128,128)
conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#Size = (64,64,256)
conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#Size = (32,32,512)
conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
conv4 = BatchNormalization()(conv4)
drop4 = Dropout(0.25)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

#Size = (32,32,1024)
conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool4)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)
conv5 = BatchNormalization()(conv5)
drop5 = Dropout(0.25)(conv5)

#Size = (64,64,512)
up6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2, 2))(drop5))
up6 = BatchNormalization()(up6)
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merge6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv6)
conv6 = BatchNormalization()(conv6)

#Size = (128,128,256)
up7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2, 2))(conv6))
up7 = BatchNormalization()(up7)
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merge7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv7)
conv7 = BatchNormalization()(conv7)

#Size = (256,256,128)
up8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2, 2))(conv7))
up8 = BatchNormalization()(up8)
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merge8)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv8)
conv8 = BatchNormalization()(conv8)

#Size = (512,512,2)
up9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2, 2))(conv8))
up9 = BatchNormalization()(up9)
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merge9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)
conv9 = BatchNormalization()(conv9)


#Size = (512,512,1)
conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

my_model = Model(inputs=inputs, outputs=conv10)

my_model.compile(optimizer=Adam(lr=1e-4), loss=BinaryCrossentropy(), metrics=['binary_accuracy'])

my_model.summary()

my_model.save('C:/Users/Asad/Desktop/u_net_3channels.h5')
"""

#Creating full size LWnet model with input size (512,512,1) and output size (512, 512, 1)
"""
inputs = Input((512,512,1))

#Size = (256,256,8)
conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
batch1 = BatchNormalization()(conv1)
drop1 = Dropout(0.25)(batch1)
pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)

#Size = (128,128,16)
conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
batch2 = BatchNormalization()(conv2)
drop2 = Dropout(0.25)(batch2)
pool2 = MaxPooling2D(pool_size=(2, 2))(batch2)

#Size = (64,64,32)
conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
batch3 = BatchNormalization()(conv3)
drop3 = Dropout(0.25)(batch3)
pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
###############Downsampling ends here#############################


##############Upsampling starts here##############################
#Size = (128,128,16)
up4 = (UpSampling2D(size=(2, 2))(pool3))
up4 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
up4 = BatchNormalization()(up4)
merge4 = concatenate([drop3, up4], axis=3)
conv4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
conv4 = BatchNormalization()(conv4)

#Size = (256,256,8)
up7 = (UpSampling2D(size=(2, 2))(conv4))
up7 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
up7 = BatchNormalization()(up7)
merge7 = concatenate([drop2, up7], axis=3)
conv7 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
conv7 = BatchNormalization()(conv7)

#Size = (512,512,1)
up8 = (UpSampling2D(size=(2, 2))(conv7))
up8 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
up8 = BatchNormalization()(up8)
merge8 = concatenate([drop1, up8], axis=3)
conv8 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
conv8 = BatchNormalization()(conv8)

conv10 = Conv2D(1, 1, activation='sigmoid')(conv8)
#######################################################################################################################
#######################################################################################################################


# attn_map = concatenate([conv10, inputs], axis=3)
# 
# #Size = (256,256,8)
# conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(attn_map)
# conv11 = BatchNormalization()(conv11)
# conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
# batch11 = BatchNormalization()(conv11)
# drop11 = Dropout(0.25)(batch11)
# pool11 = MaxPooling2D(pool_size=(2, 2))(batch11)
# 
# #Size = (128,128,16)
# conv12 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool11)
# conv12 = BatchNormalization()(conv12)
# conv12 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
# batch12 = BatchNormalization()(conv12)
# drop12 = Dropout(0.25)(batch12)
# pool12 = MaxPooling2D(pool_size=(2, 2))(batch12)
# 
# #Size = (64,64,32)
# conv13 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool12)
# conv13 = BatchNormalization()(conv13)
# conv13 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
# batch13 = BatchNormalization()(conv13)
# drop13 = Dropout(0.25)(batch13)
# pool13 = MaxPooling2D(pool_size=(2, 2))(drop13)
# ###############Downsampling ends here#############################
# 
# 
# ##############Upsampling starts here##############################
# #Size = (128,128,16)
# up14 = (UpSampling2D(size=(2, 2))(pool13))
# up14 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up14)
# up4 = BatchNormalization()(up14)
# merge14 = concatenate([drop13, up14], axis=3)
# conv14 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge14)
# conv14 = BatchNormalization()(conv14)
# conv14 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
# conv14 = BatchNormalization()(conv14)
# 
# #Size = (256,256,8)
# up17 = (UpSampling2D(size=(2, 2))(conv14))
# up17 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up17)
# up17 = BatchNormalization()(up17)
# merge17 = concatenate([drop12, up17], axis=3)
# conv17 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge17)
# conv17 = BatchNormalization()(conv17)
# conv17 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv17)
# conv17 = BatchNormalization()(conv17)
# 
# #Size = (512,512,1)
# up18 = (UpSampling2D(size=(2, 2))(conv17))
# up18 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up18)
# up18 = BatchNormalization()(up18)
# merge18 = concatenate([drop11, up18], axis=3)
# conv18 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge18)
# conv18 = BatchNormalization()(conv18)
# conv18 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv18)
# conv18 = BatchNormalization()(conv18)
# 
# conv20 = Conv2D(1, 1, activation='sigmoid')(conv18)
# 
# 
# my_model = Model(inputs=inputs, outputs=[conv10, conv20])
my_model = Model(inputs=inputs, outputs=conv10)

# my_model.compile(optimizer=Adam(lr=1e-4), loss=[BinaryCrossentropy(), BinaryCrossentropy()],
#                  loss_weights=[0.5,0.5], metrics=['binary_accuracy'])

my_model.compile(optimizer=Adam(lr=1e-4), loss=BinaryCrossentropy(), metrics=['binary_accuracy'])

my_model.summary()

my_model.save('C:/Users/Asad/Desktop/lunet.h5')
#my_model.save_weights('C:/Users/Asad/Desktop/lwnet_weights.h5')
"""