# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:33:56 2017

@author: m131199
"""

from __future__ import print_function
from __future__ import absolute_import
import warnings

import os
import numpy as np
from keras import backend as K
from keras.models import Model
from keras import layers
from keras.layers import (Input, concatenate, Conv2D, Lambda, 
                          MaxPooling2D, Conv2DTranspose, Activation, 
                          Dropout, Flatten, Reshape, Cropping3D,
                          Dense, ZeroPadding2D, AveragePooling2D,
                          GlobalAveragePooling2D, GlobalMaxPooling2D,
                          BatchNormalization,UpSampling2D, Conv3D, 
                          MaxPooling3D, UpSampling3D,LeakyReLU)
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.utils import np_utils, generic_utils, layer_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
#from resnetIdentityShortcuts import identity_block
#from resnetConvBlock import conv_block
#from inceptionConvBlock import conv2d_bn
#from keras import regularizers

def Unet(image_shape,lr=1e-04, decay=1e-08, sw=None,initializer = 'glorot_uniform', nb_classes=2):
#    initializer = 'glorot_uniform'
#    initializer = 'he_normal'
    inputs = Input(shape = image_shape)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(inputs)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool4)
    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_initializer = initializer)(drop5), drop4], axis=3)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(up6)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)    
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(up7)
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv7)
    
    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer = initializer)(conv7), conv2], axis=3)    
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(up8)
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv8)
    
    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer = initializer)(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(up9)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
#    conv9 = Conv2D(2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(nb_classes, (1,1), activation = 'relu')(conv9)    
#    out = Activation('softmax')(conv10)
    
    model = Model(inputs=[inputs], outputs=[conv10])
#    model.compile(optimizer = Adam(lr = lr, decay=decay), loss = 'categorical_crossentropy', metrics = ['accuracy'],sample_weight_mode=sw)

    return model



def Unet3D(input_shape, ds=1, pool_size=(2, 2, 2), n_labels=2,
                  initial_learning_rate=1e-4, deconvolution=False):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param ds: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    conv1 = Conv3D(int(64/ds), (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(int(64/ds), (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

    conv2 = Conv3D(int(128/ds), (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(int(128/ds), (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

    conv3 = Conv3D(int(256/ds), (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(int(256/ds), (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

    conv4 = Conv3D(int(512/ds), (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(int(512/ds), (3, 3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)

#    conv5 = Conv3D(int(1024/ds), (3, 3, 3), activation='relu', padding='same')(pool4)
#    conv5 = Conv3D(int(1024/ds), (3, 3, 3), activation='relu', padding='same')(conv5)
#    drop5 = Dropout(0.5)(conv5)
    
    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
                     nb_filters=int(256/ds), image_shape=input_shape)(drop4)
    up5 = concatenate([up5, conv3], axis=4)
    conv5 = Conv3D(int(256/ds), (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(int(256/ds), (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=int(256/ds), image_shape=input_shape)(conv5)
    up6 = concatenate([up6, conv2], axis=4)
    conv6 = Conv3D(int(128/ds), (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(128/ds), (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=int(128/ds), image_shape=input_shape)(conv6)
    up7 = concatenate([up7, conv1], axis=4)
    conv7 = Conv3D(int(64/ds), (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(64/ds), (3, 3, 3), activation='relu', padding='same')(conv7)

#    up8 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
#                     nb_filters=int(128/ds), image_shape=input_shape)(conv7)
#    up8 = concatenate([up8, conv1], axis=4)
#    conv8 = Conv3D(int(64/ds), (3, 3, 3), activation='relu', padding='same')(up8)
#    conv8 = Conv3D(int(64/ds), (3, 3, 3), activation='relu', padding='same')(conv8)
    
    conv8 = Conv3D(n_labels, (1, 1, 1))(conv7)
#    act = Activation('softmax',axis=-1)(conv8)
#    print(act)
    model = Model(inputs=inputs, outputs=conv8)

    return model


def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape])


def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):
    if deconvolution:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False.")

        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
                                                                       pool_size=pool_size, image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
                                                                                       depth=depth+1,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size)
    
