from keras.layers import Conv3D, SeparableConv2D, Dense, Reshape, Flatten, Dropout, Conv2D
from keras.layers import Input, GlobalAveragePooling2D, Concatenate, MaxPooling2D
from keras import regularizers
from keras.layers import BatchNormalization, Activation
from keras.models import Model

import keras.backend as k

k.set_image_data_format('channels_last')
k.set_learning_phase(1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define additional methods 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def conv3d_attention(inp, filters, kernel_size=(3, 3, 7), padding='same', strides=(1, 1, 1)):
    conv = Conv3D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(inp)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define proposed and compared methods
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def hyper3dnet(img_shape=(256, 256, 50, 1), classes=2):
    # Input
    d0 = Input(shape=img_shape)

    # 3D convolutions
    conv_layer1 = conv3d_attention(d0, 8)
    conv_layer2 = conv3d_attention(conv_layer1, 8)
    conv_in = Concatenate()([conv_layer1, conv_layer2])
    conv_layer3 = conv3d_attention(conv_in, 8)
    conv_in = Concatenate()([conv_in, conv_layer3])
    conv_layer4 = conv3d_attention(conv_in, 8)
    conv_in = Concatenate()([conv_in, conv_layer4])

    conv_in = Reshape((conv_in.shape[1].value, conv_in.shape[2].value,
                       conv_in.shape[3].value * conv_in.shape[4].value))(conv_in)

    conv_in = SeparableConv2D(128, kernel_size=3, strides=(1, 1), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)

    conv_in = Flatten()(conv_in)
    conv_in = Dropout(0.5)(conv_in)

    if classes == 2:
        fc1 = Dense(1, name='fc' + str(1), activation='sigmoid')(conv_in)
    else:
        fc1 = Dense(classes, name='fc' + str(classes), activation='softmax')(conv_in)

    return Model(d0, fc1)


def hybridsn(img_shape=(256, 256, 50, 1), classes=2):
    # input layer
    input_layer = Input(img_shape)

    # convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
    conv3d_shape = conv_layer3._keras_shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    # fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=classes, activation='softmax')(dense_layer2)

    return Model(input_layer, output_layer)


def Spectrum(x, squeeze_planes, expand1x1_planes, expand3x3_planes):
    squeeze_p = Conv2D(filters=squeeze_planes, kernel_size=1, activation='relu')(x)
    expand1x1 = Conv2D(filters=expand1x1_planes, kernel_size=1, padding='same', activation='relu')(squeeze_p)
    expand3x3 = Conv2D(filters=expand3x3_planes, kernel_size=3, padding='same', activation='relu')(squeeze_p)
    output = Concatenate()([expand1x1, expand3x3])

    return output


def spectrumnet(img_shape=(256, 256, 50), classes=2):
    # input layer
    input_layer = Input(img_shape)

    # convolutional layers
    x = Conv2D(filters=96, kernel_size=(2, 2), strides=(2, 2), activation='relu')(input_layer)
    x = Spectrum(x, 32, 64, 64)
    x = Spectrum(x, 32, 64, 64)
    x = Spectrum(x, 64, 128, 128)
    x = MaxPooling2D(pool_size=2, strides=2)(x)  # changed kernel size
    x = Spectrum(x, 64, 128, 128)
    x = Spectrum(x, 96, 192, 192)
    x = Spectrum(x, 96, 192, 192)
    x = Spectrum(x, 128, 256, 256)
    x = MaxPooling2D(pool_size=2, strides=2)(x)  # changed kernel size
    x = Spectrum(x, 128, 256, 256)

    x = Dropout(0.5)(x)
    x = Conv2D(filters=classes, kernel_size=(1, 1), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)

    return Model(input_layer, x)


def weedann(img_shape=(256, 256, 50, 1), classes=2):
    # input layer
    input_layer = Input(img_shape)

    fc = Dense(units=500, activation='relu', kernel_regularizer=regularizers.l2(0.005))(input_layer)
    fc = Dense(units=500, activation='relu', kernel_regularizer=regularizers.l2(0.005))(fc)
    fc = Dense(units=classes, activation='softmax')(fc)

    return Model(input_layer, fc)
