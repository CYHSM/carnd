import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Convolution2D, Dropout, Dense, BatchNormalization, Flatten, Reshape, Input, MaxPooling2D, merge, Activation, AveragePooling2D
from keras.models import Sequential
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import keras.backend
import tensorflow as tf
import json
from keras.models import model_from_json, load_model



# import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
path_to_csv = './Data/driving_log.csv'
df = pd.read_csv(path_to_csv, header=None)

steering_angles = df.ix[:,3]
steering_angles.plot()
np.sum(steering_angles<-0.1)

vec_zeros = steering_angles==0
df[vec_zeros.sample(n=10).keys()]
ind = vec_zeros.sample(frac=0.9).index
rows = np.random.choice(df.index.values, 10)
df.xs(vec_zeros.sample(n=10).keys())
#
# df2 = df.drop(ind)
# len(df2)
# len(df)
# len(ind)
# len(vec_zeros)
# vec_zeros = df.where(steering_angles==0)
#
#
# vec_zeros
# ind2 = vec_zeros.sample(frac=1)
# len(ind2)

# plt.plot(df.ix[100:500,3].values)
# #
# n_images = df.shape[0]
#
# df_randsample = df.sample(n=128)
#
# im,l = load_images_and_labels_from_df(df_randsample)
# im.shape
# l.shape
# df_randsample.shape
#
# for l in df.itertuples():
#     print(l)
# l[2]
# df_randsample.loc[:,0].values
# center_images = cv2.imread(l[1], cv2.IMREAD_GRAYSCALE)
# plt.imshow(center_images, cmap='gray')
# center_images.shape

def preprocess_image(image_array):
    image = np.subtract(np.divide(image_array, 255), -0.5)
    image = cv2.resize(image, (224, 224))
    return image

def load_images_and_labels_from_df(df_batch):
    # center - left - right - steering - throttle - break - speed
    images = []
    labels = []
    for r in df_batch.itertuples():
        center_image, left_image, right_image = cv2.imread(r[1], cv2.IMREAD_GRAYSCALE), cv2.imread(r[2], cv2.IMREAD_GRAYSCALE), cv2.imread(r[3], cv2.IMREAD_GRAYSCALE)
        center_image = preprocess_image(center_image)
        steering_angle = r[4]

        # Randomly flip image
        if np.random.random_sample() < 0.5:
            center_image = cv2.flip(center_image, 1)
            steering_angle *= -1

        images.append(center_image)
        labels.append(steering_angle)
    return np.ascontiguousarray(images), np.ascontiguousarray(labels)

# 1. Load images from folder
def generate_array_from_file(path, batch_size):
    df = pd.read_csv(path)
    while 1:
        this_batch = df.sample(n=batch_size)
        images, labels = load_images_and_labels_from_df(this_batch)

        yield(images, labels)

def create_keras_model(input_shape):
    input_model = Input(shape=input_shape)
    input_image = Reshape(target_shape=input_shape + (1,))(input_model)
    #(Reshape((160, 320, 1), input_shape=(160, 320)))
    # Inception
    tower_1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_image)
    tower_1 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(tower_1)

    tower_2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_image)
    tower_2 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3,3), strides=(1, 1), border_mode='same')(input_image)
    tower_3 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(tower_3)

    output = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)

    after_max1 = MaxPooling2D((3,3))(output)
    after_conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(after_max1)
    after_max = MaxPooling2D((3,3))(after_conv)


    output_flat = Flatten()(after_max)

    output_fc = Dense(128, activation='relu')(output_flat)

    out = Dense(1)(output_fc)

    model = Model(input=input_model, output=out)

    return model

def create_keras_model_2(input_shape):
    model = Sequential()

    model.add(Reshape(input_shape + (1,), input_shape=input_shape))

    # model.add(Convolution2D(16, 3, 3,
    #                         border_mode='valid'))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(32, 3, 3,
    #                         border_mode='valid'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 1st Layer - Add a flatten layer
    model.add(Flatten())
    # 2nd Layer - Add a fully connected layer
    model.add(Dense(128))

    # 3rd Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

    # 4th Layer - Add a fully connected layer
    model.add(Dense(1))

    #model.add(Activation('softmax'))

    return model

def create_keras_model_3(input_shape):
    model = Sequential ([
        Reshape (input_shape + (1,), input_shape=input_shape),

        Convolution2D (32, 3, 3, border_mode='valid'),
        #MaxPooling2D (pool_size=(2, 2)),
        #Dropout (0.5),
        Activation ('relu'),

        Convolution2D (64, 3, 3, border_mode='valid'),
        #MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.4),
        Activation ('relu'),
        #
        # Convolution2D (48, 3, 3, border_mode='valid'),
        # MaxPooling2D (pool_size=(2, 2)),
        # Dropout (0.5),
        # Activation ('relu'),

        # Convolution2D (64, 2, 2, border_mode='valid'),
        # MaxPooling2D (pool_size=(2, 2)),
        # Dropout (0.5),
        # Activation ('relu'),

        # Convolution2D (64, 2, 2, border_mode='valid'),
        # MaxPooling2D (pool_size=(2, 2)),
        # Dropout (0.5),
        # Activation ('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),

        Dense(256),
        Activation('relu'),

        Dense(128),
        Activation('relu'),
        Dropout (0.4),

        Dense(1)
    ])
    return model

def SqueezeNet(nb_classes, inputs=(3, 224, 224)):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """
    input_img = Input(shape=inputs)
    input_reshaped = Reshape(target_shape=input_shape + (1,))(input_img)
    conv1 = Convolution2D(
        96, 7, 7, activation='relu', init='glorot_uniform',
        subsample=(2, 2), border_mode='same', name='conv1')(input_reshaped)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

    fire2_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_squeeze')(maxpool1)
    fire2_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand2')(fire2_squeeze)
    merge2 = merge(
        [fire2_expand1, fire2_expand2], mode='concat', concat_axis=1)

    fire3_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_squeeze')(merge2)
    fire3_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand2')(fire3_squeeze)
    merge3 = merge(
        [fire3_expand1, fire3_expand2], mode='concat', concat_axis=1)

    fire4_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_squeeze')(merge3)
    fire4_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand2')(fire4_squeeze)
    merge4 = merge(
        [fire4_expand1, fire4_expand2], mode='concat', concat_axis=1)
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

    fire5_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_squeeze')(maxpool4)
    fire5_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand2')(fire5_squeeze)
    merge5 = merge(
        [fire5_expand1, fire5_expand2], mode='concat', concat_axis=1)

    fire6_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_squeeze')(merge5)
    fire6_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand2')(fire6_squeeze)
    merge6 = merge(
        [fire6_expand1, fire6_expand2], mode='concat', concat_axis=1)

    fire7_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_squeeze')(merge6)
    fire7_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand1')(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand2')(fire7_squeeze)
    merge7 = merge(
        [fire7_expand1, fire7_expand2], mode='concat', concat_axis=1)

    fire8_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_squeeze')(merge7)
    fire8_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand1')(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand2')(fire8_squeeze)
    merge8 = merge(
        [fire8_expand1, fire8_expand2], mode='concat', concat_axis=1)

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)

    fire9_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_squeeze')(maxpool8)
    fire9_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand1')(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand2')(fire9_squeeze)
    merge9 = merge(
        [fire9_expand1, fire9_expand2], mode='concat', concat_axis=1)

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        nb_classes, 1, 1, init='glorot_uniform',
        border_mode='valid', name='conv10')(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D((13, 13), name='avgpool10')(conv10)

    flatten = Flatten(name='flatten')(avgpool10)

    out = Dense(1)(flatten)
    #softmax = Activation("softmax", name='softmax')(flatten)

    return Model(input=input_img, output=out)

# Create optimizer and compile model
input_shape = (224, 224)
model = SqueezeNet(1, inputs=input_shape)

#model = create_keras_model_3(input_shape)
model.summary()

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='mse')
path_to_csv = './Data/driving_log.csv'
batch_size = 100
#keras.backend.get_session().run(tf.global_variables_initializer())
model.fit_generator(generate_array_from_file(path_to_csv, batch_size), samples_per_epoch=300*batch_size, nb_epoch=2)

# df = pd.read_csv(path_to_csv)
# # Train on batches
# for e in range(10):
#     for b in range(100):
#         this_batch = df.sample(n=batch_size)
#         images, labels = load_images_and_labels_from_df(this_batch)
#         model.train_on_batch(images, labels)

# X. Save model weights and architecture
with open("./model.json", "w") as json_file:
    json_file.write(model.to_json())
    json_file.close()
model.save_weights("./model.h5")
