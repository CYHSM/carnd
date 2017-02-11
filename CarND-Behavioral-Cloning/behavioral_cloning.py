import pandas as pd
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Dropout, Dense, BatchNormalization, Flatten, Reshape, Input, MaxPooling2D, merge, Activation, AveragePooling2D
from keras.preprocessing.image import random_shift, random_shear, load_img, flip_axis, img_to_array
import matplotlib.pyplot as plt
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""

def load_data(path, frac=0.01, cutoff=0.01):
    csv_data = pd.read_csv(path)
    print(len(csv_data))
    # Restrict steering angles of 0 by frac
    steering_angles = csv_data.ix[:,3]
    #speed = csv_data.ix[:,6]
    csv_data = pd.concat([csv_data[abs(steering_angles) >= cutoff], csv_data[abs(steering_angles) < cutoff].sample(frac=frac)])
    print(len(csv_data))

    return csv_data

def _generator(csv_data, batch_size, input_shape):
    """
    Image generator. Returns batches of images indefinitely
    - path : path to csv file
    - batch_size : batch size
    """
    while 1:
        batch = csv_data.sample(n=batch_size)
        batch_X, batch_y = process_batch(batch, input_shape=input_shape)
        yield np.array(batch_X), np.array(batch_y)

def process_batch(batch, steering_offset=0.25, input_shape=(100, 100, 1)):
    """
    Processes one batch and returns X and y
    - batch : pandas df with csv data
    """
    batch_X, batch_y = [], []
    for row in batch.itertuples():
        c_img, l_img, r_img = load_img(row[1].strip(), target_size=input_shape), load_img(row[2].strip(), target_size=input_shape), load_img(row[3].strip(), target_size=input_shape)
        steering_angle = row[4]

        # Preprocess image and create augmented examples
        grayscale = False if input_shape[2] == 3 else True
        images, steering_angles = preprocess_image([c_img,l_img,r_img], steering_angle=steering_angle, grayscale=grayscale)
        # ----------------------------------------------
        # batch_X += [images[0], images[1], images[2]]
        # batch_y += [steering_angles[0], steering_angles[1]+steering_offset, steering_angles[2]-steering_offset]

        #
        # batch_X += [images[1], images[2]]
        # batch_y += [steering_angles[1]+steering_offset, steering_angles[2]-steering_offset]

        batch_X += [images[0], images[1], images[2], flip_axis(images[0], 1), flip_axis(images[1], 1), flip_axis(images[2],1)]
        batch_y += [steering_angles[0], steering_angles[1]+steering_offset, steering_angles[2]-steering_offset, -steering_angles[0],  -(steering_angles[1]+steering_offset), -(steering_angles[2]-steering_offset)]

    return batch_X, batch_y

def preprocess_image(images, steering_angle=None, grayscale=True):
    """
    Preprocesses the image for input to the deep learning model
    - images : array of images
    - steering_angle : angle for all images in array
    """
    img_return, steering_angles = [], []
    for img in images:
        img = img_to_array(img)
        if grayscale:
            # img = np.dot(img[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
            # Use Hsv
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1][..., np.newaxis]
            img = np.subtract(img, -0.5)
        else:
            img = np.subtract(np.divide(img, 255), -0.5)
        # Create augmented images
        img, steering_angle = augment_image(img, steering_angle=steering_angle)
        steering_angles.append(steering_angle)
        img_return.append(img)
    return img_return, steering_angles

def random_darken(image, patch_size=30, frac_brightness=0.5):
    """
    Randomly darkens a part of an image
    """
    im_shape = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #convert it to hsv
    start_x, start_y = np.random.randint(0, im_shape[0]), np.random.randint(0, im_shape[1])
    end_x, end_y = np.random.randint(start_x, im_shape[0]), np.random.randint(start_y, im_shape[1])
    # Change third dimension
    hsv[start_x:end_x, start_y:end_y, 2] *= frac_brightness
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image

def augment_image(image, steering_angle=None):
    """
    Generates new examples from given image
    - images : array of images
    - steering_angle : angle for all images in array
    """
    # if np.random.uniform() < 0.5:
    #     image = flip_axis(image, 1)
    #     steering_angle = -steering_angle
    # Try random horizontal shift
    # if np.random.uniform() < 0.5:
    image = random_shift(image, 0, 0.2, 0, 1, 2)
    image = random_darken(image)

    return image, steering_angle

def create_model(input_shape):
    """
    Create keras deep learning model
    """
    convolutions = [32, 64, 128]
    fc_layers = [1000, 500, 250]

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add more convolutional layer
    for conv in convolutions:
        model.add(Convolution2D(conv, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add fully connected layers
    model.add(Flatten())
    for fc in fc_layers:
        model.add(Dense(fc, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.6))
    # Add layer for output
    model.add(Dense(1, activation='linear'))

    return model

def save_model(model):
    """
    Save model weights and architecture
    """
    with open("./model.json", "w") as json_file:
        json_file.write(model.to_json())
        json_file.close()
    model.save_weights("./model.h5")

def train():
    """
    Load data and train model
    """
    input_shape = (60, 60, 3)
    batch_size = 64

    model = create_model(input_shape)
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    csv_data = load_data(path='./Data/driving_log.csv')
    model.fit_generator(_generator(csv_data, batch_size=batch_size, input_shape=input_shape), samples_per_epoch=300*batch_size, nb_epoch=5)
    save_model(model)

if __name__ == '__main__':
    train()
