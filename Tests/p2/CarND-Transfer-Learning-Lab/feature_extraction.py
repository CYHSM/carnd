import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Activation, Flatten, Convolution1D

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 10, 'Number of Epochs')
flags.DEFINE_integer('batch_size', 256, 'Size of Batch')


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    shape_input = X_train.shape[1:]
    n_classes = len(np.unique(y_train))
    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    model = Sequential()
    model.add(Flatten(input_shape=shape_input))
    #model.add(Convolution2D(64, 1, 1, border_mode='valid',  input_shape=shape_input))
    #model.add(Activation('relu'))
    #model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    model.compile('adam','sparse_categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),  batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs, shuffle=True)



    # TODO: train your model here


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
