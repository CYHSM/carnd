from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
from datetime import datetime

# Load Mnist Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Open interactive session
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

# Define methods for weight and bias initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #http://deeplearning.net/software/theano_versions/dev/_images/numerical_padding_strides.gif
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"
with tf.device(device_name):
    # Nodes for input images and target output
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Implement first layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # Reshape image to a 4d tensor
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Convolve and ReLu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # Max Pooling
    h_pool1 = max_pool_2x2(h_conv1)

    # Implement second layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Add a fully connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Add Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Add Readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Train and evaluate
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Initialize Variables
sess.run(tf.global_variables_initializer())
# run
startTime = datetime.now()
for i in range(2000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        print("Time taken:", datetime.now() - startTime)
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        startTime = datetime.now()
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
