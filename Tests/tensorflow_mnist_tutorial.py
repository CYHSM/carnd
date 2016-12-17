from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Load Mnist Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Open interactive session
sess = tf.InteractiveSession()

# Nodes for input images and target output
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define Weights and biases b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialize Variables
sess.run(tf.initialize_all_variables())

# Implement regression model
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Train with steepest gradient
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if not i % 100:
        print(i)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_float = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_float)
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
