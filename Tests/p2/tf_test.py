import tensorflow as tf

# Create Tensorflow object called tensor
hello_constant = tf.constant('Hello World')

with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)

#-----

# Create placeholder
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x:'Hello World'})
    print(output)
