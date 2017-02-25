
import tensorflow as tf
import numpy as np

# set variables you will feed
X = tf.placeholder(tf.float32, shape = [None, 2])
Y = tf.placeholder(tf.float32, shape = [None, 1])

# set weights for learning
W1 = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([3, 1], -1.0, 1.0))

# set biases for learning
b1 = tf.Variable(tf.zeros([3]))
b2 = tf.Variable(tf.zeros([1]))

# set layers for deep learning
input_layer = tf.sigmoid(tf.matmul(X, W1) + b1)
output_layer = tf.sigmoid(tf.matmul(input_layer, W2) + b2)

# set initialisation operation for initialising
init = tf.global_variables_initializer()

# set save operation for your learning
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # you can load trained data using saver.restore(session, "path")
    saver.restore(sess, "/your/path/for/loading/")

    # You can test the trained data by feeding values by yourself
    print("Type values you want to test : ")
    while(1):
        x1 = raw_input()
        x2 = raw_input()

        x_test = [[x1, x2]]

        print(sess.run([output_layer, tf.floor(output_layer + 0.5)], feed_dict = {X: x_test}))
