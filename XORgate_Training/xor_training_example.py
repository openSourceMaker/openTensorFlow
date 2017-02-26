
import tensorflow as tf
import numpy as np

# set learning rate
learning_rate = 0.1

# load training data
xy = np.loadtxt("xor.txt", unpack = True)
x_input = np.transpose(xy[0 : -1])
y_input = np.reshape(xy[-1], (4, 1))

# set variables you will feed
X = tf.placeholder(tf.float32, shape = [None, 2])
Y = tf.placeholder(tf.float32, shape = [None, 1])

# set weights for learning
W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))

# set biases for learning
b1 = tf.Variable(tf.zeros([2]))
b2 = tf.Variable(tf.zeros([1]))

# set layers for deep learning
input_layer = tf.sigmoid(tf.matmul(X, W1) + b1)
output_layer = tf.sigmoid(tf.matmul(input_layer, W2) + b2)

# calculate cost and minimise cost
# cross entropy cost function
cost = -tf.reduce_mean(Y * tf.log(output_layer) + (1 - Y) * tf.log(1. - output_layer))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# set initialisation operation for initialising
init = tf.global_variables_initializer()

# set save operation for your learning
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        sess.run(train, feed_dict = {X: x_input, Y: y_input})

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict = {X: x_input, Y: y_input}),
                  "W1 : ", sess.run(W1),
                  "W2 : ", sess.run(W2))

    # saver.save(sess, "/your/path/for/saving")

    # correction and accuracy that you want to see how it is well learned
    correction = tf.equal(tf.floor(output_layer + 0.5), Y)
    acc = tf.reduce_mean(tf.cast(correction, "float"))

    print("Y : ", sess.run([output_layer, tf.floor(output_layer + 0.5)], feed_dict={X: x_input, Y: y_input}))
    print("Accuracy : ", acc.eval({X: x_input, Y: y_input}))
