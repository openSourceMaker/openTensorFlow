
import tensorflow as tf
import numpy as np

# set learning rate
learning_rate = 0.05

# load data
xy = np.loadtxt("breast_cancer_training.txt", unpack = True, dtype = "float32")
x_input = np.transpose(xy[1 : 10])
y_input = np.transpose(xy[10:])

# set variables you will use
X = tf.placeholder(tf.float32, [None, 9])
Y = tf.placeholder(tf.float32, [None, 1])

# set weights for learning
W1 = tf.Variable(tf.random_uniform([9, 10], -3.0, 3.0))
W2 = tf.Variable(tf.random_uniform([10, 5], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([5, 3], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([3, 5], -3.0, 3.0))
W5 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0))

# set biases for learning
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([5]))
b3 = tf.Variable(tf.zeros([3]))
b4 = tf.Variable(tf.zeros([5]))
b5 = tf.Variable(tf.zeros([1]))

# set layers for deep learning
layer0 = tf.matmul(X, W1) + b1
layer1 = tf.nn.relu(tf.matmul(layer0, W2) + b2)
layer2 = tf.tanh(tf.matmul(layer1, W3) + b3)
layer2 = tf.nn.dropout(layer2, 0.8)
layer3 = tf.nn.relu(tf.matmul(layer2, W4) + b4)
h = tf.nn.sigmoid(tf.matmul(layer3, W5) + b5)

# calculate cost and minimise cost
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1. - h))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# set initialisation variable for initialising
init = tf.global_variables_initializer()

# set save operation for your learning
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10000):
        sess.run(train, feed_dict = {X: x_input, Y: y_input})

        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict = {X: x_input, Y: y_input}),
                  sess.run(W1),
                  sess.run(W2),
                  sess.run(W3),
                  sess.run(W4),
                  sess.run(W5))

    saver.save(sess, "your/path/for/saving")

    # correction and accuracy that you want to see how it is well learned
    correction = tf.equal(tf.floor(h + 0.5), Y)
    acc = tf.reduce_mean(tf.cast(correction, "float"))

    print(sess.run([h, tf.floor(h + 0.5), correction, acc], feed_dict={X: x_input, Y: y_input}))
    print("Accuracy : ", acc.eval({X: x_input, Y: y_input}))



