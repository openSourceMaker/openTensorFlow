
import tensorflow as tf
import numpy as np

learning_rate = 0.01

xy = np.loadtxt("breast_cancer_training.txt", unpack = True, dtype = "float32")
x_input = np.transpose(xy[1 : 10])
y_input = np.transpose(xy[10:])

X = tf.placeholder(tf.float32, [None, 9])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_uniform([9, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 12], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([12, 9], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([9, 7], -1.0, 1.0))
W5 = tf.Variable(tf.random_uniform([7, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([12]))
b3 = tf.Variable(tf.zeros([9]))
b4 = tf.Variable(tf.zeros([7]))
b5 = tf.Variable(tf.zeros([1]))

layer0 = tf.matmul(X, W1) + b1
layer1 = tf.nn.relu(tf.matmul(layer0, W2) + b2)
layer2 = tf.tanh(tf.matmul(layer1, W3) + b3)
layer3 = tf.nn.relu(tf.matmul(layer2, W4) + b4)
h = tf.sigmoid(tf.matmul(layer3, W5) + b5)

cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1. - h))

optmizer = tf.train.AdamOptimizer(learning_rate)
train = optmizer.minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10000):
        sess.run(train, feed_dict = {X: x_input, Y: y_input})

        if step % 2000 == 0:
            print(step, sess.run(cost, feed_dict = {X: x_input, Y: y_input}),
                  sess.run(W1),
                  sess.run(W2),
                  sess.run(W3),
                  sess.run(W4),
                  sess.run(W5))

    # saver.save(sess, "/your/path/for/saving/")

    # Test model
    correct_prediction = tf.equal(tf.floor(h + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print(sess.run([h, tf.floor(h + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_input, Y: y_input}))
    print("Accuracy : ", accuracy.eval({X: x_input, Y: y_input}))



