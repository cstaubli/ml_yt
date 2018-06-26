import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

NUM_STEPS = 100
LEARNING_RATE = 0.05

x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2

noise = np.random.randn(1, 2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    y_true = tf.placeholder(dtype=tf.float32, shape=None)
    with tf.name_scope("inference") as scope:
        w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name="weigths")
        b = tf.Variable(0, dtype=tf.float32, name="bias")
        y_pred = tf.matmul(w, tf.transpose(x) + b)

    with tf.name_scope("loss") as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})
            if (step % 25 == 0):
                outs = sess.run([w, b])
                print("Step: {} -> Outs: {}".format(step, outs))
                wb_.append(outs)
                plt.plot(outs[0][0])
                pass
            pass
        pass

        plt.plot(w_real, "g^")
        plt.xlabel("X-Axis")
        plt.ylabel("Value")
        plt.title("After {} steps".format(NUM_STEPS))
        plt.legend(["steps 0","steps 25","steps 50","steps 75","real"])
        plt.show()
