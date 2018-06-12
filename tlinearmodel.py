import tensorflow as tf

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)

linear_model = W * x + b

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4, 5, 6, 7, 8, 9]}))

y = tf.placeholder(dtype=tf.float32)
squared_deltas = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    if i % 50 == 0:
        print("W has value: {}".format(sess.run(W)))
        print("b has value: {}".format(sess.run(b)))
        pass
    pass

print(sess.run([W, b]))
