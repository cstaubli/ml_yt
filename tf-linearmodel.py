import tensorflow as tf

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

linear_model = W * x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    if i % 50 == 0:
        print("W has value: {}".format(sess.run(W)))
        print("b has value: {}".format(sess.run(b)))
        pass
    pass

print(sess.run([W, b]))

with tf.summary.FileWriter("./my_graph", graph=sess.graph) as writer:
    writer.flush()
    pass
