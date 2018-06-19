import tensorflow as tf

a = tf.placeholder(dtype=tf.float16, name="a")
b = tf.placeholder(dtype=tf.float16, name="b")
x = tf.placeholder(dtype=tf.float16, name="mult")

adder_node = a + b

add_and_mult = tf.multiply(adder_node, x, name="add_and_multiply")

sess = tf.Session()

print (sess.run(add_and_mult, {a:2, b:2, x:3}))

with tf.summary.FileWriter("./my_graph", graph=sess.graph) as writer:
    writer.flush()
    pass
    
sess.close()
