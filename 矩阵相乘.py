import tensorflow as tf
# a=tf.matmul(x,w1)
# y=tf.matmul(a,w2)
weights = tf.Variable(tf.random_normal([2,3],stddev=2))
biases = tf.Variable(tf.zeros([3]))
w2 = tf.Variable(weights.initial_value())
w3 = tf.Variable(weights.initial_value()*2.0)

print(weights)

