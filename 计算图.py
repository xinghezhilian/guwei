import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
result = a+b
sess = tf.Session()
runResult = sess.run(result)
print(runResult)
print(a.graph is tf.get_default_graph())

g1 = tf.Graph()
with g1.as_default():
    v=tf.get_variable('v',initializer=tf.ones_initializer([1]))

g2 = tf.Graph()
with g2.as_default():
    v=tf.get_variable('v',initializer=tf.ones_initializer(shape=[1]))

with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope('',reuse=True):
        print(sess.run(tf.get_variable('v')))

with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

g = tf.Graph()
with g.device('/gpu:0'):
    result = a+b

