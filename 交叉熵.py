import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

v = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
result = tf.clip_by_value(v,2.5,4.5)
sess = tf.Session()
print(sess.run(result))
sess.close()