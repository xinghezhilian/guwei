import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
result = tf.add(a,b,name='add')
print(result)
print(result.get_shape())















