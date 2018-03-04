import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:/pyspace/testdata", one_hot=True)
# 输入定义，none 表示此张量的第一个维度可以是任何长度的。
x = tf.placeholder("float", [None, 784])
# 初始化输入的权重
W = tf.Variable(tf.zeros([784, 10]))
# 初始化偏置
b = tf.Variable(tf.zeros([10]))
# 概率回归统计输出结果
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 正确值占位变量
y_ = tf.placeholder("float", [None, 10])
# 交叉熵计算
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 梯度下降后向传播优化，微调变量，减少成本
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 初始化变量
init = tf.initialize_all_variables()
# 创建对话，并初始化变量
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
