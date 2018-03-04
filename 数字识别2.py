import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:/pyspace/testdata", one_hot=True)

learning_rate = 0.01  # 学习速率
batch_size = 128  # 每次训练数据量
n_epochs = 30  # 训练次数

# 输入定义，none 表示此张量的第一个维度可以是任何长度的。
X = tf.placeholder(tf.float32, [batch_size, 784])
# 概率回归统计输出结果
Y = tf.placeholder(tf.int32, [batch_size, 10])

with tf.name_scope("Wx_b") as scope:
    w = tf.Variable(tf.random_normal(shape=[748,10], stddev=0.01), name='weights')
    b = tf.Variable(tf.zeros(shape=[1, 10], name="bias"))
    logits = tf.matmul(X, w) + b

with tf.name_scope("cost") as scope:
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
    loss = tf.reduce_mean(entropy)
    tf.summary.scalar("loss", loss)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graphs/logistic_reg", sess.graph)
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)
    for i in range(n_batches):
        total_loss = 0
        for j in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
    summary_str = sess.run(summary, feed_dict={X: X_batch, Y: Y_batch})
    writer.add_summary(summary_str, i*n_batches)
    print('Average loss epoch {0}: {1}'.format(i,total_loss/n_batches))
