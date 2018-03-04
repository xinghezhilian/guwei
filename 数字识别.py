import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("D:/pyspace/testdata", one_hot=True)
# print("Training data size:", mnist.train.num_examples)
# print("Validating data size:",mnist.validation.num_examples)
# print("Testing data size:",mnist.test.num_examples)
# print("Exampple training data:",mnist.train.images[0])
# print("Example training data lable",mnist.train.labels[0])
# batch_size = 100
# xs,ys = mnist.train.next_batch(batch_size)
# print("X shape:",xs.shape)
# print("Y shape:",ys.shape)

INPUT_NODE = 784  # 输入
OUTPUT_NODE = 10  # 输出

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TREANING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class ==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1))+biases1
        return tf.nn.relu(tf.matmul(layer1,weights2))+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="y-input")
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    y = inference(x,None,weights1,biases1,weights2,biases2)
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=tf.argmax(y_,dimension=1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.L2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_top = tf.no_op(name="train")
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.case(correct_prediction,tf.float32))
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        for i in range(TREANING_STEPS):
            if i%1000 ==0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy "
                      "useing average model is %g" % (i,validate_acc))
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_top,feed_dict={x:xs,y_:ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy user average models is %g" % (TREANING_STEPS,test_acc))

# def main(argv=None):
#     mnist = input_data.read_data_sets("D:/pyspace/testdata", one_hot=True)
#     assert isinstance(mnist, object)
#     train(mnist)

if __name__=='__main__':
    print("start")
    mnist = input_data.read_data_sets("D:/pyspace/testdata", one_hot=True)
    train(mnist)
    # tf.app.run()

