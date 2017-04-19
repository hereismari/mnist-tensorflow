# needed libraries
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

logs_path = '/tmp/tensorflow_logs/softmax'

# one_hot=True: flatten this array into a vector of 28x28 = 784 numbers
# mnist.train = 55,000 input data
# mnist.test = 10,000 input data
# mnist.validate = 5,000 input data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# to make it simpler let's limit the train and test data set
# train: 2000 input data
# test: 500 input data
train_x, train_y = mnist.train.next_batch(2000)
test_x, test_y = mnist.test.next_batch(500)

# one test data at a time
x = tf.placeholder(tf.float32, [None, 784], name='input_data')

# weight
W = tf.Variable(tf.zeros([784, 10]), name='weight')

# bias
b = tf.Variable(tf.zeros([10]), name='bias')

with tf.name_scope('Model'):
    # test_data * W + b
    y = tf.matmul(x, W) + b
    sm = tf.nn.softmax(y)

y_ = tf.placeholder(tf.float32, [None, 10], name='real_class')

with tf.name_scope('Loss'):
    # cross entropy (loss function)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,
                                                                  labels=y_))

with tf.name_scope('TrainStep'):
    # train step
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.name_scope('Accuracy'):
    # evaluating the model
    correct_prediction = tf.equal(tf.argmax(sm, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a summary to monitor loss tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# init
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                           graph=tf.get_default_graph())

    # training
    for i in xrange(1000):
        error, ts, acc, summary = session.run([loss, train_step, accuracy,
                                               merged_summary_op],
                                              feed_dict={x: train_x,
                                                         y_: train_y})
        summary_writer.add_summary(summary, i)

    # running evaluation
    acc = accuracy.eval(feed_dict={x: test_x, y_: test_y})
    print 'Done!'
    print 'Accuracy:', acc * 100, '%'
