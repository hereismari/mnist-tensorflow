# needed libraries
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

logs_path = '/tmp/tensorflow_logs/relu-softmax'

# one_hot=True: flatten this array into a vector of 28x28 = 784 numbers
# mnist.train = 55,000 input data
# mnist.test = 10,000 input data
# mnist.validate = 5,000 input data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Implementing Relu and Softmax Layer

num_nodes = 1024
# sg, if we use all the data the trainning will take a lot of time
batch_size = 128

# input
x = tf.placeholder(tf.float32, [None, 784], name='input_data')

# --------- hidden layer (RELU) -------------

with tf.name_scope("RELU"):
    # weight
    w1 = tf.Variable(tf.truncated_normal([784, num_nodes]))
    # bias
    b1 = tf.Variable(tf.zeros([num_nodes]))
    # test_data * w1 + b1
    y1 = tf.matmul(x, w1) + b1
    relu = tf.nn.relu(y1)

# -------- output layer (sofmax) -------------

with tf.name_scope("Softmax"):
    # weight
    w2 = tf.Variable(tf.truncated_normal([num_nodes, 10]))
    # bias
    b2 = tf.Variable(tf.zeros([10]))
    # relu * w2 + b2
    y2 = tf.matmul(relu, w2) + b2
    sm = tf.nn.softmax(y2)

# cross entropy (loss function)
y_ = tf.placeholder(tf.float32, [None, 10], name='correct_labels')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y2,
                                                              labels=y_),
                      name='loss')
# train step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# evaluating the model
correct_prediction = tf.equal(tf.argmax(sm, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                          name='accuracy')

# Create a summary to monitor loss tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# init
init = tf.global_variables_initializer()

# Running the graph

num_steps = 1000
with tf.Session() as session:

    session.run(init)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                           graph=tf.get_default_graph())

    # training
    for step in xrange(num_steps):
        # Generate a minibatch.
        batch_data, batch_labels = mnist.train.next_batch(batch_size)

        error, ts, acc, summary = session.run([loss, train_step, accuracy,
                                               merged_summary_op],
                                              feed_dict={x: batch_data,
                                                         y_: batch_labels})
        summary_writer.add_summary(summary, step)

    # evaluating the model
    acc = accuracy.eval(feed_dict={x: mnist.test.images,
                                   y_: mnist.test.labels})

    print 'Done!'
    print 'Accuracy:', acc * 100, '%'
