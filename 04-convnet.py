# needed libraries
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

logs_path = '/tmp/tensorflow_logs/convnet'

# mnist.train = 55,000 input data
# mnist.test = 10,000 input data
# mnist.validate = 5,000 input data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Implementing Convnet with TF


def weight_variable(shape, name=None):
    # break simmetry
    if name:
        w = tf.truncated_normal(shape, stddev=0.1, name=name)
    else:
        w = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(w)


def bias_variable(shape, name=None):
    # avoid dead neurons
    if name:
        b = tf.constant(0.1, shape=shape, name=name)
    else:
        b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)


# pool
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# creates default conv layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters,
                   use_pooling=True):

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = weight_variable(shape)
    biases = bias_variable([num_filters])

    # input: 4D tensor (normally: [num_inputs, width, height, depth])
    # filter: 4d tensor we will learn and move in the image as defined
    #         by strides.
    #         (normally: [filter_size, filter_size, num_input_channels,
    #                    num_filters])
    # strides: [batch_stride x_stride y_stride depth_stride]
    #           batch_stride = 1 (we don't want to skip images)
    #           x_stride = move filter x positions
    #           y_stride = move filter y positions
    #           depth_stride = 1 (we don't want to skip any depth channel)
    # padding: SAME, means we will 0 pad the image, so the output
    #          will have the same dimension of the input
    layer = tf.nn.relu(tf.nn.conv2d(input=input,
                                    filter=weights,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME') + biases)

    if use_pooling:
        return max_pool_2x2(layer), weights

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = weight_variable([num_inputs, num_outputs])
    biases = bias_variable([num_outputs])

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# our network!!!


# input data
x = tf.placeholder(tf.float32, shape=[None, 28*28], name='input_data')
x_image = tf.reshape(x, [-1, 28, 28, 1])
# correct labels
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='correct_labels')

# fist conv layer
with tf.name_scope('convLayer1'):
    convlayer1, w1 = new_conv_layer(x_image, 1, 5, 32)
# second conv layer
with tf.name_scope('convLayer2'):
    convlayer2, w2 = new_conv_layer(convlayer1, 32, 5, 64)
# flat layer
with tf.name_scope('flattenLayer'):
    flat_layer, num_features = flatten_layer(convlayer2)
# fully connected layer
with tf.name_scope('FullyConnectedLayer'):
    fclayer = new_fc_layer(flat_layer, num_features, 1024)

# DROPOUT
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    drop_layer = tf.nn.dropout(fclayer, keep_prob)

# final layer
with tf.name_scope('FinalLayer'):
    W_f = weight_variable([1024, 10], name='W_f')
    b_f = bias_variable([10], name='b_f')
    y_f = tf.matmul(drop_layer, W_f) + b_f
    y_f_softmax = tf.nn.softmax(y_f)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                              logits=y_f))

# train step
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# accuracy
correct_prediction = tf.equal(tf.argmax(y_f_softmax, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a summary to monitor loss tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# init
init = tf.global_variables_initializer()

# Running the graph

num_steps = 3000
batch_size = 16
test_size = 10000
test_accuracy = 0.0

with tf.Session() as sess:

    sess.run(init)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                           graph=tf.get_default_graph())

    for step in range(num_steps):
        batch = mnist.train.next_batch(batch_size)

        ts, error, acc, summary = sess.run([train_step, loss, accuracy,
                                            merged_summary_op],
                                           feed_dict={x: batch[0],
                                                      y_: batch[1],
                                                      keep_prob: 0.5})
        if step % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %f' % (step, train_accuracy))

    print 'Done!'
    print 'Evaluating...'
    for i in xrange(test_size/50):
        batch = mnist.test.next_batch(50)
        acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],
                                       keep_prob: 1.0})
        if i % 10 == 0:
            print('%d: test accuracy %f' % (i, acc))
        test_accuracy += acc
    print 'avg test accuracy:', test_accuracy/(test_size/50.0)
