# needed libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

logs_path = '/tmp/tensorflow_logs/mnist'

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

# Visualization
plt.rc("image", cmap="binary")  # use black/white palette for plotting
for i in xrange(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_x[i].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
plt.show()

# number of neighboors
K = 5

# tf Graph Input
ph_train = tf.placeholder("float", [None, 784], name='trainImages')
ph_test = tf.placeholder("float", [784], name='testImage')

c = tf.placeholder("float")

# Calculate L2 Distance
with tf.name_scope('Distance'):
    distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(
                           tf.add(ph_train, tf.negative(ph_test))),
                           reduction_indices=1)))

# Prediction: Get K min distance index (Nearest neighbor)
with tf.name_scope('CalculateKNN'):
    pred_values_indices = tf.nn.top_k(distance, k=K, sorted=False)

with tf.name_scope('CalculateAccuracy'):
    accuracy = tf.multiply(c, 1.0/len(test_x))

# Initializing the variables
init = tf.global_variables_initializer()

# Merge all summaries into a single op
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()


with tf.Session() as session:

    session.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                           graph=tf.get_default_graph())
    correct_class = 0.0
    for i in range(len(test_x)):
        values_indices = session.run(pred_values_indices,
                                     feed_dict={
                                        ph_train: train_x,
                                        ph_test: test_x[i, :]})

        # predicting label for test data
        counter = np.zeros(10)
        for j in xrange(K):
            counter[np.argmax(train_y[values_indices.indices[j]])] += 1

        prediction = np.argmax(counter)
        if prediction == np.argmax(test_y[i]):
            correct_class += 1.0

        acc, summary = session.run([accuracy, merged_summary_op],
                                   feed_dict={c: correct_class})

        summary_writer.add_summary(summary, i)

    print 'Done!'
    print 'Accuracy:', (correct_class / len(test_x)) * 100, '%'
