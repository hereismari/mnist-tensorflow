## Exploring MNIST with TensorFlow [WIP]

![mnist](http://rodrigob.github.io/are_we_there_yet/build/images/mnist.png?1363085077)

There are a lot of awesome tutorials about how to classify handwritten
digits from [MNIST database](http://yann.lecun.com/exdb/mnist/),
so my plan is to put some of these tutorials together,
evolving from a very simple model to a nice model that achieves
99% accuracy in the test data.  

In the end I'll also show how to make an android app with your model in
which you can draw and see if the model gets your drawing right.  

Have fun :party:!

**To be clear: most of the original code is not mine (you can find the
original source in this README file), but I'm making small changes were
I think they fit.**

## Getting started

For running the "tutorials" You'll need:

* Python 2 or 3
* Numpy
* [TensorFlow](https://www.tensorflow.org/install/)
* Jupyter Notebook

If you want to run it in your cellphone... [WIP]

### Getting the data

The MNIST databased used here was the one available at TensorFlow
([read more about it](https://www.tensorflow.org/get_started/mnist/beginners)),
so you don't have to do much.  
You can download the data [here](http://yann.lecun.com/exdb/mnist/), if
you want.

For the examples in this repository we'll use all the available data and train
the models with stochastic gradient descent (except 01-KNN).

## Let's do it!

To follow the "tutorials", just run: `jupyter notebook`.  

If you want to check the code, just run manually `python <file name>.py`, also
if you want to check TensorBoard you can! Just run `tensorboard --logdir=/tmp/tensorflow_log`

## References

Thank you!

0. [TF and DL without a Phd by Martin Gorner, Google](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)  
1. [Classifying Handwritten Digits with TF.Learn, by Josh Gordon](https://www.youtube.com/watch?v=Gj0iyo265bc&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=1)  
2. [How to Make a TF Image Classifier, by Siraj Raval](https://github.com/llSourcell/How_to_make_a_tensorflow_image_classifier_LIVE)
3. [TensorFlow Tutorials, by Hvass Labs ](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
4. [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
5. [Getting_started with TensorFlow](https://www.tensorflow.org/get_started/get_started)
