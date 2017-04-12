## Exploring MNIST with TensorFlow [WIP]

![mnist](http://rodrigob.github.io/are_we_there_yet/build/images/mnist.png?1363085077)


There are a looooooot of awesome tutorials about
[MNIST database](http://yann.lecun.com/exdb/mnist/),
so my plan is to put some of these tutorials together,
envolving from a very simple model to a nice model that achieves
99% accuracy in the test data.  

In the end I'll also show how to make an android app with your model in
wich you can draw and see if the model gets your drawing right.  

Have fun :party:!

**To be clear: most of the original code is not mine (you can find the
original source in this README file), but I'm making small changes were
I think they fit.**

## Getting started

For running version 1 to version X You'll need:

* Python 2 or 3
* Numpy 
* [TensorFlow](https://www.tensorflow.org/install/)
* Jupyter Notebook

If you want to run it in your cell... [WIP]

### Getting the data

The data is available at the /data folder, so you don't have to do much.  
You can download the data [here](http://yann.lecun.com/exdb/mnist/), if
you want.

The data consists of:

* train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
* train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
* t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
* t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

## Let's do it!

To run this just run: `jupyter notebook`

## References

Thank you!

0. [TF and DL without a Phd by Martin Gorner, Google](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)  
1. [Classifying Handwritten Digits with TF.Learn, by Josh Gordon](https://www.youtube.com/watch?v=Gj0iyo265bc&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=1)  
2. [How to Make a TF Image Classifier, by Siraj Raval](https://github.com/llSourcell/How_to_make_a_tensorflow_image_classifier_LIVE)
3. [TensorFlow Tutorials, by Hvass Labs ](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
