# xNet

This is a simple C++ implementation of neural networks using the xTensor library. 

## Installation

First install xTensor and xTensor-blas:

*[xtensor](https://github.com/QuantStack/xtensor)
*[xtensor-blas](https://github.com/QuantStack/xtensor-blas)

*xTensor
*xTensor-blas

Then clone this respository and do:

```
mkdir build
cd build
cmake ..
make
```

Then you can run the tests with
```
./test_simple_net
```

To run the MNIST example, you will need to download the MNIST dataset. You can use the provided shell script:

```
sh examples/download_mnist.sh
```

Then run the MNIST example with
```
./mnist_example
```

## MNIST Example

The MNIST example shows how to build and train a simple neural network.

```
mat train_data = ReadMNISTData("train");
mat test_data = ReadMNISTData("test");
mat test_label = ReadMNISTLabel("test");
mat train_label = ReadMNISTLabel("train");

BaseLayer* layer1 = new FullLayer(784, 128, "full1");
BaseLayer* relu1 = new ReluLayer("relu1");
BaseLayer* layer2 = new FullLayer(128, 10, "full2");
BaseLayer* softmax = new SoftmaxLoss("sm");

cout << "Training..." << endl;
Sequential model = Sequential({layer1, relu1, layer2},
                    softmax);

model.fit(train_data, train_label, 20, 0.1, 128);

mat pred = model.predict(test_data);

cout << "Test accuracy: " << accuracy(pred, test_label) << endl;
```

The expected results for this example are:

```
MNIST EXAMPLE
Loading: ../examples/mnist_data/train-images.idx3-ubyte
Loading: ../examples/mnist_data/t10k-images.idx3-ubyte
Loading: ../examples/mnist_data/t10k-labels.idx1-ubyte
Loading: ../examples/mnist_data/train-labels.idx1-ubyte
Training...
Epoch: 0 Loss: 0.474713
Epoch: 1 Loss: 0.266864
Epoch: 2 Loss: 0.217499
Epoch: 3 Loss: 0.184602
Epoch: 4 Loss: 0.160319
Epoch: 5 Loss: 0.141671
Epoch: 6 Loss: 0.126914
Epoch: 7 Loss: 0.114926
Epoch: 8 Loss: 0.104993
Epoch: 9 Loss: 0.096605
Epoch: 10 Loss: 0.0894333
Epoch: 11 Loss: 0.0832029
Epoch: 12 Loss: 0.0777446
Epoch: 13 Loss: 0.0729027
Epoch: 14 Loss: 0.0685371
Epoch: 15 Loss: 0.0646405
Epoch: 16 Loss: 0.0610888
Epoch: 17 Loss: 0.0578651
Epoch: 18 Loss: 0.054893
Epoch: 19 Loss: 0.0522089
Test accuracy: 0.9749
```

## Todo
* Convolution layer
* Maxpooling
* Flatten layer
* Dropout