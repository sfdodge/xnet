mkdir mnist_data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P mnist_data/
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P mnist_data/
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P mnist_data/
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P mnist_data/
gunzip mnist_data/train-images-idx3-ubyte.gz
gunzip mnist_data/t10k-images-idx3-ubyte.gz
gunzip mnist_data/train-labels-idx1-ubyte.gz
gunzip mnist_data/t10k-labels-idx1-ubyte.gz
