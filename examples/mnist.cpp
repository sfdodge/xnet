#include <iostream>
#include <vector>
#include <fstream>
#include "../src/mat.hpp"

#include "../src/Sequential.hpp"
#include "../src/FullLayer.hpp"
#include "../src/ReluLayer.hpp"
#include "../src/SoftmaxLoss.hpp"

using std::cout;
using std::endl;
using std::ifstream;

const int N_CLASSES = 10;
const string DATA_PATH = "../examples/mnist_data/";

int ReverseInt(int i);
mat ReadMNISTData(string type);
mat ReadMNISTLabel(string type);
double accuracy(mat pred, mat gt);

int main() {
    cout << "MNIST EXAMPLE" << endl;

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
}

double accuracy(mat pred, mat gt) {
    gt = argmax(gt, 1);

    double acc = sum(equal(pred, gt))[0];
    acc = acc / (double)gt.shape()[0];

    return acc;
}

int ReverseInt(int i) {
    // modified from: https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/

    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i&255;
    ch2 = (i>>8)&255;
    ch3 = (i>>16)&255;
    ch4 = (i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

mat ReadMNISTData(string type) {
    // modified from: https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/

    mat out;
    string path;
    if (type == "test") {
        path = DATA_PATH + "t10k-images-idx3-ubyte";
    } else if (type == "train") {
        path = DATA_PATH + "train-images-idx3-ubyte";
    } else {
        assert(false);
    }

    ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        cout << "Loading: " << path << endl;

        int magic_number = 0;
        int n_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number= ReverseInt(magic_number);

        file.read((char*)&n_images, sizeof(n_images));
        n_images = ReverseInt(n_images);

        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);

        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        out = zeros<float>({n_images, n_rows * n_cols});

        for (int i = 0; i < n_images; i++) {
            for (int r = 0; r < n_rows; r++) {
                for (int c=0; c < n_cols; c++) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    out(i, (n_rows*r)+c) = ((float)temp) / 255.0;
                }
            }
        }
    } else {
        std::cerr << "Unable to load file: " << path << endl;
        std::terminate();
    }

    return out;
}

mat ReadMNISTLabel(string type) {
    // modified from: https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/

    mat out;
    string path;
    if (type == "test") {
        path = DATA_PATH + "t10k-labels-idx1-ubyte";
    } else if (type == "train") {
        path = DATA_PATH + "train-labels-idx1-ubyte";
    } else {
        assert(false);
    }
    
    ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        cout << "Loading: " << path << endl;
        int magic_number = 0;
        int n_images = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&n_images, sizeof(n_images));
        n_images = ReverseInt(n_images);

        out = zeros<float>({n_images, N_CLASSES});

        for (int i = 0; i < n_images; i++) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            out(i, (int)temp) = 1;
        }
    } else {
        std::cerr << "Unable to load file: " << path << endl;
        std::terminate();
    }

    return out;
}
