#include <iostream>
#include "../src/mat.hpp"

#include "catch.hpp"
#include "../src/Sequential.hpp"
#include "../src/FullLayer.hpp"
#include "../src/ReluLayer.hpp"
#include "../src/SoftmaxLoss.hpp"

using namespace std;

Sequential setUp() {
    /**
        Set up a small sequential network
    **/
    BaseLayer* layer1 = new FullLayer(5, 10);
    BaseLayer* relu1 = new ReluLayer();
    BaseLayer* layer2 = new FullLayer(10, 2);
    BaseLayer* softmax = new SoftmaxLoss();

    Sequential model = Sequential({layer1, relu1, layer2},
                       softmax);

    return model;
}

TEST_CASE("Test sequential backward", "[seq_backward]") {
    // make some fake input
    mat x ={{0.5, 0.2, 0.1, 0.3, 0.7},
        {0.3, 0.1, 0.05, 0.8, 0.9}};
    mat y = {{0, 1},
             {1, 0}};

    Sequential model = setUp();

    mat out = model.forward(x, y);
    mat grads = model.backward();

    // test some gradients at the input
    double h = 0.001;

    for (int i=0; i<x.shape()[0]; i++) {
        for (int j=0; j<x.shape()[1]; j++) {
            mat new_x = x;
            new_x[i,j] += h;
            mat out2 = model.forward(new_x, y);
            mat diff = (out2 - out) / h;

            REQUIRE(allclose(diff, grads[i,j],1e-05,1e-03));
        }
    }
}

TEST_CASE("Test seq update", "[seq_update]") {
    mat x ={{0.5, 0.2, 0.1, 0.3, 0.7},
        {0.3, 0.1, 0.05, 0.8, 0.9}};
    mat y = {{0, 1},
             {1, 0}};

    Sequential model = setUp();

    mat out = model.forward(x, y);
    mat grads = model.backward();

    // updating grad should decrease value
    mat old_val = out;
    bool valid = true;
    for (int i=0; i<10; i++) {
        model.update_param(0.1);
        mat new_val = model.forward(x, y);
        model.backward();

        if (new_val[0] >= old_val[0]) {
            valid = false;
        }
        old_val = new_val;

        // cout << new_val << endl;
    }

    REQUIRE(valid);
}
