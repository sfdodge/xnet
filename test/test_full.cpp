#include <iostream>
#include "../src/mat.hpp"

#include "catch.hpp"
#include "../src/FullLayer.hpp"

using namespace std;
using namespace xt;

TEST_CASE("Test full forward", "[full_forward]") {
    FullLayer layer(2, 3);

    mat w = {{1,2,3}, {4,5,6}};
    w = transpose(w);
    mat b = {{2, 4, 6},};
    mat x = {{1, 2},
                   {-1, 1}};

    layer.W = w;
    layer.b = b;

    mat out = layer.forward(x);

    mat should_be = {{11,16,21}, {5,7,9}};

    REQUIRE(allclose(should_be, out));
}

TEST_CASE("Test full backward", "[full_backward]") {
    FullLayer layer(2, 3);

    mat w = {{1,2,3}, {4,5,6}};
    w = transpose(w);
    mat b = {{2, 4, 6},};
    mat x  = {{1,2}};

    layer.W = w;
    layer.b = b;

    mat out1 = layer.forward(x);

    mat y = ones<float>({1,3});
    mat x_grad = layer.backward(y);

    // test x_grad
    float h = 0.001;
    mat x2 = x;
    x2[0,0] += h;
    mat out2 = layer.forward(x2);
    mat diff = (out2 - out1) / h;

    REQUIRE(allclose(sum(diff), x_grad[0,0],1e-05,1e-03));

    mat x3 = x;
    x3[0,1] += h;
    mat out3 = layer.forward(x3);
    mat diff2 = (out3 - out1) / h;

    REQUIRE(allclose(sum(diff2), x_grad[0,1],1e-05,1e-03));

    // test w_grad
    for (int i=0; i<w.shape()[0]; i++) {
        for (int j=0; j<w.shape()[1]; j++) {
            w[i,j] += h;
            layer.W = w;

            mat out_new = layer.forward(x);
            mat diff_new = (out_new - out1) / h;

            w[i,j] -= h;
            layer.W = w;

            REQUIRE(allclose(sum(diff_new),
                             layer.W_grad[i,j],1e-05,1e-03));
        }
    }

    // test b_grad
    for (int i=0; i<b.shape()[0]; i++) {
        b[i] += h;
        layer.b = b;

        mat out_new = layer.forward(x);
        mat diff_new = (out_new - out1) / h;

        REQUIRE(allclose(sum(diff_new),
                         layer.b_grad[i],1e-05,1e-03));

        b[i] -= h;
    }
}

TEST_CASE("Test full update", "[full_update]") {
    FullLayer layer(2, 3);

    mat w = {{1,2,3}, {4,5,6}};
    w = transpose(w);
    mat b = {{2, 4, 6},};
    mat x  = {{1,2}};

    layer.W = w;
    layer.b = b;

    mat out = layer.forward(x);

    mat y = ones<float>({1,3});
    mat x_grad = layer.backward(y);

    // updating grad should decrease value
    mat old_val = sum(out);
    bool valid = true;
    for (int i=0; i<10; i++) {
        layer.update_param(0.1);
        mat new_val = sum(layer.forward(x));
        layer.backward(y);

        if (new_val[0] >= old_val[0]) {
            valid = false;
        }
        old_val = new_val;
    }

    REQUIRE(valid);
}

