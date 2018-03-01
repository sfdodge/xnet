#include <iostream>
#include "../src/mat.hpp"

#include "catch.hpp"
#include "../src/ReluLayer.hpp"

using namespace std;
using namespace xt;

TEST_CASE("Test relu forward", "[relu_forward]") {
    ReluLayer layer;

    mat x = {{-1, 1, 0, 3}};
    x = transpose(x);

    mat y = layer.forward(x);

    mat should_be = {{0,1,0,3}};
    should_be = transpose(should_be);

    REQUIRE(allclose(should_be, y));
}


TEST_CASE("Test relu backward", "[relu_backward]") {
    ReluLayer layer;

    mat x = {{-1, 1, 0, 3}};
    x = transpose(x);
    mat y = layer.forward(x);

    mat z = ones<float>({4, 1});
    mat x_grad = layer.backward(z);

    mat should_be = {{0,1,0,1}};
    should_be = transpose(should_be);

    REQUIRE(allclose(should_be, x_grad));
}

TEST_CASE("Test relu backward numerically", "[relu_backward2]") {
    ReluLayer layer;

    mat x = {{-1, 1, 0, 3}};
    x = transpose(x);
    mat y = layer.forward(x);

    mat z = ones<float>({4, 1});
    mat x_grad = layer.backward(z);

    float h = 0.0001;
    mat x2 = x;
    x2[0,0] += h;
    mat out2 = layer.forward(x2);
    mat diff2 = (out2 - y) / h;

    REQUIRE(allclose(sum(diff2), x_grad[0],1e-05,1e-03));

    mat x3 = x;
    x3[0,1] += h;
    mat out3 = layer.forward(x3);
    mat diff3 = (out3 - y) / h;

    REQUIRE(allclose(sum(diff3), x_grad[1],1e-05,1e-03));

    mat x4 = x;
    x4[0,2] += h;
    mat out4 = layer.forward(x4);
    mat diff4 = (out4 - y) / h;

    REQUIRE(allclose(sum(diff4), x_grad[2],1e-05,1e-03));

    mat x5 = x;
    x5[0,3] += h;
    mat out5 = layer.forward(x5);
    mat diff5 = (out5 - y) / h;

    REQUIRE(allclose(sum(diff5), x_grad[3],1e-05,1e-03));
}
