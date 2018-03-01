#include <iostream>
#include "../src/mat.hpp"

#include "catch.hpp"
#include "../src/SoftmaxLoss.hpp"

using namespace std;
using namespace xt;

TEST_CASE("Test softmax forward", "[sm_forward]") {
    SoftmaxLoss loss;

    mat x = {{1, 2, 3, 4}};

    mat y = loss.forward_softmax(x);

    mat should_be = {{0.0320586, 0.087144, 0.2368828, 0.643914}};

    REQUIRE(allclose(should_be, y));
}

TEST_CASE("Test softmax loss forward", "[smloss_forward]") {
    SoftmaxLoss loss;

    mat x = {{1, 2, 3, 4}};
    mat t = {{0, 1, 0, 0}};

    mat y = loss.forward(x, t);

    float should_be = 2.44018;

    REQUIRE(allclose(should_be, y));

    mat x2 = {{1, 2, 3, 4},
                       {5, 6, 7, 8}};
    mat t2 = {{0, 1, 0, 0},
                       {1, 0, 0, 0}};

    mat y2 = loss.forward(x2, t2);

    float should_be2 = (2.44018 + 3.44018) / 2.0;

    REQUIRE(allclose(should_be2, y2));
}

TEST_CASE("Test softmax loss backward", "[smloss_backward]") {
    SoftmaxLoss loss;

    mat x = {{1, 2, 3, 4},
             {5, 6, 7, 8}};
    mat t = {{0, 1, 0, 0},
             {1, 0, 0, 0}};

    mat out1 = loss.forward(x, t);
    mat x_grad = loss.backward();

    // test x_grad
    float h = 0.001;
    mat x2 = x;
    x2[0,0] += h;
    mat out2 = loss.forward(x2, t);
    mat diff = (out2 - out1) / h;

    REQUIRE(allclose(diff, x_grad[0,0],1e-05,1e-03));

    mat x3 = x;
    x3[0,1] += h;
    mat out3 = loss.forward(x3, t);
    mat diff2 = (out3 - out1) / h;

    REQUIRE(allclose(diff2, x_grad[0,1],1e-05,1e-03));
}
