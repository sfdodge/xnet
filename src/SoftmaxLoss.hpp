#pragma once

#include <string>

#include "mat.hpp"
#include "BaseLayer.hpp"

class SoftmaxLoss : public BaseLayer {
 public:
    SoftmaxLoss(string _name = "smloss");

    mat x_store;
    mat t_store;

    mat forward(mat x, mat t);
    mat forward_softmax(mat x);
    mat backward();
};
