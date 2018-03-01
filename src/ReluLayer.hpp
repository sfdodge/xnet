#pragma once

#include <string>

#include "mat.hpp"
#include "BaseLayer.hpp"

class ReluLayer : public BaseLayer {
 public:
    ReluLayer(string _name = "Relu");

    mat y;

    mat forward(mat x);
    mat backward(mat y_grad);
};
