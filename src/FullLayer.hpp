#pragma once

#include <cmath>
#include <string>

#include "mat.hpp"
#include "BaseLayer.hpp"

class FullLayer : public BaseLayer {
 public:
    FullLayer(int _n_i, int _n_o, string _name = "Full");

    int n_i;
    int n_o;

    mat W;
    mat b;
    mat W_grad;
    mat b_grad;

    mat x_store;

    mat forward(mat x);
    mat backward(mat y_grad);
    void update_param(float lr);
};
