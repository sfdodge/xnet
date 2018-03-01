#pragma once

#include <vector>

#include "mat.hpp"
#include "BaseLayer.hpp"

using std::vector;

class Sequential : public BaseLayer {
 public:
    Sequential(vector<BaseLayer*> _layers, BaseLayer* loss);

    vector<BaseLayer*> layers;
    BaseLayer* loss;

    mat forward(mat x);
    mat forward(mat x, mat t);
    mat backward();
    void update_param(float lr);
    void fit(mat x, mat y, int epochs = 10,
             float lr = 0.1, int batch_size = 128);
    mat predict(mat x);
};
