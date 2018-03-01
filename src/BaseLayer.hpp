#ifndef BASELAYER_H
#define BASELAYER_H

#include <string>

#include "mat.hpp"

using std::cout;
using std::endl;
using std::string;

class BaseLayer {
 public:
    BaseLayer() {}

    string name = "";

    virtual mat forward(mat x) {}
    virtual mat forward(mat x, mat t) {}
    virtual mat backward(mat y_grad) {}
    virtual mat backward() {}
    virtual void update_param(float lr) {}
};

#endif
