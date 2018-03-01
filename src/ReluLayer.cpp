#include "ReluLayer.hpp"

ReluLayer::ReluLayer(string _name) {
    name = _name;
}

mat ReluLayer::forward(mat x) {
    y = x * (x > 0);
    return y;
}

mat ReluLayer::backward(mat y_grad) {
    return y_grad * (y > 0);
}
