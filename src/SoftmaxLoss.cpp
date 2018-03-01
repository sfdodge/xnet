#include "SoftmaxLoss.hpp"

SoftmaxLoss::SoftmaxLoss(string _name) {
    name = _name;
}

mat SoftmaxLoss::forward_softmax(mat x) {
    // numerically stable softmax
    x = x - amax(x);
    mat s = sum(exp(x), {1});
    s = view(s, all(), newaxis());
    x = exp(x) / s;

    return x;
}

mat SoftmaxLoss::forward(mat x, mat t) {
    t_store = t;

    mat y = forward_softmax(x);

    x_store = y;

    return -sum(log(y) * t) / float(x.shape()[0]);
}

mat SoftmaxLoss::backward() {
    return (x_store - t_store) / float(x_store.shape()[0]);
}
