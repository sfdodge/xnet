#include "FullLayer.hpp"

FullLayer::FullLayer(int _n_i, int _n_o, string _name) {
    n_o = _n_o;
    n_i = _n_i;

    name = _name;

    float a = sqrt(2.0 / float(n_i + n_o));

    W = random::randn<float>({n_o, n_i}) * a;
    b = zeros<float>({1, n_o});

    W_grad = zeros<float>({n_o, n_i});
    b_grad = zeros<float>({n_o, n_i});
}

mat FullLayer::forward(mat x) {
    x_store = x;

    mat y = linalg::dot(x, transpose(W));

    y = y + b;

    return y;
}

mat nan2num(mat in) {
    mat locs = isnan(in);
}

mat FullLayer::backward(mat y_grad) {
    mat x_grad = linalg::dot(y_grad, W);

    W_grad = eval(linalg::dot(transpose(y_grad), x_store));

    // for some reason W_grad has nans, but not if I do this
    // maybe is a bug in xtensor?
    mat try_again2 = transpose(linalg::dot(transpose(x_store), y_grad));
    mat try_again = transpose(linalg::dot(transpose(x_store), y_grad));
    W_grad = try_again;

    mat b_gr = sum(y_grad, {0});
    b_gr = view(b_gr, all(), newaxis());
    b_gr = transpose(b_gr);
    b_grad = b_gr;

    return x_grad;
}

void FullLayer::update_param(float lr) {
    mat newW = W - lr * W_grad;
    W = newW;
    mat newB = b - lr * b_grad;
    b = newB;
}
