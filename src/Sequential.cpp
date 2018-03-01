#include "Sequential.hpp"

Sequential::Sequential(vector<BaseLayer*> _layers, BaseLayer* _loss) {
    layers = _layers;
    loss = _loss;
}

mat Sequential::forward(mat x) {
    mat h = x;

    for (auto l : layers) {
        h = l->forward(h);
    }

    return h;
}

mat Sequential::forward(mat x, mat t) {
    mat h = forward(x);

    h = loss->forward(h, t);

    return h;
}

mat Sequential::backward() {
    mat h = loss->backward();

    for (int i = layers.size()-1; i >= 0; i--) {
        h = layers[i]->backward(h);
    }

    return h;
}

void Sequential::update_param(float lr) {
    for (auto l : layers) {
        l->update_param(lr);
    }
}

void Sequential::fit(mat x, mat y, int epochs,
                     float lr, int batch_size) {
    for (int e=0; e < epochs; e++) {
        float sum_loss = 0;
        int b_i = 0;

        for (int i=0; i < x.shape()[0]; i = i+batch_size) {
            mat batch_x = view(x, range(i, i+batch_size), all());
            mat batch_y = view(y, range(i, i+batch_size), all());

            mat loss = forward(batch_x, batch_y);

            sum_loss += loss[0];
            b_i++;

            backward();
            update_param(lr);
        }

        cout << "Epoch: " << e << " Loss: " << sum_loss / float(b_i) << endl;
    }
}

mat Sequential::predict(mat x) {
    // Should do this batchwise if dataset is too big
    return argmax(forward(x), 1);
}
