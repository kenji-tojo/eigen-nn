#pragma once

#include <functional>

#include <Eigen/Dense>


namespace eignn::ad {

class MatrixXf {
public:
    Eigen::MatrixXf m;
    Eigen::MatrixXf grad;

    void resize(int rows, int cols) {
        m.resize(rows,cols);
        grad.resize(rows,cols);
    }

    void init(std::function<float(int)> &&init_fn) {
        for (int ii = 0; ii < m.size(); ++ii)
            m.data()[ii] = init_fn(ii);
        grad.setZero();
    }

    void set_zero() {
        m.setZero();
        grad.setZero();
    }

    void descent() {
        if (m.size() > 0)
            m -= grad;
    }
};

} // namespace eignn::ad