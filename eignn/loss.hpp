#pragma once


#include <Eigen/Dense>

namespace eignn {

class Loss {
public:
    virtual void eval(
            const Eigen::MatrixXf &output,
            const Eigen::MatrixXf &target,
            float &val,
            Eigen::MatrixXf &d_loss
    ) = 0;
};

class MSELoss: public Loss {
public:
    void eval(
            const Eigen::MatrixXf &output,
            const Eigen::MatrixXf &target,
            float &val,
            Eigen::MatrixXf &adjoint
    ) override {
        using namespace Eigen;
        auto n = output.size();
        MatrixXf diff = output - target;
        adjoint = diff / float(n);
        val = diff.cwiseProduct(diff).sum() / float(n);
    }
};

} // namespace eignn