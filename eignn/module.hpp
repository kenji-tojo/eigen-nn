#pragma once

#include <cmath>

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "sampler.hpp"
#include "ad.hpp"


namespace eignn::module {

class Module {
public:
    bool freeze = false;
    Eigen::MatrixXf y;
    Eigen::MatrixXf x_adj;
    virtual void forward(const Eigen::MatrixXf &x) = 0;
    virtual void adjoint(const Eigen::MatrixXf &y_adj) = 0;
    virtual ~Module() = default;
};


class ReLU: public Module {
public:
    Eigen::MatrixXf d_x;

    ~ReLU() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        y = x.cwiseMax(0);
        d_x.resize(x.rows(), x.cols());
        for (int col = 0; col < x.cols(); ++col)
            for (int row = 0; row < x.rows(); ++row)
                d_x(row,col) = float(x(row,col)>0);
    }

    void adjoint(const Eigen::MatrixXf &y_adj) override {
        x_adj = y_adj.cwiseProduct(d_x);
    }
};


class Sigmoid: public Module {
public:
    Eigen::MatrixXf d_x;

    ~Sigmoid() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        y = x;
        y = ((-1.f*y).array().exp()+1.f).inverse().matrix();
        d_x = (y.array() * ((-1.f*y).array()+1.f)).matrix();
    }

    void adjoint(const Eigen::MatrixXf &y_adj) override {
        x_adj = y_adj.cwiseProduct(d_x);
    }
};


template<bool bias_ = true>
class Linear: public Module {
public:
    Eigen::MatrixXf mat;
    Eigen::VectorXf bias;

    Eigen::MatrixXf d_mat;

    Linear(int in_dim, int out_dim) {
        GaussSampler<float> gs{/*mean=*/0.f,/*stddev=*/.5f};

        mat.setZero(out_dim,in_dim);
        if constexpr(bias_)
            bias.setZero(out_dim,1);

        for (int col = 0; col < mat.cols(); ++col)
            for (int row = 0; row < mat.rows(); ++row)
                mat(row,col) = gs.sample();
    }

    ~Linear() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        y = mat * x;
        if constexpr(bias_)
            y.colwise() += bias;

        d_mat = x.transpose();
    }

    void adjoint(const Eigen::MatrixXf &y_adj) override {
        x_adj = mat.transpose() * y_adj;

        if (freeze)
            return;

        mat -= y_adj * d_mat;
        if constexpr(bias_)
            bias -= y_adj.rowwise().sum();
    }
};


class Sequential: public Module {
public:
    std::vector<std::unique_ptr<Module>> modules;

    ~Sequential() override = default;

    [[nodiscard]] size_t size() const { return modules.size(); }

    void forward(const Eigen::MatrixXf &x) override {
        assert(!modules.empty());
        assert(modules[0]);
        modules[0]->forward(x);
        for (int ii = 0; ii < size()-1; ++ii) {
            auto &m0 = modules[ii];
            auto &m1 = modules[ii+1];
            assert(m0);
            assert(m1);
            m1->forward(m0->y);
        }
        y = modules[size()-1]->y;
    }

    void adjoint(const Eigen::MatrixXf &y_adj) override {
        assert(!modules.empty());
        assert(modules[size()-1]);
        modules[size()-1]->adjoint(y_adj);
        for (int ii = 0; ii < size()-1; ++ii) {
            auto &m0 = modules[modules.size()-ii-2];
            auto &m1 = modules[modules.size()-ii-1];
            assert(m0);
            assert(m1);
            m0->adjoint(m1->x_adj);
        }
        x_adj = modules[0]->x_adj;
    }
};


class MLP: public Sequential {
public:
    MLP(int in_dim, int out_dim, int hidden_dim, int hidden_depth) {
        add_dense(in_dim, hidden_dim, /*relu=*/true);
        for (int dd = 0; dd < hidden_depth; ++dd)
            add_dense(hidden_dim, hidden_dim, /*relu=*/true);
        add_dense(hidden_dim, out_dim, /*relu=*/false);
        modules.push_back(std::make_unique<Sigmoid>());
    }

private:
    void add_dense(int in_dim, int out_dim, bool relu) {
        modules.push_back(std::make_unique<Linear<true>>(in_dim, out_dim));
        if (relu)
            modules.push_back(std::make_unique<ReLU>());
    }
};


} // namespace eignn::module