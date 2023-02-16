#pragma once

#include <cmath>

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "sampler.hpp"


namespace eignn::module {

class Module {
public:
    bool freeze = false;
    virtual void forward(const Eigen::MatrixXf &x) = 0;
    virtual void reverse(const Eigen::MatrixXf &y_bar) = 0;
    const Eigen::MatrixXf &y() const { return m_y; }
    const Eigen::MatrixXf &x_bar() const { return m_x_bar; };
    virtual ~Module() = default;

protected:
    Eigen::MatrixXf m_y;
    Eigen::MatrixXf m_x_bar;
};


class ReLU: public Module {
public:
    Eigen::MatrixXf partial_x;

    ~ReLU() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        m_y = x.cwiseMax(0);
        partial_x.resize(x.rows(), x.cols());
        for (int col = 0; col < x.cols(); ++col)
            for (int row = 0; row < x.rows(); ++row)
                partial_x(row,col) = float(x(row,col)>0);
    }

    void reverse(const Eigen::MatrixXf &y_bar) override {
        m_x_bar = y_bar.cwiseProduct(partial_x);
    }
};


class Sigmoid: public Module {
public:
    Eigen::MatrixXf partial_x;

    ~Sigmoid() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        m_y = x;
        m_y = ((-1.f*m_y).array().exp()+1.f).inverse().matrix();
        partial_x = (m_y.array() * ((-1.f*m_y).array()+1.f)).matrix();
    }

    void reverse(const Eigen::MatrixXf &y_bar) override {
        m_x_bar = y_bar.cwiseProduct(partial_x);
    }
};


template<bool bias_ = true>
class Linear: public Module {
public:
    Eigen::MatrixXf mat;
    Eigen::VectorXf bias;

    Eigen::MatrixXf partial_mat;

    Linear(int in_dim, int out_dim) {
        GaussSampler<float> gs{/*mean=*/0.f,/*stddev=*/.5f};

        mat.setZero(out_dim,in_dim);
        if constexpr(bias_)
            bias.setZero(out_dim);

        for (int col = 0; col < mat.cols(); ++col)
            for (int row = 0; row < mat.rows(); ++row)
                mat(row,col) = gs.sample();
    }

    ~Linear() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        m_y = mat * x;
        if constexpr(bias_)
            m_y.colwise() += bias;

        partial_mat = x.transpose();
    }

    void reverse(const Eigen::MatrixXf &y_bar) override {
        m_x_bar = mat.transpose() * y_bar;

        if (freeze)
            return;

        mat -= y_bar * partial_mat;
        if constexpr(bias_)
            bias -= y_bar.rowwise().sum();
    }
};


class Sequential: public Module {
public:
    std::vector<std::unique_ptr<Module>> modules;

    ~Sequential() override = default;

    size_t size() const { return modules.size(); }

    void forward(const Eigen::MatrixXf &x) override {
        assert(!modules.empty());
        assert(modules[0]);
        modules[0]->forward(x);
        for (int ii = 0; ii < size()-1; ++ii) {
            auto &m0 = modules[ii];
            auto &m1 = modules[ii+1];
            assert(m0);
            assert(m1);
            m1->forward(m0->y());
        }
        m_y = modules[size()-1]->y();
    }

    void reverse(const Eigen::MatrixXf &y_bar) override {
        assert(!modules.empty());
        assert(modules[size()-1]);
        modules[size()-1]->reverse(y_bar);
        for (int ii = 0; ii < size()-1; ++ii) {
            auto &m0 = modules[modules.size()-ii-2];
            auto &m1 = modules[modules.size()-ii-1];
            assert(m0);
            assert(m1);
            m0->reverse(m1->x_bar());
        }
        m_x_bar = modules[0]->x_bar();
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