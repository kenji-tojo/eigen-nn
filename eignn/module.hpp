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
    virtual void forward(Eigen::MatrixXf in, Eigen::MatrixXf &out) = 0;
    virtual void reverse(Eigen::MatrixXf out, Eigen::MatrixXf &in) = 0;
    virtual ~Module() = default;
    Eigen::MatrixXf memory;
};


class ReLU: public Module {
public:
    ~ReLU() override = default;

    void forward(Eigen::MatrixXf in, Eigen::MatrixXf &out) override {
        out = in.cwiseMax(0);
        memory = std::move(in); // for reverse-mode evaluation
    }

    void reverse(Eigen::MatrixXf out, Eigen::MatrixXf &in) override {
        assert(memory.size() > 0);
        in.resize(memory.rows(),memory.cols());
        for (int ii = 0; ii < in.cols(); ++ii) {
            for (int jj = 0; jj < in.rows(); ++jj) {
                in(jj,ii) = float(memory(jj,ii)>0)*out(jj,ii);
            }
        }
    }
};


class Sigmoid: public Module {
public:
    ~Sigmoid() override = default;

    void forward(Eigen::MatrixXf in, Eigen::MatrixXf &out) override {
        const auto ones = Eigen::MatrixXf::Ones(in.rows(), in.cols());

        out = in;
        for (int ii = 0; ii < out.cols(); ++ii) {
            for (int jj = 0; jj < out.rows(); ++jj) {
                out(jj,ii) = 1.f/(1.f+std::exp(-out(jj,ii)));
            }
        }
        memory = out.cwiseProduct(ones-out);
    }

    void reverse(Eigen::MatrixXf out, Eigen::MatrixXf &in) override {
        in = memory.cwiseProduct(out);
    }
};


template<bool bias_ = true>
class Linear: public Module {
public:
    Eigen::MatrixXf mat;
    Eigen::VectorXf bias;

    Linear(int in_dim, int out_dim) {
        Sampler<float> sampler;
        mat.setZero(out_dim,in_dim);
        if constexpr(bias_)
            bias.setZero(out_dim);

        for (int ii = 0; ii < mat.rows(); ++ii) {
            for (int jj = 0; jj < mat.cols(); ++jj) {
                mat(ii,jj) = 2.f*(sampler.sample()-.5f);
            }
            if constexpr(bias_)
                bias[ii] = 2.f*(sampler.sample()-.5f);
        }
    }

    ~Linear() override = default;

    void forward(Eigen::MatrixXf in, Eigen::MatrixXf &out) override {
        out = mat * in;
        if constexpr(bias_) {
            out.colwise() += bias;
        }
        memory = std::move(in);
    }

    void reverse(Eigen::MatrixXf out, Eigen::MatrixXf &in) override {
        assert(memory.size() > 0);
        in = mat.transpose() * out;
        if (freeze)
            return;
        mat += out * memory.transpose();
        if constexpr(bias_)
            bias += out.rowwise().sum();
    }
};


class Sequential: public Module {
public:
    std::vector<std::unique_ptr<Module>> modules;

    ~Sequential() override = default;

    void forward(Eigen::MatrixXf in, Eigen::MatrixXf &out) override {
        assert(!modules.empty());
        for (auto &m: modules) {
            assert(m);
            m->forward(in,in);
        }
        out = in;
    }

    void reverse(Eigen::MatrixXf out, Eigen::MatrixXf &in) override {
        assert(!modules.empty());
        for (int ii = 0; ii < modules.size(); ++ii) {
            auto &m = modules[modules.size()-ii-1];
            assert(m);
            m->reverse(out,out);
        }
        in = out;
    }
};


class MLP: public Sequential {
public:
    MLP(int in_dim, int out_dim, int hidden_dim, int hidden_depth) {
        add_dense(in_dim, hidden_dim, hidden_dim, /*relu=*/true);
        for (int dd = 0; dd < hidden_depth; ++dd)
            add_dense(hidden_dim, hidden_dim, hidden_dim, /*relu=*/true);
        add_dense(hidden_dim, out_dim, hidden_dim, /*relu=*/false);
        modules.push_back(std::make_unique<Sigmoid>());
    }

private:
    void add_dense(int in_dim, int out_dim, int hidden_dim, bool relu) {
        modules.push_back(std::make_unique<Linear<true>>(in_dim, out_dim));
//        modules.push_back(std::make_unique<Linear<false>>(in_dim, hidden_dim));
//        modules.push_back(std::make_unique<Linear<true>>(hidden_dim, out_dim));
        if (relu)
            modules.push_back(std::make_unique<ReLU>());
    }
};


} // namespace eignn::module