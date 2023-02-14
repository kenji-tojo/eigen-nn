#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>


namespace eignn::module {

class Module {
public:
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
        for (int ii = 0; ii < in.rows(); ++ii) {
            for (int jj = 0; jj < in.cols(); ++jj) {
                in(ii,jj) = float(memory(ii,jj)>0)*out(ii,jj);
            }
        }
    }
};


template<bool bias_ = true>
class Linear: public Module {
public:
    Eigen::MatrixXf mat;
    Eigen::VectorXf bias;

    Linear(int in_dim, int out_dim) {
        mat.setZero(out_dim,in_dim);
        if constexpr(bias_)
            bias.setZero(out_dim);
    }

    ~Linear() override = default;

    void forward(Eigen::MatrixXf in, Eigen::MatrixXf &out) override {
        out = mat * in;
        if constexpr(bias_) {
            for (int ii = 0; ii < out.cols(); ++ii)
                out.col(ii) += bias;
        }
        memory = std::move(in);
    }

    void reverse(Eigen::MatrixXf out, Eigen::MatrixXf &in) override {
        assert(memory.size() > 0);
        const Eigen::VectorXf out_red = out.rowwise().sum();
        if constexpr(bias_)
            bias += out_red;
        mat += out_red * memory.rowwise().sum().transpose();
        in = mat.transpose() * out;
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
    }

private:
    void add_dense(int in_dim, int out_dim, int hidden_dim, bool relu) {
        modules.push_back(std::make_unique<Linear<false>>(in_dim, hidden_dim));
        modules.push_back(std::make_unique<Linear<true>>(hidden_dim, out_dim));
        if (relu)
            modules.push_back(std::make_unique<ReLU>());
    }
};


} // namespace eignn::module