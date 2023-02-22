#pragma once

#include <iostream>
#include <limits>
#include <memory>

#include "ad.hpp"


namespace eignn {

class Optimizer {
public:
    float learning_rate = 1e-1f;

    virtual void add_parameters (std::vector<std::shared_ptr<ad::Matrixf>> &&_parameters) {
        auto idx = parameters.size();
        parameters.resize(parameters.size()+_parameters.size());
        for (const auto &p: _parameters) {
            assert(p);
            parameters[idx] = p;
            idx += 1;
        }
    }

    virtual void descent() {
        for (auto &p: parameters)
            p->descent(learning_rate);
    }

protected:
    std::vector<std::shared_ptr<ad::Matrixf>> parameters;
};


class SGD: public Optimizer {
public:
    float beta = .9f;

    void add_parameters (std::vector<std::shared_ptr<ad::Matrixf>> &&_parameters) override {
        auto idx = parameters.size();
        parameters.resize(parameters.size()+_parameters.size());
        momentum.resize(momentum.size()+_parameters.size());
        for (const auto &p: _parameters) {
            assert(p);
            parameters[idx] = p;
            momentum[idx].resize(p->m.rows(), p->m.cols());
            idx += 1;
        }

        reset();
    }

    void reset() {
        for (auto &m: momentum)
            m.setZero();

        steps = 1;
        beta_acc = 1.f;
    }

    void descent() override {
        if (steps == std::numeric_limits<unsigned int>::max())
            return;

        steps += 1;
        beta_acc *= beta;

        for (int ii = 0; ii < parameters.size(); ++ii) {
            auto &m = momentum[ii];
            auto &g = parameters[ii]->grad;

            m = beta * m + (1.f-beta) * g;
            g = m / (1.f-beta_acc);

            parameters[ii]->descent(learning_rate);
        }
    }

private:
    std::vector<Eigen::MatrixXf> momentum;
    unsigned int steps = 1;
    float beta_acc = 1.f;
};


class Adam: public Optimizer {
public:
    float beta1 = .9f;
    float beta2 = .99f;
    float eps = 1e-15f;

    void add_parameters (std::vector<std::shared_ptr<ad::Matrixf>> &&_parameters) override {
        auto idx = parameters.size();
        parameters.resize(parameters.size()+_parameters.size());
        momentum1.resize(momentum1.size()+_parameters.size());
        momentum2.resize(momentum2.size()+_parameters.size());
        for (const auto &p: _parameters) {
            assert(p);
            parameters[idx] = p;
            momentum1[idx].resize(p->m.rows(), p->m.cols());
            momentum2[idx].resize(p->m.rows(), p->m.cols());
            idx += 1;
        }

        reset();
    }

    void reset() {
        for (auto &m1: momentum1)
            m1.setZero();

        for (auto &m2: momentum2)
            m2.setZero();

        steps = 1;
        beta1_acc = 1.f;
        beta2_acc = 1.f;
    }

    void descent() override {
        if (steps == std::numeric_limits<unsigned int>::max())
            return;

        steps += 1;
        beta1_acc *= beta1;
        beta2_acc *= beta2;

        for (int ii = 0; ii < parameters.size(); ++ii) {
            auto &m1 = momentum1[ii];
            auto &m2 = momentum2[ii];
            auto &g = parameters[ii]->grad;

            m1 = beta1 * m1 + (1.f-beta1) * g;
            m2 = beta2 * m2 + (1.f-beta2) * g.array().square().matrix();

            const auto m1_corr = (m1 / (1.f - beta1_acc)).array();
            const auto m2_corr = (m2 / (1.f - beta2_acc)).array();

            g = (m1_corr / (m2_corr.sqrt()+eps)).matrix();

            parameters[ii]->descent(learning_rate);
        }
    }

private:
    std::vector<Eigen::MatrixXf> momentum1;
    std::vector<Eigen::MatrixXf> momentum2;
    unsigned int steps = 1;
    float beta1_acc = 1.f;
    float beta2_acc = 1.f;
};

} // eignn