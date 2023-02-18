#pragma once

#include <limits>
#include <memory>

#include "ad.hpp"


namespace eignn {

class Optimizer {
public:
    virtual void add_parameters (std::vector<std::shared_ptr<ad::MatrixXf>> &&_parameters) {
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
            p->descent();
    }

protected:
    std::vector<std::shared_ptr<ad::MatrixXf>> parameters;

};

class SGD: public Optimizer {
public:
    float beta = .9f;

    void add_parameters (std::vector<std::shared_ptr<ad::MatrixXf>> &&_parameters) override {
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

        beta_acc *= beta;
        steps += 1;

        for (int ii = 0; ii < parameters.size(); ++ii) {
            auto &m = momentum[ii];
            auto &g = parameters[ii]->grad;
            m = beta * m + (1.f-beta) * g;
            g = m / (1.f-beta_acc);
            parameters[ii]->descent();
        }
    }

private:
    std::vector<Eigen::MatrixXf> momentum;
    float beta_acc = 1.f;
    unsigned int steps = 1;

};

class Adam: public Optimizer {
public:
    float beta1 = .9f;
    float beta2 = .999f;
    float eps = 1e-8f;

    void add_parameters (std::vector<std::shared_ptr<ad::MatrixXf>> &&_parameters) override {
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
        for (auto &m: momentum1)
            m.setZero();
        for (auto &m: momentum2)
            m.setZero();
        beta1_acc = 1.f;
        beta2_acc = 1.f;
    }

    void descent() override {
        if (steps == std::numeric_limits<unsigned int>::max())
            return;

        beta1_acc *= beta1;
        beta2_acc *= beta2;
        steps += 1;

        for (int ii = 0; ii < parameters.size(); ++ii) {
            auto &m1 = momentum1[ii];
            auto &m2 = momentum2[ii];
            auto &g = parameters[ii]->grad;
            m1 = beta1 * m1 + (1.f-beta1) * g;
            m2 = beta2 * m2 + (1.f-beta2) * g.cwiseProduct(g);

            g = m1 / (1.f-beta1_acc);
            g = g.cwiseProduct(((m2.array()/(1.f-beta2_acc)).sqrt()+eps).inverse().matrix());
            parameters[ii]->descent();
        }
    }

private:
    std::vector<Eigen::MatrixXf> momentum1;
    std::vector<Eigen::MatrixXf> momentum2;
    float beta1_acc = 1.f;
    float beta2_acc = 1.f;
    unsigned int steps = 1;
};

} // eignn