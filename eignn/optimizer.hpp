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
            momentum[idx].setZero(p->m.rows(), p->m.cols());
            idx += 1;
        }
    }

    void reset() {
        for (auto &m: momentum)
            m.setZero();
        steps = 1;
    }

    void descent() override {
        if (steps == std::numeric_limits<unsigned int>::max())
            return;

        for (int ii = 0; ii < parameters.size(); ++ii) {
            auto &m = momentum[ii];
            auto &g = parameters[ii]->grad;
            m = beta * m + (1.f-beta) * g;
            g = m / (1.f - std::pow(beta, steps));
            parameters[ii]->descent();
        }
        steps += 1;
    }

private:
    std::vector<Eigen::MatrixXf> momentum;
    unsigned int steps = 1;

};

} // eignn