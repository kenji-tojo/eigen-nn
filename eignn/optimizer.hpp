#pragma once

#include <memory>

#include "ad.hpp"


namespace eignn {

class Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<ad::MatrixXf>> &&_parameters)
            : parameters(std::move(_parameters)) {}

    virtual void descent() {
        for (auto &p: parameters)
            p->descent();
    }

private:
    std::vector<std::shared_ptr<ad::MatrixXf>> parameters;

};

} // eignn