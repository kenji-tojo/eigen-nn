#pragma once

#include <random>


namespace eignn {

template<typename Float_>
class Sampler {
public:
    Sampler(): random_device(), generator(random_device()), distribution(0,1) {
        static_assert(std::is_floating_point_v<Float_>);
    }

    Float_ sample() { return distribution(generator); }

private:
    std::random_device random_device;
    std::mt19937 generator;
    std::uniform_real_distribution<Float_> distribution;
};

template<typename Float_>
class GaussSampler {
public:
    explicit GaussSampler(Float_ mean = 0, Float_ stddev = 1)
    : random_device()
    , generator(random_device())
    , distribution(mean, stddev) {
        static_assert(std::is_floating_point_v<Float_>);
    }

    Float_ sample() { return distribution(generator); }

private:
    std::random_device random_device;
    std::mt19937 generator;
    std::normal_distribution<Float_> distribution{0,.5f};
};

} // namespace eignn
