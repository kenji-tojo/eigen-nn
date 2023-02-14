#pragma once

#include <cmath>
#if !defined(M_PI)
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

#include <vector>

#include <Eigen/Dense>


namespace eignn::encoder {
namespace {

template<typename Scalar_>
void fourier_feature(
        const Eigen::MatrixXf &x,
        Eigen::MatrixXf &x_enc,
        const std::vector<Scalar_> &freq
) {
    const auto dim = x.rows();
    const auto L = freq.size();

    x_enc.resize(dim*(1+2*L), x.cols());
    x_enc.block(0,0,dim,x.cols()) = x;

    for (int ii = 0; ii < x.cols(); ++ii) {
        for (int jj = 0; jj < dim; ++jj) {
            for (int kk = 0; kk < L; ++kk) {
                auto pi2 = 2.f*float(M_PI);
                auto f = float(freq[kk]);
                x_enc(dim+2*L*jj+2*kk+0, ii) = std::cos(pi2*f*x(jj,ii));
                x_enc(dim+2*L*jj+2*kk+1, ii) = std::sin(pi2*f*x(jj,ii));
            }
        }
    }

}

} // namespace
} // namespace eignn::encoder