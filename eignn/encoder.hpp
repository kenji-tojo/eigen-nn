#pragma once

#include <cmath>
#if !defined(M_PI)
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

#include <vector>

#include "module.hpp"


namespace eignn::module {

class FourierFeature: public Module {
public:
    std::vector<float> freq;

    ~FourierFeature() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        m_dim = x.rows();
        const int freqs = freq.size();

        m_y.resize(m_dim*(1+2*freqs), x.cols());
        m_y.block(0,0,m_dim,x.cols()) = x;

        for (int ii = 0; ii < x.cols(); ++ii) {
            for (int jj = 0; jj < m_dim; ++jj) {
                for (int kk = 0; kk < freqs; ++kk) {
                    auto pi2 = 2.f*float(M_PI);
                    m_y(m_dim+2*freqs*jj+2*kk+0, ii) = std::cos(pi2*freq[kk]*x(jj,ii));
                    m_y(m_dim+2*freqs*jj+2*kk+1, ii) = std::sin(pi2*freq[kk]*x(jj,ii));
                }
            }
        }
    }

    void reverse(const Eigen::MatrixXf &y_bar) override {
        m_x_bar.resize(m_dim, y_bar.cols());
    }

private:
    int m_dim = 0;
};

} // namespace eignn::module