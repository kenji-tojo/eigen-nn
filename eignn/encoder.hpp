#pragma once

#include <cmath>
#if !defined(M_PI)
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

#include <iostream>

#include <vector>

#include "module.hpp"
#include "hash.hpp"


namespace eignn::module {

class FourierFeature: public Module {
public:
    std::vector<float> freq;
    Eigen::MatrixXf partial_x;

    ~FourierFeature() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        m_dim = x.rows();
        const int freqs = freq.size();

        m_y.resize(m_dim*(1+2*freqs), x.cols());
        m_y.block(0,0,m_dim,x.cols()) = x;

        if (freqs == 0)
            return;

        partial_x.resize(m_dim*2*freqs, x.cols());

        for (int ii = 0; ii < x.cols(); ++ii) {
            for (int jj = 0; jj < m_dim; ++jj) {
                for (int kk = 0; kk < freqs; ++kk) {
                    int enc_id = 2*freqs*jj+2*kk;
                    auto pi2 = 2.f*float(M_PI);
                    m_y(m_dim+enc_id+0, ii) = std::cos(pi2*freq[kk]*x(jj,ii));
                    m_y(m_dim+enc_id+1, ii) = std::sin(pi2*freq[kk]*x(jj,ii));
                    partial_x(enc_id+0, ii) = -pi2*freq[kk]*std::sin(pi2*freq[kk]*x(jj,ii));
                    partial_x(enc_id+1, ii) = pi2*freq[kk]*std::cos(pi2*freq[kk]*x(jj,ii));
                }
            }
        }
    }

    void reverse(const Eigen::MatrixXf &y_bar) override {
        m_x_bar = y_bar.block(0,0,m_dim,y_bar.cols());
        const int freqs = freq.size();
        if (freqs == 0)
            return;

        for (int ii = 0; ii < y_bar.cols(); ++ii) {
            for (int jj = 0; jj < m_dim; ++jj) {
                for (int kk = 0; kk < freqs; ++kk) {
                    int enc_id = 2*freqs*jj+2*kk;
                    m_x_bar(jj,ii) -= y_bar(m_dim+enc_id+0,ii) * partial_x(enc_id+0,ii);
                    m_x_bar(jj,ii) -= y_bar(m_dim+enc_id+1,ii) * partial_x(enc_id+1,ii);
                }
            }
        }
    }

private:
    int m_dim = 0;
};


template<int ndim_ = 2>
class FeatureGrid: public Module {
public:
    const int table_size;
    const Eigen::ArrayXi shape;

    Eigen::MatrixXf feature;

    explicit FeatureGrid(Eigen::ArrayXi _shape, int dim, int _table_size)
            : shape(std::move(_shape))
            , table_size(_table_size) {
        static_assert(ndim_ == 2);
        feature.resize(dim, table_size);
        GaussSampler<float> gs{0.f,.5f};
        for (int ii = 0; ii < feature.size(); ++ii)
            feature.data()[ii] = gs.sample();
    }

    [[nodiscard]] int dim() const { return feature.rows(); }

    void forward(const Eigen::MatrixXf &x) override {
        static_assert(ndim_ == 2);
        forward_2d(x);
    }

    void reverse(const Eigen::MatrixXf &y_bar) override {
        static_assert(ndim_ == 2);
        reverse_2d(y_bar);
    }

private:

    Eigen::ArrayXf x0, x1;
    Eigen::ArrayXf c0, c1;

    void forward_2d(const Eigen::MatrixXf &x) {
        using namespace Eigen;

        const unsigned int batch_size = x.cols();
        m_y.resize(dim()+x.rows(), batch_size);
        m_y.block(dim(),0,x.rows(),batch_size) = x;

        x0 = x.row(0).array();
        x1 = x.row(1).array();
        x0 *= float(shape[0]);
        x1 *= float(shape[1]);
        c0 = x0.floor().max(0).min(shape[0]-1);
        c1 = x1.floor().max(0).min(shape[1]-1);
        x0 -= c0;
        x1 -= c1;

        for (int ii = 0; ii < batch_size; ++ii) {
            auto iw = int(c0[ii]);
            auto ih = int(c1[ii]);

            const VectorXf u00 = feature.col(hash::enc_2d(iw,ih,table_size));
            const VectorXf u01 = feature.col(hash::enc_2d(iw,ih+1,table_size));
            const VectorXf u10 = feature.col(hash::enc_2d(iw+1,ih,table_size));
            const VectorXf u11 = feature.col(hash::enc_2d(iw+1,ih+1,table_size));

            m_y.col(ii).block(0,0,dim(),1) = (1.f-x0[ii]) * (1.f-x1[ii]) * u00
                                             + (1.f-x0[ii]) * x1[ii] * u01
                                             + x0[ii] * (1.f-x1[ii]) * u10
                                             + x0[ii] * x1[ii] * u11;
        }
    }

    void reverse_2d(const Eigen::MatrixXf &y_bar) {
        using namespace Eigen;

        const unsigned int batch_size = y_bar.cols();
        assert(x0.size() == batch_size);
        m_x_bar.setZero(ndim_, batch_size); // gradient could be computed

        for (int ii = 0; ii < batch_size; ++ii) {
            auto iw = int(c0[ii]);
            auto ih = int(c1[ii]);

            const VectorXf u = y_bar.col(ii).block(0,0,dim(),1);

            feature.col(hash::enc_2d(iw,ih,table_size)) -= (1.f-x0[ii]) * (1.f-x1[ii]) * u;
            feature.col(hash::enc_2d(iw,ih+1,table_size)) -= (1.f-x0[ii]) * x1[ii] * u;
            feature.col(hash::enc_2d(iw+1,ih,table_size)) -= x0[ii] * (1.f-x1[ii]) * u;
            feature.col(hash::enc_2d(iw+1,ih+1,table_size)) -= x0[ii] * x1[ii] * u;
        }
    }

    void forward_3d(const Eigen::MatrixXf &x) {}
    void reverse_3d(const Eigen::MatrixXf &y_bar) {}

};

} // namespace eignn::module