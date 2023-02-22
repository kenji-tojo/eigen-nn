#pragma once

#include <cmath>
#if !defined(M_PI)
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

#include <iostream>

#include <vector>

#include "module.hpp"
#include "hash.hpp"
#include "ad.hpp"


namespace eignn::module {

template<unsigned int ndim_ = 2>
class FourierFeature: public Module {
public:
    unsigned int freqs = 0;
    Eigen::MatrixXf d_x;

    ~FourierFeature() override = default;

    [[nodiscard]] unsigned int out_dim() const { return ndim_*(1+2*freqs); }

    void forward(const Eigen::MatrixXf &x) override {
        assert(ndim_ == x.rows());

        y.resize(out_dim(), x.cols());
        y.block(0,0,ndim_,x.cols()) = x;

        if (freqs == 0)
            return;

        d_x.resize(ndim_*2*freqs, x.cols());

        for (int ii = 0; ii < x.cols(); ++ii) {
            for (int jj = 0; jj < ndim_; ++jj) {
                float F = 2.f * float(M_PI);

                for (int kk = 0; kk < freqs; ++kk) {
                    unsigned int index = 2*freqs*jj+2*kk;
                    float cos_f = std::cos(F * x(jj,ii));
                    float sin_f = std::sin(F * x(jj,ii));

                    y(ndim_+index+0, ii) = cos_f;
                    y(ndim_+index+1, ii) = sin_f;
                    d_x(index+0, ii) = -F * sin_f;
                    d_x(index+1, ii) = F * cos_f;

                    F *= 2.f;
                }
            }
        }
    }

    void adjoint(const Eigen::MatrixXf &y_adj) override {
        x_adj.resize(ndim_,y_adj.cols());
        x_adj = y_adj.block(0,0,ndim_,y_adj.cols());

        if (freqs == 0)
            return;

        for (int ii = 0; ii < y_adj.cols(); ++ii) {
            for (int jj = 0; jj < ndim_; ++jj) {
                for (int kk = 0; kk < freqs; ++kk) {
                    unsigned int index = 2*freqs*jj+2*kk;
                    x_adj(jj,ii) -= y_adj(ndim_+index+0,ii) * d_x(index+0,ii);
                    x_adj(jj,ii) -= y_adj(ndim_+index+1,ii) * d_x(index+1,ii);
                }
            }
        }
    }

};


template<unsigned int ndim_ = 2>
class FeatureGrid: public Module {
public:
    std::vector<std::shared_ptr<ad::Matrixf>> feature;
    float b = 1.5f;

    explicit FeatureGrid(
            int min_res,
            int _dim,
            int _levels,
            int table_size_log2
    ): dim(_dim), levels(_levels) {
        static_assert(ndim_ == 2); // TODO: support ndim_ == 3

        if (levels <= 0) return;

        table_size.resize(levels);
        feature.resize(levels);
        res.resize(levels);
        res[0] = min_res;

        for (int ii = 0; ii < levels; ++ii) {
            if (ii > 0)
                res[ii] = std::floor(res[ii-1]*b);

            auto &T = table_size[ii];
            T = 1 << table_size_log2;
            int elems = (res[ii]+1)*(res[ii]+1);
            while (T > elems*2)
                T >>= 1; // shrink to fit

            std::cout << "level = " << ii << "; "
                      << "res = " << res[ii] << "; "
                      << "table_size = " << T
                      << std::endl;

            auto &ft = feature[ii];
            ft = std::make_shared<ad::Matrixf>();
            ft->resize(dim, table_size[ii]);

            GaussSampler<float> gs{0.f,.5f};
            ft->init([&gs](int index){ return gs.sample(); });
        }
    }

    [[nodiscard]] unsigned int out_dim() const { return dim * levels; }

    void forward(const Eigen::MatrixXf &x) override {
        static_assert(ndim_ == 2);
        forward_2d(x);
    }

    void adjoint(const Eigen::MatrixXf &y_bar) override {
        static_assert(ndim_ == 2);
        adjoint_2d(y_bar);
    }

    std::vector<std::shared_ptr<ad::Matrixf>> parameters() override {
        return feature;
    }

private:
    unsigned int dim = 0;
    unsigned int levels = 0;

    std::vector<uint32_t> table_size;
    std::vector<unsigned int> res;

    std::vector<Eigen::ArrayXf> x0_vec, x1_vec;
    std::vector<Eigen::ArrayXf> c0_vec, c1_vec;

    void forward_2d(const Eigen::MatrixXf &x) {
        using namespace Eigen;

        assert(x.rows() == ndim_);

        y.resize(dim*levels, x.cols());

        if (levels <= 0)
            return;

        x0_vec.resize(levels);
        x1_vec.resize(levels);
        c0_vec.resize(levels);
        c1_vec.resize(levels);

        for (int level_id = 0; level_id < levels; ++level_id) {
            auto &x0 = x0_vec[level_id];
            auto &x1 = x1_vec[level_id];
            auto &c0 = c0_vec[level_id];
            auto &c1 = c1_vec[level_id];

            x0 = x.row(0).array();
            x1 = x.row(1).array();
            x0 *= float(res[level_id]);
            x1 *= float(res[level_id]);
            c0 = x0.floor().max(0).min(res[level_id]-1);
            c1 = x1.floor().max(0).min(res[level_id]-1);
            x0 -= c0;
            x1 -= c1;

            assert((x0 > 0.f-1e-12f).all());
            assert((x0 < 1.f+1e-12f).all());

            for (int ii = 0; ii < x.cols(); ++ii) {
                const auto &ft = feature[level_id]->m;
                const auto T = table_size[level_id];

                const auto iw = int(c0[ii]);
                const auto ih = int(c1[ii]);

                const VectorXf u00 = ft.col(hash::enc_2d(iw,ih,T));
                const VectorXf u01 = ft.col(hash::enc_2d(iw,ih+1,T));
                const VectorXf u10 = ft.col(hash::enc_2d(iw+1,ih,T));
                const VectorXf u11 = ft.col(hash::enc_2d(iw+1,ih+1,T));

                y.block(dim*level_id,ii,dim,1) = (1.f-x0[ii]) * (1.f-x1[ii]) * u00
                                                 + (1.f-x0[ii]) * x1[ii] * u01
                                                 + x0[ii] * (1.f-x1[ii]) * u10
                                                 + x0[ii] * x1[ii] * u11;
            }
        }
    }

    void adjoint_2d(const Eigen::MatrixXf &y_adj) {
        using namespace Eigen;

        // gradient could be back-propagated...
        x_adj = MatrixXf::Zero(ndim_, y_adj.cols());

        if (levels == 0) return;

        for (auto &ft: feature)
            ft->grad.setZero();

        for (int level_id = 0; level_id < levels; ++level_id) {
            const auto &x0 = x0_vec[level_id];
            const auto &x1 = x1_vec[level_id];
            const auto &c0 = c0_vec[level_id];
            const auto &c1 = c1_vec[level_id];

            assert(x0.size() == y_adj.cols());
            assert(x1.size() == y_adj.cols());
            assert(c0.size() == y_adj.cols());
            assert(c1.size() == y_adj.cols());

            for (int ii = 0; ii < y_adj.cols(); ++ii) {
                auto iw = int(c0[ii]);
                auto ih = int(c1[ii]);

                const VectorXf u = y_adj.block(dim*level_id,ii,dim,1);

                auto &ft_grad = feature[level_id]->grad;
                const auto T = table_size[level_id];

                ft_grad.col(hash::enc_2d(iw,ih,T)) += (1.f-x0[ii]) * (1.f-x1[ii]) * u;
                ft_grad.col(hash::enc_2d(iw,ih+1,T)) += (1.f-x0[ii]) * x1[ii] * u;
                ft_grad.col(hash::enc_2d(iw+1,ih,T)) += x0[ii] * (1.f-x1[ii]) * u;
                ft_grad.col(hash::enc_2d(iw+1,ih+1,T)) += x0[ii] * x1[ii] * u;
            }
        }
    }

    void forward_3d(const Eigen::MatrixXf &x) { /* TODO */ }
    void adjoint_3d(const Eigen::MatrixXf &y_bar) { /* TODO */ }

};

} // namespace eignn::module