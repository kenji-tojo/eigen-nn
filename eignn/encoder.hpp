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

class FourierFeature: public Module {
public:
    std::vector<float> freq;
    Eigen::MatrixXf d_x;

    ~FourierFeature() override = default;

    void forward(const Eigen::MatrixXf &x) override {
        m_dim = x.rows();
        const int freqs = freq.size();

        y.resize(m_dim*(1+2*freqs), x.cols());
        y.block(0,0,m_dim,x.cols()) = x;

        if (freqs == 0)
            return;

        d_x.resize(m_dim*2*freqs, x.cols());

        for (int ii = 0; ii < x.cols(); ++ii) {
            for (int jj = 0; jj < m_dim; ++jj) {
                for (int kk = 0; kk < freqs; ++kk) {
                    int enc_id = 2*freqs*jj+2*kk;
                    auto pi2 = 2.f*float(M_PI);
                    float cs = std::cos(pi2*freq[kk]*x(jj,ii));
                    float sn = std::sin(pi2*freq[kk]*x(jj,ii));
                    y(m_dim+enc_id+0, ii) = cs;
                    y(m_dim+enc_id+1, ii) = sn;
                    d_x(enc_id+0, ii) = -pi2*freq[kk]*sn;
                    d_x(enc_id+1, ii) = pi2*freq[kk]*cs;
                }
            }
        }
    }

    void adjoint(const Eigen::MatrixXf &y_adj) override {
        x_adj.resize(m_dim,y_adj.cols());
        x_adj = y_adj.block(0,0,m_dim,y_adj.cols());

        const int freqs = freq.size();

        if (freqs == 0)
            return;

        for (int ii = 0; ii < y_adj.cols(); ++ii) {
            for (int jj = 0; jj < m_dim; ++jj) {
                for (int kk = 0; kk < freqs; ++kk) {
                    int enc_id = 2*freqs*jj+2*kk;
                    x_adj(jj,ii) -= y_adj(m_dim+enc_id+0,ii) * d_x(enc_id+0,ii);
                    x_adj(jj,ii) -= y_adj(m_dim+enc_id+1,ii) * d_x(enc_id+1,ii);
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
    std::vector<std::shared_ptr<ad::MatrixXf>> feature;

    explicit FeatureGrid(
            int min_res,
            int levels,
            int dim,
            int table_size_log2
    ): m_dim(dim), m_levels(levels) {
        static_assert(ndim_ == 2); // TODO: support ndim_ == 3

        if (levels <= 0)
            return;

        table_size.resize(levels);
        feature.resize(levels);
        res.resize(levels);
        for (int ii = 0; ii < levels; ++ii) {
            res[ii] = std::floor(std::pow(1.5f, ii)*min_res);

            int elems = (res[ii]+1)*(res[ii]+1);
            auto &T = table_size[ii];
            T = 1 << table_size_log2;
            while (T > elems*2)
                T >>= 1;

            std::cout << "level = " << ii << "; "
                      << "res = " << res[ii] << "; "
                      << "table_size = " << T
                      << std::endl;

            auto &ft = feature[ii];
            ft = std::make_shared<ad::MatrixXf>();
            ft->resize(dim, table_size[ii]);
            GaussSampler<float> gs{0.f,.5f};
            ft->init([&gs](int index){ return gs.sample(); });
        }
    }

    [[nodiscard]] int dim() const { return m_dim; }
    [[nodiscard]] int levels() const { return m_levels; }

    void forward(const Eigen::MatrixXf &x) override {
        static_assert(ndim_ == 2);
        forward_2d(x);
    }

    void adjoint(const Eigen::MatrixXf &y_bar) override {
        static_assert(ndim_ == 2);
        adjoint_2d(y_bar);
    }

    std::vector<std::shared_ptr<ad::MatrixXf>> parameters() override {
        return feature;
    }

private:
    int m_dim = 0;
    int m_levels = 0;

    std::vector<uint32_t> table_size;
    std::vector<int> res;

    std::vector<Eigen::ArrayXf> x0_vec, x1_vec;
    std::vector<Eigen::ArrayXf> c0_vec, c1_vec;

    void forward_2d(const Eigen::MatrixXf &x) {
        using namespace Eigen;

        assert(x.rows() >= ndim_);

        y.resize(dim()*levels()+x.rows(), x.cols());
        y.block(dim()*levels(),0,x.rows(),x.cols()) = x;

        if (levels() <= 0)
            return;

        x0_vec.resize(levels());
        x1_vec.resize(levels());
        c0_vec.resize(levels());
        c1_vec.resize(levels());

        for (int level_id = 0; level_id < levels(); ++level_id) {
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
            assert((x0 > 0.f-1e-5f).all());
            assert((x0 < 1.f+1e-5f).all());

            for (int ii = 0; ii < x.cols(); ++ii) {
                const auto &ft = feature[level_id]->m;
                const auto T = table_size[level_id];

                const auto iw = int(c0[ii]);
                const auto ih = int(c1[ii]);

                const VectorXf u00 = ft.col(hash::enc_2d(iw,ih,T));
                const VectorXf u01 = ft.col(hash::enc_2d(iw,ih+1,T));
                const VectorXf u10 = ft.col(hash::enc_2d(iw+1,ih,T));
                const VectorXf u11 = ft.col(hash::enc_2d(iw+1,ih+1,T));

                y.block(dim()*level_id,ii,dim(),1) = (1.f-x0[ii]) * (1.f-x1[ii]) * u00
                                                     + (1.f-x0[ii]) * x1[ii] * u01
                                                     + x0[ii] * (1.f-x1[ii]) * u10
                                                     + x0[ii] * x1[ii] * u11;
            }
        }
    }

    void adjoint_2d(const Eigen::MatrixXf &y_adj) {
        using namespace Eigen;

        if (levels() <= 0) {
            x_adj = y_adj;
            return;
        }

        x_adj = y_adj.block(
                dim()*levels(),0,
                y_adj.rows()-dim()*levels(),y_adj.cols()
        );

        for (auto &ft: feature)
            ft->grad.setZero();

        for (int level_id = 0; level_id < levels(); ++level_id) {
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

                const VectorXf u = y_adj.block(dim()*level_id,ii,dim(),1);

                auto &ft_grad = feature[level_id]->grad;
                const auto T = table_size[level_id];

                ft_grad.col(hash::enc_2d(iw,ih,T)) += (1.f-x0[ii]) * (1.f-x1[ii]) * u;
                ft_grad.col(hash::enc_2d(iw,ih+1,T)) += (1.f-x0[ii]) * x1[ii] * u;
                ft_grad.col(hash::enc_2d(iw+1,ih,T)) += x0[ii] * (1.f-x1[ii]) * u;
                ft_grad.col(hash::enc_2d(iw+1,ih+1,T)) += x0[ii] * x1[ii] * u;
            }
        }
    }

    void forward_3d(const Eigen::MatrixXf &x) {}
    void adjoint_3d(const Eigen::MatrixXf &y_bar) {}

};

} // namespace eignn::module