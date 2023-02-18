#pragma once

#include <cstdint>


namespace eignn::hash {
namespace {

constexpr uint32_t p1 = 1;
constexpr uint32_t p2 = 2654435761;
constexpr uint32_t p3 = 805459861;

inline uint32_t enc_2d(uint32_t i1, uint32_t i2, uint32_t table_size) {
    return ((p1*i1) ^ (p2*i2)) % table_size;
}

inline uint32_t enc_3d(uint32_t i1, uint32_t i2, uint32_t i3, uint32_t table_size) {
    return ((p1*i1) ^ (p2*i2) ^ (p3*i3)) % table_size;
}

} // namespace
} // namespace eignn::hash