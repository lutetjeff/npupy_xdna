#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// INT16_TILE_SIZE must match LINE_SIZE in col_independent.py.
// Value 512 = min(SUPPORTED_SHAPES["col_indep"]) / (8 cols * 4 cores).
static constexpr int INT16_TILE_SIZE = 512;
static constexpr int VEC_FACTOR = 32;

static void relu_vec(int16_t *restrict a, int16_t *restrict c,
                     const int32_t size) {
  ::aie::vector<int16_t, 32> zeroes = ::aie::broadcast<int16_t, 32>(0);
  int16_t *__restrict pA = a;
  int16_t *__restrict pC = c;
#pragma clang loop min_iteration_count(1)
  for (int i = 0; i < size; i += VEC_FACTOR) {
    ::aie::vector<int16_t, 32> a_v = ::aie::load_v<32>(pA);
    pA += VEC_FACTOR;
    ::aie::vector<int16_t, 32> c_v = ::aie::max(a_v, zeroes);
    ::aie::store_v(pC, c_v);
    pC += VEC_FACTOR;
  }
}

static void eltwise_add_vec(int16_t *restrict a, int16_t *restrict b,
                            int16_t *restrict c, const int32_t size) {
  int16_t *__restrict pA = a;
  int16_t *__restrict pB = b;
  int16_t *__restrict pC = c;
#pragma clang loop min_iteration_count(1)
  for (int i = 0; i < size; i += VEC_FACTOR) {
    ::aie::vector<int16_t, 32> a_v = ::aie::load_v<32>(pA);
    pA += VEC_FACTOR;
    ::aie::vector<int16_t, 32> b_v = ::aie::load_v<32>(pB);
    pB += VEC_FACTOR;
    ::aie::vector<int16_t, 32> c_v = ::aie::add(a_v, b_v);
    ::aie::store_v(pC, c_v);
    pC += VEC_FACTOR;
  }
}

extern "C" {

void relu_int16(int16_t *restrict a_in, int16_t *restrict c_out,
                int32_t size) {
  relu_vec(a_in, c_out, size);
}

void int16_relu(int16_t *a_in, int16_t *c_out) {
  relu_vec(a_in, c_out, INT16_TILE_SIZE);
}

void int16_eltwise_add(int16_t *a_in, int16_t *b_in, int16_t *c_out) {
  eltwise_add_vec(a_in, b_in, c_out, INT16_TILE_SIZE);
}

}
