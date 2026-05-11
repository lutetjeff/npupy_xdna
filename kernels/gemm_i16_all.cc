// Combined kernel for GemmFusionTemplate — all functions in one binary.
// Required compile flags:
//   -Di16_i16_ONLY -DDIM_M=64 -DDIM_K=64 -DDIM_N=64 -DB_COL_MAJ -DVECTORIZED_ONLY
// Optional compile flags (epilogue/prologue):
//   -DEPILOGUE_RELU
//   -DEPILOGUE_BIAS_ADD [-DBIAS_VAL=N]
//   -DPROLOGUE_SCALE   [-DALPHA_VAL=N]
// Exported: zero_i16, matmul_i16_i16, relu_i16_tile*, add_bias_i16_tile*, scale_i16_tile*

#define NOCPP
#include "/home/lutet/mlir-aie/aie_kernels/aie2p/mm.cc"

extern "C" {

#ifdef EPILOGUE_RELU
void relu_i16_tile(int16_t* restrict c) {
    constexpr int r = 32;
    constexpr int tile_n = DIM_M * DIM_N;
    aie::vector<int16_t, r> zeros = aie::zeros<int16_t, r>();
    for (int i = 0; i < tile_n; i += r) {
        aie::vector<int16_t, r> v = aie::load_v<r>(c + i);
        aie::store_v(c + i, aie::max(v, zeros));
    }
}
#endif

#ifdef EPILOGUE_BIAS_ADD
#ifndef BIAS_VAL
#define BIAS_VAL 0
#endif
void add_bias_i16_tile(int16_t* restrict c) {
    constexpr int r = 32;
    constexpr int tile_n = DIM_M * DIM_N;
    aie::vector<int16_t, r> bvec = aie::broadcast<int16_t, r>((int16_t)BIAS_VAL);
    for (int i = 0; i < tile_n; i += r) {
        aie::vector<int16_t, r> v = aie::load_v<r>(c + i);
        aie::store_v(c + i, aie::add(v, bvec));
    }
}
#endif

#ifdef PROLOGUE_SCALE
#ifndef ALPHA_VAL
#define ALPHA_VAL 1
#endif
void scale_i16_tile(int16_t* restrict c) {
    constexpr int tile_n = DIM_M * DIM_N;
    const int32_t factor = ALPHA_VAL;
    for (int i = 0; i < tile_n; i++) {
        int32_t tmp = (int32_t)c[i] * factor;
        if (tmp > 32767) tmp = 32767;
        if (tmp < -32768) tmp = -32768;
        c[i] = (int16_t)tmp;
    }
}
#endif

} // extern "C"
