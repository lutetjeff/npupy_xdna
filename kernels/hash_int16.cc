#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// INT16_TILE_SIZE must match LINE_SIZE in col_independent.py.
// Value 512 = min(SUPPORTED_SHAPES["col_indep"]) / (8 cols * 4 cores).
static constexpr int INT16_TILE_SIZE = 512;

static constexpr uint16_t FNV_OFFSET = (uint16_t)0x811Cu;
static constexpr uint16_t FNV_PRIME  = (uint16_t)0x0193u;

static void hash_fnv1a(int16_t *restrict a, int16_t *restrict c,
                       const int32_t size) {
    for (int i = 0; i < size; i++) {
        uint16_t hash = FNV_OFFSET;
        uint16_t x = (uint16_t)a[i];
        hash ^= x; hash = (uint16_t)(hash * FNV_PRIME); x = (uint16_t)(hash >> 1u);
        hash ^= x; hash = (uint16_t)(hash * FNV_PRIME); x = (uint16_t)(hash >> 1u);
        hash ^= x; hash = (uint16_t)(hash * FNV_PRIME); x = (uint16_t)(hash >> 1u);
        hash ^= x; hash = (uint16_t)(hash * FNV_PRIME); x = (uint16_t)(hash >> 1u);
        hash ^= x; hash = (uint16_t)(hash * FNV_PRIME); x = (uint16_t)(hash >> 1u);
        hash ^= x; hash = (uint16_t)(hash * FNV_PRIME); x = (uint16_t)(hash >> 1u);
        hash ^= x; hash = (uint16_t)(hash * FNV_PRIME); x = (uint16_t)(hash >> 1u);
        hash ^= x; hash = (uint16_t)(hash * FNV_PRIME); x = (uint16_t)(hash >> 1u);
        (void)x;
        c[i] = (int16_t)hash;
    }
}

extern "C" {

void int16_hash(int16_t *a_in, int16_t *c_out) {
    hash_fnv1a(a_in, c_out, INT16_TILE_SIZE);
}

}
