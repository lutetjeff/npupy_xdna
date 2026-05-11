#define NOCPP
#include <stdint.h>

extern "C" {

void cgra_mul(int16_t *restrict a, int16_t *restrict b, int16_t *restrict out) {
    for (int i = 0; i < 256; i++) {
        out[i] = (int16_t)(a[i] * b[i]);
    }
}

}
