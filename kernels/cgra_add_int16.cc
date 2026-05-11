#define NOCPP
#include <stdint.h>

extern "C" {

void cgra_add(int16_t *restrict a, int16_t *restrict b, int16_t *restrict out) {
    for (int i = 0; i < 256; i++) {
        out[i] = a[i] + b[i];
    }
}

}
