#define NOCPP
#include <stdint.h>

extern "C" {

void horner_stage(int16_t *restrict in, int16_t *restrict out) {
    for (int i = 0; i < 256; i++) {
        int32_t tmp = (int32_t)in[i] * 3 + 7;
        if (tmp > 32767) tmp = 32767;
        if (tmp < -32768) tmp = -32768;
        out[i] = (int16_t)tmp;
    }
}

}
