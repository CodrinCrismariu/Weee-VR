// Image sampling for the MediaPipe hand models, kept in this -O3 C++ target: the Swift app target is
// built Debug by xtool, and this loop touches 65k pixels per model run.
#include "XRSLAMCore.h"

#include <algorithm>
#include <cmath>

extern "C" void weee_crop_rgb(const uint8_t *luma, int luma_stride, const uint8_t *chroma, int chroma_stride,
                              int width, int height, const float m[6], int size, float *out) {
    const int plane = size * size;
    float *r = out, *g = out + plane, *b = out + 2 * plane;
    const float maxX = float(width - 1), maxY = float(height - 1);
    for (int v = 0; v < size; ++v) {
        float x = m[1] * v + m[2];
        float y = m[4] * v + m[5];
        for (int u = 0; u < size; ++u, x += m[0], y += m[3]) {
            const int i = v * size + u;
            if (x < 0 || y < 0 || x > maxX || y > maxY) {
                r[i] = g[i] = b[i] = 0;
                continue;
            }
            // Bilinear luma, nearest chroma (it is half resolution anyway).
            const int x0 = std::min(int(x), width - 2), y0 = std::min(int(y), height - 2);
            const float fx = x - x0, fy = y - y0;
            const uint8_t *row0 = luma + y0 * luma_stride + x0, *row1 = row0 + luma_stride;
            const float lum = (row0[0] * (1 - fx) + row0[1] * fx) * (1 - fy) + (row1[0] * (1 - fx) + row1[1] * fx) * fy;
            const uint8_t *c = chroma + (int(y) >> 1) * chroma_stride + (int(x) >> 1) * 2;
            const float cb = float(c[0]) - 128, cr = float(c[1]) - 128;
            // Full-range BT.601, as the passthrough shader.
            r[i] = std::clamp((lum + 1.402f * cr) * (1.0f / 255), 0.0f, 1.0f);
            g[i] = std::clamp((lum - 0.344136f * cb - 0.714136f * cr) * (1.0f / 255), 0.0f, 1.0f);
            b[i] = std::clamp((lum + 1.772f * cb) * (1.0f / 255), 0.0f, 1.0f);
        }
    }
}
