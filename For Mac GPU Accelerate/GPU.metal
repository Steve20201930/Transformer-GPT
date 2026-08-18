//
//  GPU.metal
//  TSFM_Sweden
//
//  Created by Steve William on 8/13/26.
//

#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(device const float* A      [[buffer(0)]],
                          device const float* B      [[buffer(1)]],
                          device float*       C      [[buffer(2)]],
                          constant uint&      M      [[buffer(3)]],
                          constant uint&      K      [[buffer(4)]],
                          constant uint&      N      [[buffer(5)]],
                          constant uint&      tA     [[buffer(6)]],
                          constant uint&      tB     [[buffer(7)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        float a = tA ? A[k * M + gid.y] : A[gid.y * K + k];
        float b = tB ? B[gid.x * K + k] : B[k * N + gid.x];
        sum += a * b;
    }
    C[gid.y * N + gid.x] = sum;
}
