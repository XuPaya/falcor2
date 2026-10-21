// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// The primal image and the interior term (derivatives of path contributions in material form).

/// Frames per seed: camera sample i of a pass with seed s is frame s * FRAMES_PER_SEED + i, as the reference path
/// tracer's `iteration` (falcor2 toolkits render frame seed * 65536 + i).
static const uint FRAMES_PER_SEED = 65536;

/// Running compensated sums of the primal image (the AccumulatorNode's single_compensated history).
public RWStructuredBuffer<float3> primal_sum;
public RWStructuredBuffer<float3> primal_residual;

/// Camera path of pixel sample i (the same paths in all kernels: the reference path tracer's path of frame
/// seed * FRAMES_PER_SEED + i); `antithetic` and `dual` as in trace_path().
void trace_camera_path(uint2 pixel, uint i, uint spp, bool antithetic, bool dual, out PathRecord rec)
{
    UniformSampleGenerator sg = UniformSampleGenerator(pixel, seed * FRAMES_PER_SEED + i);
    trace_path(primary_ray(pixel, sg), sg, rec, antithetic, dual);
}

float neumaier_correction(float sum, float value, float next)
{
    float correction;
    if (abs(sum) >= abs(value))
        correction = fma(-1.f, next, sum) + value;
    else
        correction = fma(-1.f, next, value) + sum;
    return correction;
}

// The primal image, the interior and the pixel boundary term are separate kernels: the derivative code
// of the interior term makes its kernel much heavier than the others. With antithetic sampling, a camera path whose
// first vertex samples a glossy reflection is averaged with its antithetic partner (see trace_path()) in the interior
// term; the primal image is the plain estimate.

/// Frame `frame` (sample `sample_index` of the pass) of the primal image: the camera path's radiance plus the frame's
/// light splats (light_primal), accumulated as falcor2's AccumulatorNode accumulates the reference / adjoint path
/// tracer's frames (non-finite samples dropped, compensated sum); primal holds the running mean.
public void render_primal(uint2 pixel, uint frame, uint sample_index)
{
    uint index = pixel.y * camera.dims.x + pixel.x;
    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frame);
    PathRecord rec;
    trace_path(primary_ray(pixel, sg), sg, rec);
    float3 value = rec.radiance;
    if (light_tracing)
        value = value + asfloat(light_primal.Load3(index * 12));
    value = !all(isfinite(value)) ? float3(0.f) : value;
    if (sample_index == 0) {
        primal_sum[index] = value;
        primal_residual[index] = float3(0.f);
        primal[index] = value;
        return;
    }
    float3 sum = primal_sum[index];
    float3 next = sum + value;
    float3 residual = primal_residual[index] + float3(neumaier_correction(sum.x, value.x, next.x),
                                                      neumaier_correction(sum.y, value.y, next.y),
                                                      neumaier_correction(sum.z, value.z, next.z));
    primal_sum[index] = next;
    primal_residual[index] = residual;
    float inverse_count = 1.f / float(sample_index + 1);
    primal[index] = (next + residual) * inverse_count;
}

/// Samples first, ..., first + count - 1 of spp (one dispatch per chunk of samples: long dispatches time out).
public void render_interior(uint2 pixel, uint spp, uint first, uint count, bool antithetic)
{
    uint index = pixel.y * camera.dims.x + pixel.x;
    float3 adjoint = backward ? dl_di[index] / spp : float3(0);
    float3 derivative = 0;
    for (uint i = first; i < first + count; i++) {
        [loop]
        for (uint branch = 0; branch < 2; branch++) {
            PathRecord rec;
            trace_camera_path(pixel, i, spp, antithetic, branch == 1, rec);
            float weight = rec.antithetic ? 0.5 : 1.0;
            float3 d;
            differentiate_path(rec, true, adjoint * weight, d);
            // Non-finite derivatives of degenerate samples are dropped, as backward mode drops their gradients.
            if (all(isfinite(d)))
                derivative += d * weight;
            if (!rec.antithetic)
                break;
        }
    }
    if (!backward)
        interior[index] += derivative / spp;
}
