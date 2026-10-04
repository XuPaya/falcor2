// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// The primal image and the interior term (derivatives of path contributions in material form).

/// Camera path of pixel sample i (the same paths in all kernels); `antithetic` and `dual` as in trace_path().
void trace_camera_path(uint2 pixel, uint i, uint spp, bool antithetic, bool dual, out PathRecord rec)
{
    UniformSampleGenerator sg = UniformSampleGenerator(pixel, seed * spp + i);
    trace_path(camera.position, camera_dir(float2(pixel) + next_float2(sg)), sg, rec, antithetic, dual);
}

// The primal image, the interior and the pixel boundary term are separate kernels: the derivative code
// of the interior term makes its kernel much heavier than the others. With antithetic sampling, a camera path whose
// first vertex samples a glossy reflection is averaged with its antithetic partner (see trace_path()), in both
// kernels: the derivative is the derivative of the primal estimate.

public void render_primal(uint2 pixel, uint spp, bool antithetic)
{
    float3 value = 0;
    for (uint i = 0; i < spp; i++) {
        [loop]
        for (uint branch = 0; branch < 2; branch++) {
            PathRecord rec;
            trace_camera_path(pixel, i, spp, antithetic, branch == 1, rec);
            value += path_value(rec) * (rec.antithetic ? 0.5 : 1.0);
            if (!rec.antithetic)
                break;
        }
    }
    primal[pixel.y * camera.dims.x + pixel.x] = value / spp;
}

public void render_interior(uint2 pixel, uint spp, bool antithetic)
{
    uint index = pixel.y * camera.dims.x + pixel.x;
    float3 adjoint = backward ? dl_di[index] / spp : float3(0);
    float3 derivative = 0;
    for (uint i = 0; i < spp; i++) {
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
        interior[index] = derivative / spp;
}
