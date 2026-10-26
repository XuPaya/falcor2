// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Pixel boundary term: visibility boundaries on the borders of the pixels (PSDR-Enzyme's
// boundary_pixel).

/// Antithetic positions on the pixel's perimeter (see pixel_sample()): u, its reflection through the pixel's center
/// and its mirror images across the pixel's vertical and horizontal center lines.
float antithetic_position(float u, uint a)
{
    return a == 0 ? u : a == 1 ? frac(u + 0.5) : a == 2 ? frac(1.75 - u) : frac(1.25 - u);
}

/// One pixel-boundary sample at position u on the pixel's perimeter, which runs from the left side (u = 0) to the
/// top, right and bottom sides.
float3 pixel_sample(uint2 pixel, float u, float3 adjoint, inout UniformSampleGenerator sg)
{
    float3 right = normalize(camera.image_u), up = normalize(camera.image_v);
    float w = length(camera.image_w);
    float2 side = float2(2 * length(camera.image_u) / (w * camera.dims.x), 2 * length(camera.image_v) / (w * camera.dims.y));
    float q = u * 4, length_ = side.y;
    float2 offset;
    // Sides of the pixel (film y grows downwards) and their inward directions.
    float3 edge_dir, visible_dir;
    if (q < 1) { offset = float2(0, q); edge_dir = up; visible_dir = right; }
    else if (q < 2) { offset = float2(q - 1, 1); edge_dir = right; visible_dir = up; length_ = side.x; }
    else if (q < 3) { offset = float2(1, 3 - q); edge_dir = up; visible_dir = -right; }
    else { offset = float2(4 - q, 0); edge_dir = right; visible_dir = -up; length_ = side.x; }
    float3 d = camera_dir(float2(pixel) + offset);
    PathRecord rec;
    // The environment does not move with the scene: no boundary term.
    if (!trace_path(camera.position, d, NO_SLOT, sg, rec))
        return 0;
    float3 value = rec.radiance;
    float3 xb = camera.position + d / dot(d, camera_forward());
    Background b = background(rec, xb, d);
    float3 nb = cross(edge_dir, d);
    nb *= sign(dot(nb, visible_dir));
    float weight = sensor_importance(b.p) * boundary_factor(b.p, xb, edge_dir, b.n) * 4 * length_;
    EdgeRef e = {};
    float velocity = differentiate_velocity(b, xb, nb, false, e, 0, dot(adjoint, value) * weight);
    return value * velocity * weight;
}

/// `antithetic` samples (1, 2 or 4) per perimeter position: see antithetic_position(). Samples first, ...,
/// first + count - 1 of spp.
public void render_pixel_boundary(uint2 pixel, uint spp, uint first, uint count, uint antithetic)
{
    uint index = pixel.y * camera.dims.x + pixel.x;
    float3 adjoint = backward ? dl_di[index] / spp : float3(0);
    float3 boundary = 0;
    for (uint i = first; i < first + count; i++) {
        UniformSampleGenerator sp = UniformSampleGenerator(pixel, 0x80000000 | (seed * spp + i));
        float u = next_float(sp);
        // Antithetic samples share their other random numbers. One call site keeps the kernel small.
        [loop]
        for (uint a = 0; a < antithetic; a++) {
            UniformSampleGenerator sa = sp;
            boundary += pixel_sample(pixel, antithetic_position(u, a), adjoint / antithetic, sa) / antithetic;
        }
    }
    if (!backward)
        pixel_boundary[index] += boundary / spp;
}
