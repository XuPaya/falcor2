// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Primary edge term: silhouette edges seen from the camera (PSDR-Enzyme's boundary_primary).

void add_primary(uint index, float3 value)
{
    [ForceUnroll]
    for (uint c = 0; c < 3; c++)
        atomic_add(primary_boundary, 12 * index + 4 * c, value[c]);
}

public void render_primary(uint tid, uint samples)
{
    if (primary_edge_count == 0)
        return;
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid, 0x7fffffff), seed);
    float t;
    Edge edge = primary_edges[sample_edge(primary_edges, primary_edge_count, next_float(sg), t)];
    EdgeRef e = edge_ref(edge);
    float3 p[3];
    load_edge(0, e, p);
    float3 xb = lerp(p[0], p[1], t);
    float3 d = normalize(xb - camera.position);
    float z = dot(d, camera_forward());
    if (z <= 0)
        return;
    // Edge-plane normal, pointing away from the edge's faces (for a silhouette edge both lie on the other side).
    float3 nb = normalize(cross(p[0] - p[1], d));
    nb *= -sign(dot(nb, p[2] - p[0]));
    // The rays below run along the edge's own faces at grazing angles. Float32 rounding can put xb just inside the
    // occluder, and a hit on those faces then drops the sample (see docs/psdr-primary-edge-grazing-bias.md), so
    // cast them from a point just off the occluder.
    float3 xo = xb + nb * max(ray_epsilon, 1e-5 * max(length(xb), length(camera.position)));
    if (!visible(camera.position, xo))
        return;
    float2 film = camera.project_world_to_pixel(xb);
    if (any(film < 0) || any(film >= float2(camera.dims)))
        return;
    uint index = uint(film.y) * camera.dims.x + uint(film.x);
    // The surface behind the edge, or the environment it hides.
    PathRecord rec;
    trace_path(xo, d, NO_SLOT, sg, rec);
    float3 value = rec.radiance;
    if (all(value == 0))
        return;
    Background b = background(rec, xb, d);
    float weight = sensor_importance(b.p) * boundary_factor(b.p, xb, normalize(p[0] - p[1]), b.n)
        * primary_edge_length / samples;
    float3 adjoint = backward ? dl_di[index] : float3(0);
    float velocity = differentiate_velocity(b, xb, nb, true, e, t, dot(adjoint, value) * weight);
    if (!backward && all(isfinite(value * velocity * weight)))
        add_primary(index, value * velocity * weight);
}
