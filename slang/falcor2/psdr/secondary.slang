// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Secondary edge term: silhouette edges seen from path vertices, by photon mapping (PSDR-Enzyme's
// boundary_direct_pm).

public struct Photon
{
    public float3 position;
    public float t;
    public float3 normal0;
    public uint edge;
    public float3 normal1;
    public int has_second_face;
};

public RWStructuredBuffer<Photon> photons;
public RWStructuredBuffer<float3> photon_bounds;
public RaytracingAccelerationStructure photon_tlas;
public float photon_radius;
public bool check_visibility;

/// Samples one edge photon per thread; photon_bounds holds its sphere AABB.
public void sample_photons(uint tid)
{
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid, 0x3fffffff), seed);
    float t;
    uint index = sample_edge(photon_edges, photon_edge_count, next_float(sg), t);
    EdgeRef e = edge_ref(photon_edges[index]);
    float3 p[3];
    load_edge(0, e, p);
    Photon photon = {};
    photon.position = lerp(p[0], p[1], t);
    photon.t = t;
    photon.edge = index;
    photon.normal0 = load_vertex(0, { e.gid, e.prim, float2(0) }).n;
    photon.has_second_face = e.prim1 >= 0;
    if (e.prim1 >= 0)
        photon.normal1 = load_vertex(0, { e.gid, uint(e.prim1), float2(0) }).n;
    photons[tid] = photon;
    photon_bounds[2 * tid] = photon.position - photon_radius;
    photon_bounds[2 * tid + 1] = photon.position + photon_radius;
}

[Differentiable]
float3 intersect_triangle(float3 origin, float3 dir, float3 p[3], no_diff bool material)
{
    float3 e1 = p[1] - p[0];
    float3 e2 = p[2] - p[0];
    float3 q = cross(dir, e2);
    float divisor = dot(q, e1);
    if (abs(divisor) < 1e-8f)
        divisor = divisor > 0 ? 1e-8f : -1e-8f;
    float3 s = origin - p[0];
    float u = dot(s, q) / divisor;
    float v = dot(dir, cross(s, e1)) / divisor;
    if (material)
        return (1 - detach(u) - detach(v)) * p[0] + detach(u) * p[1] + detach(v) * p[2];
    return (1 - u - v) * detach(p[0]) + u * detach(p[1]) + v * detach(p[2]);
}

/// Normal velocity of a secondary edge: source S, edge B (v0, v1, third vertex) and receiver D. Edge
/// points project onto the receiver from the source point behind them or, for an infinite source, along
/// its direction.
[Differentiable]
float normal_velocity(float3 s[3], float3 b[3], float3 d[3], no_diff float t, no_diff float3 dir, no_diff bool infinite)
{
    float3 xb = b[0] + (b[1] - b[0]) * t;
    float3 sb = -dir, projection_dir = -dir;
    if (!infinite) {
        float3 xs = intersect_triangle(xb, dir, s, true);
        if (length(xs - xb) < 1e-3f)
            return 0;
        float divisor = detach(dot(cross(dir, s[2] - s[0]), s[1] - s[0]));
        sb = detach(abs(divisor) < 1e-8f ? xb - xs : -dir);
        projection_dir = normalize(xb - xs);
    }
    float3 nb = detach(cross(b[0] - b[1], sb));
    if (dot(nb, nb) < 1e-20f)
        return 0;
    nb = normalize(nb);
    nb *= -sign(dot(nb, b[2] - b[0]));
    float3 xd = intersect_triangle(xb, projection_dir, d, false);
    float3 nd = detach(normalize(cross(d[1] - d[0], d[2] - d[0])));
    float3 u = cross(nb, nd);
    if (dot(u, u) < 1e-20f)
        return 0;
    u = normalize(cross(u, nd));
    u *= sign(dot(u, nb));
    float projection = dot(xd, detach(u));
    return -(projection - detach(projection));
}

/// Source of a path segment: a mesh hit, a triangle in the plane of an analytic light (moving with the
/// light's translation slots), or a direction (infinite: directional and environment light samples, and
/// BSDF samples that escape).
struct SegmentSource
{
    bool fixed;
    bool infinite;
    HitRef hit;
    float3 triangle[3];
    /// Light plane normal, or the direction of an infinite source.
    float3 normal;
    uint slot;
};

SegmentSource mesh_source(HitRef h)
{
    SegmentSource s = {};
    s.hit = h;
    s.slot = NO_SLOT;
    return s;
}

SegmentSource direction_source(float3 dir)
{
    SegmentSource s = {};
    s.fixed = true;
    s.infinite = true;
    s.normal = dir;
    s.slot = NO_SLOT;
    return s;
}

/// Area lights use their plane; point lights a plane facing the receiver (a small disk light).
SegmentSource light_source(LightRecord l, float3 receiver)
{
    if (l.kind == LIGHT_DIRECTIONAL)
        return direction_source(l.pos);
    SegmentSource s = {};
    s.fixed = true;
    s.normal = l.kind == LIGHT_AREA ? l.normal : normalize(receiver - l.pos);
    float3 t = normalize(abs(s.normal.x) > 0.5 ? cross(s.normal, float3(0, 1, 0)) : cross(s.normal, float3(1, 0, 0)));
    s.triangle = { l.pos, l.pos + t, l.pos + cross(s.normal, t) };
    s.slot = l.slot;
    return s;
}

Vertex source_vertex(SegmentSource s)
{
    if (!s.fixed)
        return load_vertex(0, s.hit);
    Vertex v = { s.triangle[0], s.normal, 1 };
    return v;
}

[Differentiable]
float segment_velocity(float theta, no_diff SegmentSource source, no_diff EdgeRef edge, no_diff HitRef receiver,
                       no_diff float t, no_diff float3 dir)
{
    float3 s[3], b[3], d[3];
    if (source.fixed) {
        float3 offset = param3(theta, float3(0), offset_slot(source.slot, 3));
        [ForceUnroll]
        for (uint i = 0; i < 3; i++)
            s[i] = source.triangle[i] + offset;
    } else {
        load_triangle(theta, source.hit.gid, source.hit.prim, s);
    }
    load_edge(theta, edge, b);
    load_triangle(theta, receiver.gid, receiver.prim, d);
    return normal_velocity(s, b, d, t, dir, source.infinite);
}

/// Detached Jacobian and edge pdf factor of one photon on a segment from receiver D in direction `dir`
/// to source S. For an infinite source, the limit of a distant source: the distance ratios cancel.
float edge_weight(Vertex s, Vertex d, float3 dir, bool infinite, float3 xb, float3 edge_dir)
{
    float3 sd_dir = -dir;
    float3 projected = normalize(cross(d.n, cross(edge_dir, sd_dir)));
    float sin_b = length(cross(edge_dir, sd_dir)), sin_d = length(cross(sd_dir, projected));
    if (sin_b <= 1e-6f || sin_d <= 1e-6f)
        return 0;
    if (infinite)
        return sin_b / sin_d * abs(dot(d.n, dir)) * photon_edge_length / 0.5;
    float j = length(d.p - s.p) / length(xb - s.p) * sin_b / sin_d;
    float source_cos = abs(dot(s.n, -normalize(s.p - xb)));
    if (source_cos == 0 || length(s.p - xb) == 0)
        return 0;
    j *= dot(s.p - xb, s.p - xb) / source_cos;
    float3 ds_dir = s.p - d.p;
    float dist = length(ds_dir);
    ds_dir /= dist;
    float g = abs(dot(d.n, ds_dir) * dot(s.n, -ds_dir)) / (dist * dist);
    return j * g * photon_edge_length / 0.5;
}

struct Neighbors
{
    float dist[16];
    uint index[16];
    uint count;
    float max_dist;
};

/// Gathers up to 16 photons on silhouette edges closest to the segment.
Neighbors gather(float3 origin, float3 dir, float t_max)
{
    Neighbors knn = {};
    RayDesc ray = { origin, ray_epsilon, dir, t_max - ray_epsilon };
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(photon_tlas, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {
        if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE)
            continue;
        uint i = q.CandidatePrimitiveIndex();
        Photon photon = photons[i];
        float3 oc = photon.position - origin;
        float t_center = dot(oc, dir);
        float3 perpendicular = oc - t_center * dir;
        float dist = length(perpendicular);
        if (dist > photon_radius || t_center + photon_radius < ray.TMin || t_center - photon_radius > ray.TMax)
            continue;
        if (photon.has_second_face != 0 && sign(dot(dir, photon.normal0)) * sign(dot(dir, photon.normal1)) > -0.5f)
            continue;
        if (knn.count < 16) {
            knn.dist[knn.count] = dist;
            knn.index[knn.count] = i;
            knn.count++;
            knn.max_dist = max(knn.max_dist, dist);
        } else if (dist < knn.max_dist) {
            uint worst = 0;
            for (uint k = 1; k < 16; k++)
                if (knn.dist[k] > knn.dist[worst])
                    worst = k;
            knn.dist[worst] = dist;
            knn.index[worst] = i;
            knn.max_dist = 0;
            for (uint k = 0; k < 16; k++)
                knn.max_dist = max(knn.max_dist, knn.dist[k]);
        }
    }
    return knn;
}

/// Photon estimate on one path segment. `weight` is the detached path throughput through it.
float3 segment_gradient(HitRef receiver_hit, SegmentSource source, float3 weight, float3 adjoint)
{
    Vertex d = load_vertex(0, receiver_hit), s = source_vertex(source);
    float3 dir = source.infinite ? source.normal : normalize(s.p - d.p);
    Neighbors knn = gather(d.p, dir, source.infinite ? 1e30 : length(s.p - d.p));
    float r = knn.count == 16 ? knn.max_dist : photon_radius;
    float density = 1.0 / (r * r * float.PI * photon_count);
    float3 result = 0;
    for (uint k = 0; k < knn.count; k++) {
        Photon photon = photons[knn.index[k]];
        if (dot(photon.position - d.p, d.n) < 1e-4f || (!source.infinite && dot(photon.position - s.p, s.n) < 1e-4f))
            continue;
        if (check_visibility && (!visible(d.p, photon.position)
                                 || !(source.infinite ? visible_dir(photon.position, dir) : visible(s.p, photon.position))))
            continue;
        EdgeRef e = edge_ref(photon_edges[photon.edge]);
        float3 p[3];
        load_edge(0, e, p);
        float factor = density * edge_weight(s, d, dir, source.infinite, photon.position, normalize(p[0] - p[1]));
        if (factor == 0)
            continue;
        if (backward) {
            var theta = diffPair(0.0);
            bwd_diff(segment_velocity)(theta, source, e, receiver_hit, photon.t, dir, dot(adjoint, weight) * factor);
        } else {
            result += weight * factor * fwd_diff(segment_velocity)(diffPair(0.0, 1.0), source, e, receiver_hit, photon.t, dir).d;
        }
    }
    return result;
}

/// Per-vertex detached contributions of a recorded camera path (radiance towards the previous vertex).
public void render_secondary(uint2 pixel, uint spp_index, uint spp)
{
    uint index = pixel.y * camera.dims.x + pixel.x;
    UniformSampleGenerator sg = UniformSampleGenerator(pixel, 0x40000000 | (seed * spp + spp_index));
    PathRecord rec;
    if (!trace_path(camera.position, camera_dir(float2(pixel) + next_float2(sg)), sg, rec))
        return;
    // Prefix throughput and suffix radiance give the weight of every path through a segment.
    float3 suffix[MAX_DEPTH + 2];
    path_terms(rec, suffix);
    float3 adjoint = backward ? dl_di[index] / spp : float3(0);
    float3 root = 1;
    float3 result = 0;
    for (uint k = 0; k < rec.length; k++) {
        LightRecord l = rec.light[k];
        VertexTerms t = rec.terms[k];
        if (l.kind == LIGHT_TRIANGLE)
            result += segment_gradient(rec.hit[k], mesh_source(l.hit), root * t.nee, adjoint);
        else if (l.kind != LIGHT_NONE)
            result += segment_gradient(rec.hit[k], light_source(l, load_vertex(0, rec.hit[k]).p), root * t.nee, adjoint);
        if (k + 1 < rec.length && rec.pdf[k + 1] > 0)
            result += segment_gradient(rec.hit[k], mesh_source(rec.hit[k + 1]), root * t.step * suffix[k + 1], adjoint);
        else if (k + 1 == rec.length && rec.miss_pdf > 0)
            result += segment_gradient(rec.hit[k], direction_source(rec.miss_dir), root * t.step, adjoint);
        root *= t.step;
    }
    if (!backward)
        secondary_boundary[index] += result / spp;
}
