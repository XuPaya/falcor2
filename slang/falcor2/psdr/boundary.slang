// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Visibility boundaries, shared by the boundary terms: edge lists, the surface behind a boundary and
// the boundary's normal velocity.

float boundary_factor(float3 point, float3 edge_point, float3 edge_dir, float3 normal)
{
    float3 d = normalize(point - camera.position), projected = normalize(cross(normal, cross(edge_dir, d)));
    return length(point - camera.position) / length(edge_point - camera.position) * length(cross(edge_dir, d))
        / length(cross(d, projected)) * abs(dot(normal, d));
}

/// Moving-ray intersection with the hit triangle, expressed in its reference (detached) coordinates.
[Differentiable]
float3 reference_point(float theta, no_diff HitRef h, float3 o, float3 d)
{
    float3 p[3];
    load_triangle(theta, h.gid, h.prim, p);
    float3 e1 = p[1] - p[0], e2 = p[2] - p[0];
    float3 q = cross(d, e2), t = o - p[0], r = cross(t, e1);
    float inv = 1 / dot(e1, q), u = dot(t, q) * inv, v = dot(d, r) * inv;
    return detach(p[0]) + u * detach(p[1] - p[0]) + v * detach(p[2] - p[0]);
}

struct EdgeRef
{
    GeometryInstanceID gid;
    uint prim;
    uint k;
    int prim1;
};

[Differentiable]
void load_edge(float theta, no_diff EdgeRef e, out float3 p[3])
{
    float3 t[3];
    load_triangle(theta, e.gid, e.prim, t);
    p[0] = t[e.k];
    p[1] = t[(e.k + 1) % 3];
    p[2] = t[(e.k + 2) % 3];
}

/// Detached geometric normal. Differentiable code must not call world_vertex() with a constant
/// theta: its backward derivative accumulates regardless of the incoming differential.
float3 surface_normal(HitRef h) { return load_vertex(0, h).n; }

/// Surface behind a visibility boundary seen from the camera: a mesh hit or, for the environment, a fixed
/// plane (point p, normal n) facing the camera.
struct Background
{
    bool is_hit;
    HitRef hit;
    float3 p;
    float3 n;
};

Background background(PathRecord rec, float3 origin, float3 dir)
{
    Background b = {};
    b.is_hit = rec.length > 0;
    b.hit = rec.hit[0];
    if (b.is_hit) {
        Vertex v = load_vertex(0, rec.hit[0]);
        b.p = v.p;
        b.n = v.n;
    } else {
        b.p = origin + dir * length(origin - camera.position);
        b.n = -dir;
    }
    return b;
}

/// Normal velocity of the visibility boundary seen from the camera through `origin`.
[Differentiable]
float boundary_velocity(float theta, no_diff Background b, no_diff float3 xb, no_diff float3 nb, no_diff bool on_edge,
                        no_diff EdgeRef e, no_diff float t)
{
    float3 origin = xb;
    if (on_edge) {
        float3 p[3];
        load_edge(theta, e, p);
        origin = lerp(p[0], p[1], t);
    }
    float3 d = normalize(origin - camera.position);
    float3 x = b.is_hit ? reference_point(theta, b.hit, origin, d) : origin + d * (dot(b.p - origin, b.n) / dot(d, b.n));
    float3 u = normalize(cross(cross(nb, b.n), b.n));
    u *= sign(dot(u, nb));
    return -dot(x, u);
}

float differentiate_velocity(Background b, float3 xb, float3 nb, bool on_edge, EdgeRef e, float t, float weight)
{
    if (backward) {
        var theta = diffPair(0.0);
        bwd_diff(boundary_velocity)(theta, b, xb, nb, on_edge, e, t, weight);
        return 0;
    }
    return fwd_diff(boundary_velocity)(diffPair(0.0, 1.0), b, xb, nb, on_edge, e, t).d;
}

/// Edge list entry with cumulative length distribution.
public struct Edge
{
    public uint gid;
    public uint prim;
    public uint k;
    public int prim1;
    public float cdf;
};

public StructuredBuffer<Edge> primary_edges;
public uint primary_edge_count;
public float primary_edge_length;
public StructuredBuffer<Edge> photon_edges;
public uint photon_edge_count;
public float photon_edge_length;

uint sample_edge(StructuredBuffer<Edge> edges, uint count, float u, out float t)
{
    uint lo = 0, hi = count - 1;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (edges[mid].cdf <= u)
            lo = mid + 1;
        else
            hi = mid;
    }
    float c0 = lo > 0 ? edges[lo - 1].cdf : 0;
    t = (u - c0) / (edges[lo].cdf - c0);
    return lo;
}

EdgeRef edge_ref(Edge e) { return { GeometryInstanceID(e.gid), e.prim, e.k, e.prim1 }; }
