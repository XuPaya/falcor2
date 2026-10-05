// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Differentiable geometry: world-space vertices (world_vertex(), whose derivative is the object-space
// vertex derivative), surface points in material form, and ray queries.

float3 object_vertex(uint v) { return g_scene.get_triangle_vertex(v).position; }

/// World-space vertex position; its derivative is the transformed object-space vertex derivative.
[Differentiable, ForwardDerivative(fwd_world_vertex), BackwardDerivative(bwd_world_vertex)]
float3 world_vertex(float theta, no_diff GeometryInstanceID gid, no_diff uint v)
{
    return mul(g_scene.get_world_from_object(gid), float4(object_vertex(v), 1)).xyz;
}

DifferentialPair<float3> fwd_world_vertex(DifferentialPair<float> theta, GeometryInstanceID gid, uint v)
{
    float4x4 m = g_scene.get_world_from_object(gid);
    return diffPair(mul(m, float4(object_vertex(v), 1)).xyz, theta.d * mul(m, float4(vertex_tangent[v], 0)).xyz);
}

void bwd_world_vertex(inout DifferentialPair<float> theta, GeometryInstanceID gid, uint v, float3 d)
{
    theta = diffPair(theta.p, 0.0);
    float3 g = mul(float4(d, 0), g_scene.get_world_from_object(gid)).xyz;
    float n = length(g);
    if (vertex_derivative_limit > 0 && n > vertex_derivative_limit)
        g *= vertex_derivative_limit / n;
    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
        atomic_add(vertex_grad, 12 * v + 4 * i, g[i]);
}

struct HitRef
{
    GeometryInstanceID gid;
    uint prim;
    float2 bary;
};

struct Vertex : IDifferentiable
{
    float3 p;
    float3 n;
    /// Area ratio A / detach(A): value 1, derivative of the material-form Jacobian.
    float J;
};

float transform_sign(GeometryInstanceID gid)
{
    float4x4 m = g_scene.get_world_from_object(gid);
    return determinant(float3x3(m[0].xyz, m[1].xyz, m[2].xyz)) < 0 ? -1.0 : 1.0;
}

/// Triangle vertex positions of a hit.
[Differentiable]
void load_triangle(float theta, no_diff GeometryInstanceID gid, no_diff uint prim, out float3 p[3])
{
    uint[3] ids = g_scene.get_triangle_indices(gid, prim);
    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
        p[i] = world_vertex(theta, gid, ids[i]);
}

/// Surface point with fixed barycentrics (material form).
[Differentiable]
Vertex load_vertex(float theta, no_diff HitRef h)
{
    float3 p[3];
    load_triangle(theta, h.gid, h.prim, p);
    Vertex v;
    v.p = p[0] + h.bary.x * (p[1] - p[0]) + h.bary.y * (p[2] - p[0]);
    float3 n = cross(p[1] - p[0], p[2] - p[0]);
    float area = length(n);
    v.n = n / area * no_diff transform_sign(h.gid);
    v.J = area / detach(area);
    return v;
}

bool intersect(float3 origin, float3 dir, float t_max, out HitRef h)
{
    h = {};
    RayDesc ray = { origin, ray_epsilon, dir, t_max };
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(g_scene.tlas, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {}
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return false;
    h.gid = GeometryInstanceID(q.CommittedInstanceID() + q.CommittedGeometryIndex());
    h.prim = q.CommittedPrimitiveIndex();
    h.bary = q.CommittedTriangleBarycentrics();
    return true;
}

bool visible(float3 a, float3 b)
{
    float3 d = b - a;
    float t = length(d);
    if (t <= 2 * ray_epsilon)
        return true;
    RayDesc ray = { a, ray_epsilon, d / t, t - ray_epsilon };
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(g_scene.tlas, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {}
    return q.CommittedStatus() == COMMITTED_NOTHING;
}
