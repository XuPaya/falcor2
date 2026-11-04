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

/// Stored object-space shading normal (k = 0) or tangent (k = 1) of vertex v, the vertex attributes that shading
/// frames interpolate; their derivatives are frame_tangent (forward mode) and frame_grad (backward mode).
[Differentiable, ForwardDerivative(fwd_vertex_attribute), BackwardDerivative(bwd_vertex_attribute)]
float3 vertex_attribute(float theta, no_diff uint v, no_diff uint k)
{
    TriangleVertex t = g_scene.get_triangle_vertex(v);
    return k == 0 ? t.normal : t.tangent;
}

DifferentialPair<float3> fwd_vertex_attribute(DifferentialPair<float> theta, uint v, uint k)
{
    TriangleVertex t = g_scene.get_triangle_vertex(v);
    return diffPair(k == 0 ? t.normal : t.tangent, theta.d * frame_tangent[2 * v + k]);
}

void bwd_vertex_attribute(inout DifferentialPair<float> theta, uint v, uint k, float3 d)
{
    theta = diffPair(theta.p, 0.0);
    float n = length(d);
    float limit = vertex_derivative_limit * frame_limit_scale;
    if (limit > 0 && n > limit)
        d *= limit / n;
    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
        atomic_add(frame_grad, 24 * v + 12 * k + 4 * i, d[i]);
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

/// The interpolated shading frame of a hit as make_triangle_surface_interaction() builds it, as a function of the
/// stored vertex normals and tangents (vertex_attribute()). handedness: the primal frame's.
[Differentiable]
Frame interpolated_frame(float theta, no_diff HitRef h, no_diff float handedness)
{
    uint[3] ids = g_scene.get_triangle_indices(h.gid, h.prim);
    float w0 = 1.0 - h.bary.x - h.bary.y;
    float3 n = w0 * vertex_attribute(theta, ids[0], 0) + h.bary.x * vertex_attribute(theta, ids[1], 0)
        + h.bary.y * vertex_attribute(theta, ids[2], 0);
    float3 t = w0 * vertex_attribute(theta, ids[0], 1) + h.bary.x * vertex_attribute(theta, ids[1], 1)
        + h.bary.y * vertex_attribute(theta, ids[2], 1);
    float3x3 world_from_object = float3x3(g_scene.get_world_from_object(h.gid));
    float3x3 inverse_transpose = transpose(float3x3(g_scene.get_object_from_world(h.gid)));
    return Frame(normalize(mul(inverse_transpose, n)), normalize(mul(world_from_object, t)), handedness);
}

/// Backward mode: the adjoint of the shading frame of the vertex being differentiated (vertex_shading_frame()), its hit
/// and handedness. backward_vertices() hands it to backward_frames(), which chains it to the stored vertex normals and
/// tangents in a kernel of its own: in the BSDFs' reverse-mode kernels, that chain makes the SpongeCake kernel of a
/// Vulkan material-family pass hang.
static Frame.Differential frame_adjoint;
static HitRef frame_hit;
static float frame_handedness;
static bool frame_pending = false;

/// The interpolated shading frame `primal` of hit h (SurfaceInteraction.shading_frame_ws); with `active`, its
/// derivatives are interpolated_frame()'s (backward mode: through frame_adjoint), otherwise zero.
[Differentiable, ForwardDerivative(fwd_vertex_shading_frame), BackwardDerivative(bwd_vertex_shading_frame)]
Frame vertex_shading_frame(float theta, no_diff HitRef h, no_diff Frame primal, no_diff bool active)
{
    return primal;
}

DifferentialPair<Frame> fwd_vertex_shading_frame(DifferentialPair<float> theta, HitRef h, Frame primal, bool active)
{
    Frame.Differential d = Frame.dzero();
    if (active)
        d = fwd_diff(interpolated_frame)(theta, h, primal.handedness_sign()).d;
    return diffPair(primal, d);
}

void bwd_vertex_shading_frame(inout DifferentialPair<float> theta, HitRef h, Frame primal, bool active,
                              Frame.Differential d)
{
    theta = diffPair(theta.p, 0.0);
    if (!active)
        return;
    if (!frame_pending)
        frame_adjoint = Frame.dzero();
    frame_adjoint = Frame.dadd(frame_adjoint, d);
    frame_hit = h;
    frame_handedness = primal.handedness_sign();
    frame_pending = true;
}

bool intersect(float3 origin, float3 dir, float t_max, out HitRef h)
{
    h = {};
    RayDesc ray = { origin, ray_epsilon, dir, t_max };
    TriangleHitInfo hit;
    bool found = SceneTraversal.closest_opaque(ray, hit);
    h.gid = hit.geometry_instance_id;
    h.prim = hit.primitive_index;
    h.bary = hit.barycentrics;
    return found;
}

bool visible(float3 a, float3 b)
{
    float3 d = b - a;
    float t = length(d);
    if (t <= 2 * ray_epsilon)
        return true;
    RayDesc ray = { a, ray_epsilon, d / t, t - ray_epsilon };
    return SceneTraversal.visible_opaque(ray);
}
