// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Scene traversal (detached): inline ray queries, or a ray-tracing pipeline (TraceRay) for devices without them.
//
// A program's traversal is the link-time type SceneTraversal (PSDR's traversal option, exported by the renderer's
// prelude), so each program contains only one implementation. TraceRay is valid only in ray-tracing stages: behind
// a branch on a link-time constant it would remain in the compute kernels of inline traversal, which the HLSL target
// rejects (E36107).
//
// The opacity methods take the evaluator's seed, not an OpacityEvaluator: a requirement whose signature names another
// link-time type crashes the Slang front end (2026.16) when the program exports SceneTraversal.
#include "falcor2/render/scene_ray_tracing.slangh"

public interface IPsdrTraversal
{
    /// Closest triangle hit, all geometry opaque.
    public static bool closest_opaque(RayDesc ray, out TriangleHitInfo hit);
    /// Whether no triangle lies along the ray, all geometry opaque.
    public static bool visible_opaque(RayDesc ray);
    /// Closest triangle hit with the opacity evaluation of OpacityEvaluator(opacity_seed) (falcor2's
    /// RayQuerySceneIntersector for triangles; the reference path tracer's any-hit shader evaluates the same opacity per
    /// candidate).
    public static bool closest_opacity(Ray ray, uint opacity_seed, out TriangleHitInfo hit);
    public static bool occluded_opacity(Ray ray, uint opacity_seed);
    /// Up to 16 photons closest to the segment (secondary.slang's gather()).
    public static Neighbors gather_photons(RayDesc ray);
};

/// Inline ray queries (valid in every stage).
public struct InlineTraversal : IPsdrTraversal
{
    public static bool closest_opaque(RayDesc ray, out TriangleHitInfo hit)
    {
        hit = TriangleHitInfo(GeometryInstanceID(0), 0, float2(0));
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(g_scene.tlas, RAY_FLAG_NONE, 0xff, ray);
        while (q.Proceed()) {}
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            return false;
        hit = TriangleHitInfo(GeometryInstanceID(q.CommittedInstanceID() + q.CommittedGeometryIndex()),
                              q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
        return true;
    }

    public static bool visible_opaque(RayDesc ray)
    {
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(g_scene.tlas, RAY_FLAG_NONE, 0xff, ray);
        while (q.Proceed()) {}
        return q.CommittedStatus() == COMMITTED_NOTHING;
    }

    public static bool closest_opacity(Ray ray, uint opacity_seed, out TriangleHitInfo hit)
    {
        OpacityEvaluator opacity = OpacityEvaluator(opacity_seed);
        hit = TriangleHitInfo(GeometryInstanceID(0), 0, float2(0));
        RayQuery<OpacityEvaluator.RAY_FLAGS> q;
        q.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
        while (q.Proceed()) {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                TriangleHitInfo candidate = TriangleHitInfo(
                    GeometryInstanceID(q.CandidateInstanceID() + q.CandidateGeometryIndex()), q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics());
                if (opacity.is_valid_hit(ray, candidate, SHADING_HINTS))
                    q.CommitNonOpaqueTriangleHit();
            }
        }
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            return false;
        hit = TriangleHitInfo(GeometryInstanceID(q.CommittedInstanceID() + q.CommittedGeometryIndex()),
                              q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
        return true;
    }

    public static bool occluded_opacity(Ray ray, uint opacity_seed)
    {
        RayQuerySceneIntersector intersector = RayQuerySceneIntersector(OpacityEvaluator(opacity_seed));
        return intersector.is_occluded(ray, SHADING_HINTS);
    }

    public static Neighbors gather_photons(RayDesc ray)
    {
        Neighbors knn = {};
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(photon_tlas, RAY_FLAG_NONE, 0xff, ray);
        while (q.Proceed()) {
            if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE)
                continue;
            uint i = q.CandidatePrimitiveIndex();
            float dist, t_center;
            if (photon_candidate(i, ray.Origin, ray.Direction, ray.TMin, ray.TMax, dist, t_center))
                insert_neighbor(knn, i, dist);
        }
        return knn;
    }
};

/// Ray-tracing pipeline: the tracing entry points run as ray-generation shaders.
public struct PipelineTraversal : IPsdrTraversal
{
    public static bool closest_opaque(RayDesc ray, out TriangleHitInfo hit)
    {
        return trace_scene_closest(ray, OpacityEvaluator(0), RAY_FLAG_FORCE_OPAQUE, hit);
    }

    public static bool visible_opaque(RayDesc ray)
    {
        return trace_scene_visible(ray, OpacityEvaluator(0), RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES);
    }

    public static bool closest_opacity(Ray ray, uint opacity_seed, out TriangleHitInfo hit)
    {
        return trace_scene_closest(ray.to_ray_desc(), OpacityEvaluator(opacity_seed), OpacityEvaluator.RAY_FLAGS, hit);
    }

    public static bool occluded_opacity(Ray ray, uint opacity_seed)
    {
        return !trace_scene_visible(ray.to_ray_desc(), OpacityEvaluator(opacity_seed), OpacityEvaluator.RAY_FLAGS);
    }

    public static Neighbors gather_photons(RayDesc ray)
    {
        PhotonPayload payload = {};
        TraceRay(photon_tlas, RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xff,
                 PHOTON_HIT_GROUP_INDEX, 0, PHOTON_MISS_INDEX, ray, payload);
        return payload.neighbors;
    }
};

/// The traversal of a program (exported by the renderer's prelude).
public extern struct SceneTraversal : IPsdrTraversal = InlineTraversal;

static const uint PSDR_INTERSECT_RAY = 0;
static const uint PSDR_VISIBILITY_RAY = 1;

struct SceneHitPayload
{
    TriangleHitInfo hit;
    bool valid;
    OpacityEvaluator opacity;
};

struct SceneVisibilityPayload
{
    bool visible;
    OpacityEvaluator opacity;
};

bool trace_scene_closest(RayDesc ray, OpacityEvaluator opacity, uint flags, out TriangleHitInfo hit)
{
    SceneHitPayload payload = { TriangleHitInfo(GeometryInstanceID(0), 0, float2(0)), false, opacity };
    // The corresponding inline closest-hit queries never commit procedural candidates.
    TraceRay(g_scene.tlas, flags | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, 0xff,
             PSDR_INTERSECT_RAY, 0, PSDR_INTERSECT_RAY, ray, payload);
    hit = payload.hit;
    return payload.valid;
}

bool trace_scene_visible(RayDesc ray, OpacityEvaluator opacity, uint flags)
{
    SceneVisibilityPayload payload = { false, opacity };
    TraceRay(g_scene.tlas, flags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xff, PSDR_VISIBILITY_RAY, 0, PSDR_VISIBILITY_RAY, ray, payload);
    return payload.visible;
}

SCENE_RAY_TYPE_MISS(psdr_intersect, psdr_intersect_miss, SceneHitPayload);
void psdr_intersect_miss(inout SceneHitPayload payload) { payload.valid = false; }

SCENE_RAY_TYPE_CLOSEST_HIT(psdr_intersect, psdr_intersect_hit, SceneHitPayload);
void psdr_intersect_hit<D : IDeferredHitInfo>(inout SceneHitPayload payload, D deferred_hit)
{
    IHitInfo hit = deferred_hit.create();
    if (let triangle = hit as TriangleHitInfo) {
        payload.hit = triangle;
        payload.valid = true;
    }
}

SCENE_RAY_TYPE_ANY_HIT(psdr_intersect, psdr_intersect_any_hit, SceneHitPayload);
void psdr_intersect_any_hit<D : IDeferredHitInfo>(inout SceneHitPayload payload, D deferred_hit)
{
    Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    if (!payload.opacity.is_valid_hit(ray, deferred_hit.create(), MaterialInstanceHints::adjust_shading_normal))
        IgnoreHit();
}

SCENE_RAY_TYPE_MISS(psdr_visibility, psdr_visibility_miss, SceneVisibilityPayload);
void psdr_visibility_miss(inout SceneVisibilityPayload payload) { payload.visible = true; }

SCENE_RAY_TYPE_ANY_HIT(psdr_visibility, psdr_visibility_any_hit, SceneVisibilityPayload);
void psdr_visibility_any_hit<D : IDeferredHitInfo>(inout SceneVisibilityPayload payload, D deferred_hit)
{
    Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    if (payload.opacity.is_valid_hit(ray, deferred_hit.create(), MaterialInstanceHints::adjust_shading_normal))
        AcceptHitAndEndSearch();
    else
        IgnoreHit();
}
