// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Detached scene traversal for devices with ray-tracing pipelines but no inline ray queries.
#include "falcor2/render/scene_ray_tracing.slangh"

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
