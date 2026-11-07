// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Probe for test_adjoint_pathtracer.py: directional transmittance of the surface hit by a vertical ray at
// the scene's center (the glass in the window scene), by sampling its BSDF.

import falcor2.render;
import falcor2.utils;

#include "material_probe.slangh"

public float transmittance_probe(uint tid, float3 wi)
{
    const IHitInfo hit = probe_hit();
    if (!hit.valid)
        return -1.0;
    const float3 dir = normalize(wi);
    SurfaceInteraction si = g_scene.make_surface_interaction(hit, dir);
    let mi = g_scene.get_material_instance(hit, si, ExplicitLodSampler(0), MaterialInstanceHints::adjust_shading_normal);
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid, 7u), 0u);
    MaterialSample sample = {};
    if (!mi.sample(si, sg, sample))
        return 0.0;
    if (dot(sample.wo_ws, si.normal_ws) * dot(dir, si.normal_ws) >= 0.0)
        return 0.0;
    return luminance(sample.weight);
}

/// The same transmittance by integrating eval() over the transmission hemisphere (uniform directions).
public float transmittance_probe_eval(uint tid, float3 wi)
{
    const IHitInfo hit = probe_hit();
    if (!hit.valid)
        return -1.0;
    const float3 dir = normalize(wi);
    SurfaceInteraction si = g_scene.make_surface_interaction(hit, dir);
    let mi = g_scene.get_material_instance(hit, si, ExplicitLodSampler(0), MaterialInstanceHints::adjust_shading_normal);
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid, 11u), 0u);
    const float3 local = square_to_uniform_sphere(next_float2(sg));
    const float3 n = dot(dir, si.normal_ws) >= 0.0 ? si.normal_ws : -si.normal_ws;
    const Frame frame = Frame(-n);
    const float3 wo = local.x * frame.tangent + local.y * frame.bitangent + abs(local.z) * frame.normal;
    return luminance(mi.eval(si, wo, sg)) * float.TWO_PI;
}

/// Transmittance seen by a path tracer: the BSDF integrated over its viewing direction (wi uniform on the side
/// away from the light, wo = toward the light), i.e. the flux transmitted from a collimated beam along -wi_light.
public float transmittance_probe_reverse(uint tid, float3 wi)
{
    const IHitInfo hit = probe_hit();
    if (!hit.valid)
        return -1.0;
    const float3 light = normalize(wi);
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid, 13u), 0u);
    const float3 local = square_to_uniform_sphere(next_float2(sg));
    SurfaceInteraction probe = g_scene.make_surface_interaction(hit, light);
    const float3 n = probe.front_facing ? probe.normal_ws : -probe.normal_ws;
    const Frame frame = Frame(-n);
    const float3 view = local.x * frame.tangent + local.y * frame.bitangent + abs(local.z) * frame.normal;
    SurfaceInteraction si = g_scene.make_surface_interaction(hit, view);
    let mi = g_scene.get_material_instance(hit, si, ExplicitLodSampler(0), MaterialInstanceHints::adjust_shading_normal);
    // eval = f |cos_light|; flux fraction = integral f |cos_view| d(view)
    const float cos_light = abs(dot(light, n));
    return luminance(mi.eval(si, light, sg)) * abs(local.z) / cos_light * float.TWO_PI;
}

/// BSDF value f (without foreshortening) of the surface hit by the vertical probe ray, for light from wi_light and
/// view direction wo (world space; wi of the interaction = the camera side wo, as the path tracer evaluates it).
public float3 bsdf_probe_eval(float3 wi_light, float3 wo)
{
    const IHitInfo hit = probe_hit();
    if (!hit.valid)
        return float3(-1.0);
    SurfaceInteraction si = g_scene.make_surface_interaction(hit, normalize(wo));
    let mi = g_scene.get_material_instance(hit, si, ExplicitLodSampler(0), MaterialInstanceHints::adjust_shading_normal);
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(0u, 17u), 0u);
    // eval returns f |cos(wi_light)| (the interaction's wi = wo, as in the path tracer's NEE).
    const float3 value = mi.eval(si, normalize(wi_light), sg);
    return value / max(abs(dot(normalize(wi_light), si.normal_ws)), 1e-6);
}
