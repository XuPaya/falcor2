// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Probes for test_spongecake_material.py: the BSDF of the SpongeCake quad at the origin (normal +y) in world space.

import falcor2.render;
import falcor2.utils;

#include "material_probe.slangh"

/// (f |cos wo|, pdf) for light arriving from wi (pointing away from the surface) and outgoing direction wo.
public float4 eval_probe(float3 wi, float3 wo)
{
    const IHitInfo hit = probe_hit();
    if (!hit.valid)
        return float4(-1.0);
    SurfaceInteraction si = g_scene.make_surface_interaction(hit, normalize(wi));
    let mi = g_scene.get_material_instance(hit, si, ExplicitLodSampler(0), MaterialInstanceHints::none);
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(0u, 3u), 0u);
    BSDFEval e = mi.eval_with_pdf(si, normalize(wo), sg);
    return float4(e.value, e.pdf);
}

/// Sample (weight.rgb, flags) and (wo, pdf) for light arriving from wi.
public float4 sample_probe(uint tid, float3 wi, uint which)
{
    const IHitInfo hit = probe_hit();
    if (!hit.valid)
        return float4(-1.0);
    SurfaceInteraction si = g_scene.make_surface_interaction(hit, normalize(wi));
    let mi = g_scene.get_material_instance(hit, si, ExplicitLodSampler(0), MaterialInstanceHints::none);
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid, 12345u), 0u);
    MaterialSample s = {};
    if (!mi.sample(si, sg, s))
        return float4(0.0, 0.0, 0.0, -1.0);
    return which == 0 ? float4(s.weight, float(uint(s.flags))) : float4(s.wo_ws, s.pdf);
}
