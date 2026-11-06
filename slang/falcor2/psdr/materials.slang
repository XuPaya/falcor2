// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Materials: shading frames and StandardMaterial and SpongeCakeMaterial BSDFs and emission with derivatives
// (OpenPBRMaterial in openpbr.slang). Other materials are evaluated without derivatives.

static const uint MATERIAL_OTHER = 0;
/// Same material hints as the reference path tracer.
static const MaterialInstanceHints SHADING_HINTS = MaterialInstanceHints::adjust_shading_normal;
static const uint MATERIAL_STANDARD = 1;
static const uint MATERIAL_OPENPBR = 2;
static const uint MATERIAL_SPONGECAKE = 3;

/// Inlined so Slang folds the link-time family test: each family's backward program then contains only that family's
/// reverse-mode BSDF code. As a call, the test stays a runtime comparison and DXC compiles all families into every
/// family's program (hours for the largest kernels).
[ForceInline]
bool backward_material_family(uint family)
{
    return BACKWARD_MATERIAL_FAMILY == 0xffffffff || BACKWARD_MATERIAL_FAMILY == family;
}

/// Select the current vertex only. Its derivative may still write to neighboring vertices or an emitter of
/// another family, so neither geometry/parameter writes nor surface/emission decoding are filtered.
bool backward_material_hit(HitRef h)
{
    if (BACKWARD_MATERIAL_FAMILY == 0xffffffff)
        return true;
    MaterialID material = g_scene.get_geometry_instance_desc(h.gid).header.material_id;
    uint type = g_scene.material_system.get_material_data(material).header.type_id;
    uint family = type == standard_type_id ? MATERIAL_STANDARD
        : type == openpbr_type_id ? MATERIAL_OPENPBR
        : type == spongecake_type_id ? MATERIAL_SPONGECAKE : MATERIAL_OTHER;
    return backward_material_family(family);
}

/// Non-differentiable shading context of a path vertex, built from the primal incident direction.
struct Surface
{
    SurfaceInteraction si;
    HitRef hit;
    /// The shading frame follows the stored vertex normals and tangents (frame_derivatives, camera-path vertices;
    /// light-path vertices keep constant frames: no geometric derivatives along light paths).
    bool vertex_frame;
    MaterialID material;
    uint kind;
    uint slot;
    StandardMaterial standard;
    StandardMaterialInstance standard_instance;
    OpenPBRMaterial openpbr;
    SpongeCakeMaterial spongecake;
};

/// exterior_ior: the IOR outside the surface (the path tracer's interior tracking).
Surface make_surface(HitRef h, float3 wi_ws, float exterior_ior = 1.0)
{
    Surface s = {};
    s.si = g_scene.make_triangle_surface_interaction(TriangleHitInfo(h.gid, h.prim, h.bary), wi_ws);
    s.si.exterior_ior = exterior_ior;
    s.hit = h;
    s.material = g_scene.get_geometry_instance_desc(h.gid).header.material_id;
    s.slot = uint(s.material) * MATERIAL_SLOTS;
    MaterialData data = g_scene.material_system.get_material_data(s.material);
    if (HAS_STANDARD_MATERIAL && data.header.type_id == standard_type_id) {
        s.kind = MATERIAL_STANDARD;
        s.standard = reinterpret<StandardMaterial>(data);
        s.standard_instance = s.standard.setup_material_instance(s.si, ExplicitLodSampler(), SHADING_HINTS);
    } else if (HAS_OPENPBR_MATERIAL && data.header.type_id == openpbr_type_id) {
        s.kind = MATERIAL_OPENPBR;
        s.openpbr = reinterpret<OpenPBRMaterial>(data);
    } else if (HAS_SPONGECAKE_MATERIAL && data.header.type_id == spongecake_type_id) {
        s.kind = MATERIAL_SPONGECAKE;
        s.spongecake = reinterpret<SpongeCakeMaterial>(data);
    }
    return s;
}

IMaterialInstance material_instance(Surface s)
{
    return g_scene.get_material(s.material).setup_material_instance(s.si, ExplicitLodSampler(), SHADING_HINTS);
}

/// Shading frame with the material's normal map (texels and scale differentiable), as in the materials'
/// setup_material_instance() with SHADING_HINTS. With s.vertex_frame, the interpolated frame is differentiable with
/// respect to the stored vertex normals and tangents.
[Differentiable]
Frame shading_frame(float theta, no_diff Surface s, no_diff uint normal_texture, no_diff float4 sampled, float scale,
                    no_diff bool orient_backface)
{
    Frame frame = vertex_shading_frame(theta, s.hit, s.si.shading_frame_ws, s.vertex_frame);
    if (RawTextureHandle(normal_texture).is_valid()) {
        float3 n = stdmat::decode_normal_map(texture_param(theta, normal_texture, s.si.uv, sampled).xyz);
        n = normalize(lerp(float3(0.0, 0.0, 1.0), n, scale));
        n = normal_maps ? n : detach(n);
        frame = Frame(frame.to_global(n), frame.tangent, no_diff s.si.shading_frame_ws.handedness_sign());
    }
    frame = stdmat::orient_shading_frame(s.si, frame, orient_backface);
    // adjust_shading_normal(): blend the shading normal towards the geometric normal at grazing angles. (Made
    // differentiable, falcor2's crashes Slang's backward code generation.)
    float3 ng = no_diff s.si.front_normal_ws();
    float sign_n = dot(detach(frame.normal), ng) >= 0.0 ? 1.0 : -1.0;
    float cos_theta = dot(s.si.wi_ws, sign_n * frame.normal);
    if (detach(cos_theta) <= 0.1)
        frame = Frame(normalize(sign_n * lerp(ng, sign_n * frame.normal, saturate(cos_theta * 10.0))), frame.tangent,
                      no_diff detach(frame).handedness_sign());
    return normal_maps || s.vertex_frame ? frame : detach(frame);
}

[Differentiable]
Frame standard_frame(float theta, no_diff Surface s)
{
    StandardMaterial m = s.standard;
    float4 sampled = m.normal_texture.sample(ExplicitLodSampler(), s.si.uv, float4(0.0, 0.0, 1.0, 0.0));
    return shading_frame(theta, s, m.normal_texture.data, sampled,
                         param(theta, float(m.normal_texture_scale), s.slot + 14),
                         is_set(m.header.flags, MaterialFlags::two_sided));
}

/// StandardMaterial.setup_material_instance() with its factors and texels routed through param().
[Differentiable]
StandardBSDFData standard_bsdf_data(float theta, no_diff Surface s)
{
    StandardMaterial m = s.standard;
    float2 uv = s.si.uv;
    ExplicitLodSampler lod = ExplicitLodSampler();
    float3 base_color = texture_param(theta, m.base_color_texture.data, uv,
                                      m.base_color_texture.sample(lod, uv, float4(1.0))).rgb
        * param3(theta, float3(m.base_color_factor.xyz), s.slot);
    float4 metallic_roughness = texture_param(theta, m.metallic_roughness_texture.data, uv,
                                              m.metallic_roughness_texture.sample(lod, uv, float4(1.0)));
    float metallic = clamp(
        channel(metallic_roughness, m.metallic_texture_channel) * param(theta, float(m.metallic_factor), s.slot + 3),
        0.0,
        1.0
    );
    float roughness = clamp(
        channel(metallic_roughness, m.roughness_texture_channel) * param(theta, float(m.roughness_factor), s.slot + 4),
        0.0,
        1.0
    );
    float ior = max(param(theta, float(m.ior), s.slot + 5), 1e-3);
    float f = (ior - 1.0) / (ior + 1.0);
    StandardBSDFData data = s.standard_instance.bsdf_data;
    data.diffuse = base_color * (1.0 - metallic);
    data.specular = lerp(float3(f * f), base_color, metallic);
    data.roughness = roughness;
    data.metallic = metallic;
    data.eta = s.si.front_facing ? (s.si.exterior_ior / ior) : (ior / s.si.exterior_ior);
    data.transmission = texture_param(theta, m.transmission_texture.data, uv,
                                      m.transmission_texture.sample(lod, uv, float4(1.0))).rgb
        * param3(theta, float3(m.transmission_factor.xyz), s.slot + 11);
    data.diffuse_transmission = saturate(param(theta, float(m.diffuse_transmission_factor), s.slot + 9));
    data.specular_transmission = saturate(param(theta, float(m.specular_transmission_factor), s.slot + 10));
    return data;
}

[Differentiable]
Frame spongecake_frame(float theta, no_diff Surface s)
{
    SpongeCakeMaterial m = s.spongecake;
    float4 sampled = m.normal_texture.sample(ExplicitLodSampler(), s.si.uv, float4(0.0, 0.0, 1.0, 0.0));
    return shading_frame(theta, s, m.normal_texture.data, sampled, param(theta, m.normal_texture_scale, s.slot + 14),
                         true);
}

/// SpongeCakeMaterial.setup_material_instance() with its factors and texels routed through param(). The tables'
/// derivatives (thicknesses and roughnesses) enter through table_tangent.
[Differentiable]
SpongeCakeBSDF spongecake_bsdf(float theta, no_diff Surface s)
{
    SpongeCakeMaterial m = s.spongecake;
    float2 uv = s.si.uv;
    ExplicitLodSampler lod = ExplicitLodSampler();
    SpongeCakeBSDF b;
    b.body.albedo = saturate(texture_param(theta, m.albedo_texture.data, uv,
                                           m.albedo_texture.sample(lod, uv, float4(1.0))).rgb
                             * param3(theta, m.albedo, s.slot));
    b.body.tau = max(param(theta, m.thickness, s.slot + 3), 0.0);
    b.body.alpha = clamp(param(theta, m.roughness, s.slot + 4), 0.01, 1.0);
    b.body.fiber = (m.layer_flags & 1) != 0;
    b.body.hg = (m.layer_flags & 8) != 0;
    b.top.albedo = saturate(texture_param(theta, m.top_albedo_texture.data, uv,
                                          m.top_albedo_texture.sample(lod, uv, float4(1.0))).rgb
                            * param3(theta, m.top_albedo, s.slot + 5));
    b.top.tau = max(param(theta, m.top_thickness, s.slot + 8), 0.0);
    b.top.alpha = clamp(param(theta, m.top_roughness, s.slot + 9), 0.01, 1.0);
    b.top.fiber = (m.layer_flags & 2) != 0;
    b.top.hg = (m.layer_flags & 16) != 0;
    b.base_color = saturate(texture_param(theta, m.base_color_texture.data, uv,
                                          m.base_color_texture.sample(lod, uv, float4(1.0))).rgb
                            * param3(theta, m.base_color, s.slot + 10));
    b.base_weight = saturate(param(theta, m.base_weight, s.slot + 13));
    b.delta_transmission = (m.layer_flags & 4) != 0;
    b.forward_lobes = (m.layer_flags & 32) != 0;
    b.tables = m.tables;
    b.table_tangent = float4(b.top.tau - detach(b.top.tau), b.top.alpha - detach(b.top.alpha),
                             b.body.tau - detach(b.body.tau), b.body.alpha - detach(b.body.alpha));
    b.table_derivatives = true;
    return b;
}

/// BSDF values including outgoing foreshortening for two outgoing directions (world-space and normalized,
/// or zero for a zero value), sharing the BSDF setup. Other materials return `other1` and `other2`, their
/// values recorded by the path tracer: evaluation code never dispatches through the material system, whose
/// code would be inlined at every call site.
[Differentiable]
void eval_bsdf(float theta, no_diff Surface s, float3 wi_ws, float3 wo1, float3 wo2, no_diff float3 other1,
               no_diff float3 other2, out float3 f1, out float3 f2)
{
    f1 = float3(0);
    f2 = float3(0);
    if (HAS_STANDARD_MATERIAL && backward_material_family(MATERIAL_STANDARD) && s.kind == MATERIAL_STANDARD) {
        UniformSampleGenerator sg = UniformSampleGenerator(0);
        Frame frame = standard_frame(theta, s);
        float3 wi_ls = frame.to_local(wi_ws), wo_ls1 = frame.to_local(wo1), wo_ls2 = frame.to_local(wo2);
        StandardBSDF bsdf = StandardBSDF(wi_ls, standard_bsdf_data(theta, s));
        Frame primal = s.standard_instance.shading_frame_ws;
        if (no_diff stdmat::is_valid_hemisphere_reflection_or_transmission(s.si, primal, wi_ls, wo_ls1, wo1))
            f1 = bsdf.eval(wi_ls, wo_ls1, sg, BSDFContext());
        if (no_diff stdmat::is_valid_hemisphere_reflection_or_transmission(s.si, primal, wi_ls, wo_ls2, wo2))
            f2 = bsdf.eval(wi_ls, wo_ls2, sg, BSDFContext());
    } else if (HAS_OPENPBR_MATERIAL && backward_material_family(MATERIAL_OPENPBR) && s.kind == MATERIAL_OPENPBR) {
        eval_openpbr(theta, s, wi_ws, wo1, wo2, f1, f2);
    } else if (HAS_SPONGECAKE_MATERIAL && backward_material_family(MATERIAL_SPONGECAKE) && s.kind == MATERIAL_SPONGECAKE) {
        Frame frame = spongecake_frame(theta, s);
        float3 wi_ls = frame.to_local(wi_ws), wo_ls1 = frame.to_local(wo1), wo_ls2 = frame.to_local(wo2);
        SpongeCakeBSDF bsdf = spongecake_bsdf(theta, s);
        Frame primal = detach(frame);
        // Unroll the fixed directions to work around Vulkan reverse-mode loop failures.
        float3 f[2] = { float3(0), float3(0) };
        [ForceUnroll]
        for (uint d = 0; d < 2; d++) {
            float3 wo = d == 0 ? wo1 : wo2, wo_ls = d == 0 ? wo_ls1 : wo_ls2;
            if (no_diff stdmat::is_valid_hemisphere_reflection_or_transmission(s.si, primal, wi_ls, wo_ls, wo))
                f[d] = bsdf.eval(wi_ls, wo_ls);
        }
        f1 = f[0];
        f2 = f[1];
    } else {
        f1 = other1;
        f2 = other2;
    }
}

/// Derivative carrier (value 1) of a delta transmission sampled at a vertex seen from `wi_ws`: SpongeCakeMaterial's
/// unscattered transmittance (1 - base weight) exp(-sum tau lambda(wi)) over its primal value. Other materials' delta
/// lobes have no derivatives.
[Differentiable]
float delta_scale(float theta, no_diff Surface s, float3 wi_ws)
{
    if (!HAS_SPONGECAKE_MATERIAL || !backward_material_family(MATERIAL_SPONGECAKE) || s.kind != MATERIAL_SPONGECAKE)
        return 1.0;
    Frame frame = spongecake_frame(theta, s);
    SpongeCakeBSDF bsdf = spongecake_bsdf(theta, s);
    float t = (1.0 - bsdf.base_weight) * bsdf.transmittance(frame.to_local(wi_ws));
    return ratio(t, detach(t));
}

/// Whether a surface emits from both sides: two-sided materials do, as at the reference path tracer's emitter hits
/// (its next-event and emission sampling cover the front side only, so the back side is reached by BSDF sampling).
bool emits_both_sides(Surface s)
{
    return is_set(g_scene.material_system.get_material_data(s.material).header.flags, MaterialFlags::two_sided);
}

/// Radiance emitted by a surface towards `to_ws` (unnormalized). Emission is front-facing only, except for two-sided
/// materials (emits_both_sides()). Other materials return `other`, their recorded emitted radiance.
[Differentiable]
float3 eval_emission(float theta, no_diff Surface s, float3 normal, float3 to_ws, no_diff float3 other)
{
    if (dot(detach(normal), detach(to_ws)) <= 0 && !emits_both_sides(s))
        return float3(0);
    if (HAS_STANDARD_MATERIAL && s.kind == MATERIAL_STANDARD)
        return texture_param(theta, s.standard.emissive_texture.data, s.si.uv,
                             s.standard.emissive_texture.sample(ExplicitLodSampler(), s.si.uv, float4(1.0))).rgb
            * param3(theta, s.standard.emissive_factor, s.slot + 6);
    if (HAS_OPENPBR_MATERIAL && s.kind == MATERIAL_OPENPBR)
        return openpbr_emission(theta, s);
    return other;
}
