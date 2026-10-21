// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct SpongeCakeMaterialInstance : IMaterialInstance
{
    public Frame shading_frame_ws;
    public SpongeCakeBSDF bsdf;

    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both, S : ISampleGenerator>(
        const SurfaceInteraction si,
        const float3 wo_ws,
        inout S sg
    )
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        float3 wo_ls = shading_frame_ws.to_local(wo_ws);
        if (!stdmat::is_valid_hemisphere_reflection_or_transmission(si, shading_frame_ws, wi_ls, wo_ls, wo_ws))
            return BSDFEval();
        BSDFEval result = {};
        if (is_set(Hints, BSDFEvalHints::eval))
            result.value = bsdf.eval(wi_ls, wo_ls);
        if (is_set(Hints, BSDFEvalHints::pdf))
            result.pdf = bsdf.pdf(wi_ls, wo_ls);
        return result;
    }

    public bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample)
    {
        sample = {};
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        BSDFSample bsdf_sample;
        if (!bsdf.sample(wi_ls, bsdf_sample, sg))
            return false;
        sample.pdf = bsdf_sample.pdf;
        sample.weight = bsdf_sample.weight;
        sample.flags = bsdf_sample.flags;
        sample.wo_ws = shading_frame_ws.to_global(bsdf_sample.wo);
        return stdmat::is_valid_hemisphere_reflection_or_transmission(
            si,
            shading_frame_ws,
            wi_ls,
            bsdf_sample.wo,
            sample.wo_ws
        );
    }

    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};
        result.material_color = saturate(bsdf.body.albedo);
        result.roughness = bsdf.body.alpha;
        result.guide_normal = shading_frame_ws.normal;
        result.diffuse_reflection_albedo = bsdf.body.albedo * (1.0 - bsdf.transmittance(float3(0.0, 0.0, 1.0)));
        result.specular_reflection_albedo = bsdf.top.albedo * (1.0 - exp(-bsdf.top.tau));
        result.diffuse_transmission_albedo = (1.0 - bsdf.base_weight) * result.diffuse_reflection_albedo;
        return result;
    }

    public BSDFFlags get_lobe_types(const SurfaceInteraction si) { return bsdf.lobe_types(); }
};

/// SpongeCake material: two layers (SGGX microflakes or Henyey-Greenstein media) and an optional Lambertian base (see spongecake_bsdf.slang and
/// src/falcor2/render/material/spongecake_material.h). Two-sided and thin-walled; both sides show the top layer.
public struct SpongeCakeMaterial : IMaterial
{
    typedef SpongeCakeMaterialInstance MaterialInstance;

    public MaterialHeader header;
    public BufferHandle tables;
    public TextureHandle albedo_texture;
    public TextureHandle top_albedo_texture;
    public TextureHandle base_color_texture;
    public TextureHandle normal_texture;
    public float3 albedo;
    public float thickness;
    public float3 top_albedo;
    public float top_thickness;
    public float3 base_color;
    public float base_weight;
    public float roughness;
    public float top_roughness;
    public float normal_texture_scale;
    /// Bit 0: body fibers, bit 1: top fibers, bit 2: delta transmission, bit 3: HG body, bit 4: HG top, bit 5: the
    /// tables have the forward multiple-scattering lobes.
    public uint layer_flags;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        MaterialInstance mi = {};
        if (normal_texture.is_valid() && stdmat::is_normal_mapping_enabled(hints)) {
            float3 normal
                = stdmat::decode_normal_map(normal_texture.sample(lod_sampler, si.uv, float4(0.0, 0.0, 1.0, 0.0)).xyz);
            normal = normalize(lerp(float3(0.0, 0.0, 1.0), normal, normal_texture_scale));
            mi.shading_frame_ws = Frame(
                si.shading_frame_ws.to_global(normal),
                si.shading_frame_ws.tangent,
                si.shading_frame_ws.handedness_sign()
            );
        } else {
            mi.shading_frame_ws = si.shading_frame_ws;
        }
        mi.shading_frame_ws = stdmat::orient_shading_frame(si, mi.shading_frame_ws, true);
        if (hints & MaterialInstanceHints::adjust_shading_normal)
            adjust_shading_normal(mi.shading_frame_ws, si.front_normal_ws(), si.wi_ws);

        SpongeCakeBSDF b = {};
        b.top.albedo = saturate(top_albedo * top_albedo_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb);
        b.top.tau = top_thickness;
        b.top.alpha = top_roughness;
        b.body.albedo = saturate(albedo * albedo_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb);
        b.body.tau = thickness;
        b.body.alpha = roughness;
        b.base_color = saturate(base_color * base_color_texture.sample(lod_sampler, si.uv, float4(1.0)).rgb);
        b.base_weight = base_weight;
        b.set_layer_flags(layer_flags);
        b.tables = tables;
        b.table_tangent = float4(0.0);
        b.table_derivatives = false;
        mi.bsdf = b;
        return mi;
    }

    public override MaterialFlags flags() { return header.flags; }
};
