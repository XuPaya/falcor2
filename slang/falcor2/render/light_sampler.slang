// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Interface for complete scene-light sampling strategies.
public interface ILightSampler
{
    /// Samples incident illumination from any active scene light source.
    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    );

    /// Evaluates the complete scene-light PDF for an environment direction.
    public float eval_environment_pdf(const LightSampleContext ctx, const float3 dir_ws);

    /// Evaluates the complete scene-light PDF for a known emissive triangle hit.
    public float eval_emissive_pdf(
        const LightSampleContext ctx,
        const EmissiveTriangleID triangle_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    );

    /// Evaluates the complete next-event PDF (solid angle at ctx.pos_ws, including selection) of a finite analytic
    /// light's sample at pos_ws with normal normal_ws (zero for point lights, whose position is a delta: the
    /// selection probability is returned).
    public float eval_analytic_pdf(
        const LightSampleContext ctx,
        const uint light_index,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    );

    /// Selects an emitter for emission (light tracing), independently of a receiving point: type and index
    /// (analytic: LightID; environment: environment light index; emissive triangle: EmissiveTriangleID) with
    /// probability pmf.
    public bool sample_emitter(const float u, out EmitterType type, out uint index, out float pmf);

    /// Probability of sample_emitter() selecting an emitter.
    public float eval_emitter_pmf(const EmitterType type, const uint index);
}

namespace detail {

/// Probability of selecting the finite-emitter domain when both domains have positive estimates.
static const float FINITE_EMITTER_SELECTION_PROBABILITY = 0.5;

/// Marginal probabilities of selecting each emitter type before conditional sampling.
struct EmitterTypePMF
{
    float analytic;
    float environment;
    float emissive_triangle;
};

EmitterTypePMF compute_emitter_type_pmf(
    const float analytic_light_power,
    const float environment_light_power,
    const float emissive_triangle_power
)
{
    EmitterTypePMF pmf = {};
    const bool has_finite_emitters = analytic_light_power > 0.0 || emissive_triangle_power > 0.0;
    const bool has_environment_lights = environment_light_power > 0.0;

    if (!has_finite_emitters) {
        pmf.environment = has_environment_lights ? 1.0 : 0.0;
        return pmf;
    }

    const float finite_probability = has_environment_lights ? FINITE_EMITTER_SELECTION_PROBABILITY : 1.0;
    const float finite_power_scale = max(analytic_light_power, emissive_triangle_power);
    const float analytic_power = analytic_light_power / finite_power_scale;
    const float emissive_power = emissive_triangle_power / finite_power_scale;
    const float normalized_finite_power = analytic_power + emissive_power;
    if (analytic_power <= emissive_power) {
        pmf.analytic = finite_probability * (analytic_power / normalized_finite_power);
        pmf.emissive_triangle = finite_probability - pmf.analytic;
    } else {
        pmf.emissive_triangle = finite_probability * (emissive_power / normalized_finite_power);
        pmf.analytic = finite_probability - pmf.emissive_triangle;
    }
    pmf.environment = has_environment_lights ? 1.0 - finite_probability : 0.0;
    return pmf;
}

/// Next-event PDF (solid angle at the receiver) of an analytic light's own position sampling, without selection.
float eval_analytic_light_pdf(const LightSampleContext ctx, const uint light_index, const float3 pos_ws,
                              const float3 normal_ws)
{
    const LightData data = g_scene.light_system.light_data[light_index];
    const LightType type = LightType(reinterpret<uint>(data.header));
    if (type == LightType::point_light)
        return 1.0;
    float area = 0.0;
    if (type == LightType::disk_light)
        area = reinterpret<DiskLight>(data).surface_area;
    else if (type == LightType::rect_light)
        area = reinterpret<RectLight>(data).surface_area;
    else if (type == LightType::sphere_light) {
        const SphereLight sphere = reinterpret<SphereLight>(data);
        area = sphere.surface_area;
        if (sphere.enable_virtual_sphere_shrinking) {
            const float d2 = dot(ctx.pos_ws - sphere.center_ws, ctx.pos_ws - sphere.center_ws);
            const float r2 = sphere.radius_ws * sphere.radius_ws;
            area *= min(r2, 0.99 * d2) / max(r2, 1e-30);
        }
    } else
        return 0.0;
    const float3 to_light = pos_ws - ctx.pos_ws;
    const float d2 = max(dot(to_light, to_light), 1.0e-9);
    const float cos_on_light = dot(normal_ws, -to_light * rsqrt(d2));
    if (!(cos_on_light > 0.0) || !(area > 0.0))
        return 0.0;
    return d2 / (cos_on_light * area);
}

} // namespace detail

/// Samples every scene light source with uniform probability.
public struct UniformLightSampler : ILightSampler
{
    uint get_source_count()
    {
        return g_scene.light_system.light_count + g_scene.emissive_geometry.active_triangle_count;
    }

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        sample = {};

        const uint source_count = get_source_count();
        if (source_count == 0)
            return false;

        const uint analytic_light_count = g_scene.light_system.analytic_light_count;
        const uint environment_light_count = g_scene.light_system.environment_light_count;

        const uint source_index = min(uint(next_float(sg) * source_count), source_count - 1);
        const float selection_pdf = 1.0 / float(source_count);

        if (source_index < analytic_light_count) {
            // Sample an analytic light.
            const LightID light_id = LightID(source_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, sg, sample))
                return false;
            sample.emitter_id = uint(light_id);
        } else if (source_index < analytic_light_count + environment_light_count) {
            // Sample an environment light.
            const uint environment_light_index = source_index - analytic_light_count;
            const LightID light_id = LightID(uint(g_scene.light_system.environment_light_id) + environment_light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, sg, sample))
                return false;
            sample.emitter_id = uint(light_id);
            if (environment_light_count > 1) {
                sample.pdf = eval_environment_mixture_pdf(ctx, sample.dir_ws);
                if (sample.pdf <= 0.0)
                    return false;
                sample.weight = g_scene.light_system.eval_environment_radiance(ctx, sample.dir_ws) / sample.pdf;
            }
        } else {
            // Sample an emissive triangle.
            const uint active_triangle_index = source_index - analytic_light_count - environment_light_count;
            const EmissiveTriangleID triangle_id
                = g_scene.emissive_geometry.active_triangle_id_to_triangle_id[active_triangle_index];
            if (!sample_emissive_triangle(ctx.pos_ws, triangle_id, next_float2(sg), sample))
                return false;
        }

        // Adjust the sample PDF and weight to account for the uniform selection probability.
        sample.pdf *= selection_pdf;
        sample.weight /= selection_pdf;

        return true;
    }

    /// Evaluates the selection PDF for a source known to be valid.
    public float eval_source_selection_pdf() { return 1.0 / float(get_source_count()); }

    public float eval_analytic_pdf(
        const LightSampleContext ctx,
        const uint light_index,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (light_index >= g_scene.light_system.analytic_light_count)
            return 0.0;
        return eval_source_selection_pdf()
            * detail::eval_analytic_light_pdf(ctx, light_index, emitter_pos_ws, emitter_normal_ws);
    }

    public bool sample_emitter(const float u, out EmitterType type, out uint index, out float pmf)
    {
        type = EmitterType::unknown;
        index = 0;
        pmf = 0.0;
        const uint source_count = get_source_count();
        if (source_count == 0)
            return false;
        const uint analytic_light_count = g_scene.light_system.analytic_light_count;
        const uint environment_light_count = g_scene.light_system.environment_light_count;
        const uint source_index = min(uint(u * source_count), source_count - 1);
        pmf = 1.0 / float(source_count);
        if (source_index < analytic_light_count) {
            type = EmitterType::analytic;
            index = source_index;
        } else if (source_index < analytic_light_count + environment_light_count) {
            type = EmitterType::environment;
            index = source_index - analytic_light_count;
        } else {
            type = EmitterType::emissive_triangle;
            index = uint(g_scene.emissive_geometry.active_triangle_id_to_triangle_id[
                source_index - analytic_light_count - environment_light_count]);
        }
        return true;
    }

    public float eval_emitter_pmf(const EmitterType type, const uint index)
    {
        const uint source_count = get_source_count();
        if (source_count == 0)
            return 0.0;
        if (type == EmitterType::emissive_triangle) {
            if (index >= g_scene.emissive_geometry.triangle_count
                || g_scene.emissive_geometry.triangle_id_to_active_triangle_id[index] == ActiveEmissiveTriangleID::invalid)
                return 0.0;
        }
        return 1.0 / float(source_count);
    }

    float eval_environment_mixture_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        float pdf = 0.0;
        for (uint i = 0; i < g_scene.light_system.environment_light_count; ++i)
            pdf += g_scene.light_system.get_environment_light(i).eval_pdf(ctx, dir_ws);
        return pdf;
    }

    public float eval_environment_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        if (g_scene.light_system.environment_light_count == 0)
            return 0.0;
        return eval_source_selection_pdf() * eval_environment_mixture_pdf(ctx, dir_ws);
    }

    public float eval_emissive_pdf(
        const LightSampleContext ctx,
        const EmissiveTriangleID triangle_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (uint(triangle_id) >= g_scene.emissive_geometry.triangle_count)
            return 0.0;
        const ActiveEmissiveTriangleID active_triangle_id
            = g_scene.emissive_geometry.triangle_id_to_active_triangle_id[uint(triangle_id)];
        if (active_triangle_id == ActiveEmissiveTriangleID::invalid)
            return 0.0;
        return eval_source_selection_pdf()
            * eval_emissive_triangle_pdf(ctx, triangle_id, emitter_pos_ws, emitter_normal_ws);
    }
}

/// Samples finite and infinite emitters with configured probability, then power-weights within each domain.
public struct PowerLightSampler : ILightSampler
{
    public AliasTable1D analytic_light_distribution;
    public AliasTable1D environment_light_distribution;
    public AliasTable1D emissive_triangle_distribution;
    /// Luminance-weighted emitted radiant flux of all analytic lights.
    public float analytic_light_power;
    /// Luminance-weighted integral of incident radiance over solid angle for all infinite lights.
    public float environment_light_power;
    /// Luminance-weighted emitted radiant flux of all emissive triangles.
    public float emissive_triangle_power;

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        sample = {};
        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        if (emitter_type_pmf.analytic <= 0.0 && emitter_type_pmf.environment <= 0.0
            && emitter_type_pmf.emissive_triangle <= 0.0)
            return false;

        float light_sample = next_float(sg);
        float selection_pdf = 0.0;
        if (light_sample < emitter_type_pmf.analytic) {
            // Sample an analytic light.
            light_sample /= emitter_type_pmf.analytic;
            uint light_index;
            if (!analytic_light_distribution.sample(light_sample, light_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.analytic;
            if (selection_pdf <= 0.0)
                return false;

            const LightID light_id = LightID(light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, sg, sample))
                return false;
            sample.emitter_id = uint(light_id);
        } else if (light_sample < 1.0 - emitter_type_pmf.environment) {
            // Sample an emissive triangle.
            light_sample = (light_sample - emitter_type_pmf.analytic) / emitter_type_pmf.emissive_triangle;
            uint active_triangle_index;
            if (!emissive_triangle_distribution.sample(light_sample, active_triangle_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.emissive_triangle;
            if (selection_pdf <= 0.0)
                return false;

            const EmissiveTriangleID triangle_id
                = g_scene.emissive_geometry.active_triangle_id_to_triangle_id[active_triangle_index];
            if (!sample_emissive_triangle(ctx.pos_ws, triangle_id, next_float2(sg), sample))
                return false;
        } else {
            // Sample an environment light.
            light_sample = (light_sample - (1.0 - emitter_type_pmf.environment)) / emitter_type_pmf.environment;
            uint environment_light_index;
            if (!environment_light_distribution.sample(light_sample, environment_light_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.environment;
            if (selection_pdf <= 0.0)
                return false;

            const LightID light_id = LightID(uint(g_scene.light_system.environment_light_id) + environment_light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, sg, sample))
                return false;
            sample.emitter_id = uint(light_id);
            if (g_scene.light_system.environment_light_count > 1) {
                sample.pdf = eval_environment_mixture_pdf(ctx, sample.dir_ws);
                if (sample.pdf <= 0.0)
                    return false;
                sample.weight = g_scene.light_system.eval_environment_radiance(ctx, sample.dir_ws) / sample.pdf;
                selection_pdf = emitter_type_pmf.environment;
            }
        }

        // Adjust the conditional sample PDF and weight for the applicable selection probability.
        sample.pdf *= selection_pdf;
        sample.weight /= selection_pdf;

        return true;
    }

    public float eval_source_selection_pdf(const EmitterType type, const uint index)
    {
        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        if (type == EmitterType::analytic && emitter_type_pmf.analytic > 0.0)
            return emitter_type_pmf.analytic * analytic_light_distribution.eval_pdf(index);
        if (type == EmitterType::environment && emitter_type_pmf.environment > 0.0)
            return emitter_type_pmf.environment * environment_light_distribution.eval_pdf(index);
        if (type == EmitterType::emissive_triangle && emitter_type_pmf.emissive_triangle > 0.0)
            return emitter_type_pmf.emissive_triangle * emissive_triangle_distribution.eval_pdf(index);
        return 0.0;
    }

    public float eval_analytic_pdf(
        const LightSampleContext ctx,
        const uint light_index,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (light_index >= g_scene.light_system.analytic_light_count)
            return 0.0;
        return eval_source_selection_pdf(EmitterType::analytic, light_index)
            * detail::eval_analytic_light_pdf(ctx, light_index, emitter_pos_ws, emitter_normal_ws);
    }

    public bool sample_emitter(const float u, out EmitterType type, out uint index, out float pmf)
    {
        type = EmitterType::unknown;
        index = 0;
        pmf = 0.0;
        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        float v = u;
        float selection = 0.0;
        if (v < emitter_type_pmf.analytic) {
            if (!analytic_light_distribution.sample(v / emitter_type_pmf.analytic, index, selection))
                return false;
            type = EmitterType::analytic;
            pmf = emitter_type_pmf.analytic * selection;
        } else if (v < 1.0 - emitter_type_pmf.environment) {
            uint active_index;
            if (!emissive_triangle_distribution.sample((v - emitter_type_pmf.analytic) / emitter_type_pmf.emissive_triangle,
                                                       active_index, selection))
                return false;
            type = EmitterType::emissive_triangle;
            index = uint(g_scene.emissive_geometry.active_triangle_id_to_triangle_id[active_index]);
            pmf = emitter_type_pmf.emissive_triangle * selection;
        } else {
            if (!environment_light_distribution.sample(
                    (v - (1.0 - emitter_type_pmf.environment)) / emitter_type_pmf.environment, index, selection))
                return false;
            type = EmitterType::environment;
            pmf = emitter_type_pmf.environment * selection;
        }
        return pmf > 0.0;
    }

    public float eval_emitter_pmf(const EmitterType type, const uint index)
    {
        if (type != EmitterType::emissive_triangle)
            return eval_source_selection_pdf(type, index);
        if (index >= g_scene.emissive_geometry.triangle_count)
            return 0.0;
        const ActiveEmissiveTriangleID active = g_scene.emissive_geometry.triangle_id_to_active_triangle_id[index];
        if (active == ActiveEmissiveTriangleID::invalid)
            return 0.0;
        return eval_source_selection_pdf(type, uint(active));
    }

    float eval_environment_mixture_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        float pdf = 0.0;
        for (uint i = 0; i < g_scene.light_system.environment_light_count; ++i)
            pdf += environment_light_distribution.eval_pdf(i)
                * g_scene.light_system.get_environment_light(i).eval_pdf(ctx, dir_ws);
        return pdf;
    }

    public float eval_environment_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        const float environment_probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .environment;
        if (environment_probability <= 0.0)
            return 0.0;
        return environment_probability * eval_environment_mixture_pdf(ctx, dir_ws);
    }

    public float eval_emissive_pdf(
        const LightSampleContext ctx,
        const EmissiveTriangleID triangle_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        if (uint(triangle_id) >= g_scene.emissive_geometry.triangle_count)
            return 0.0;
        const ActiveEmissiveTriangleID active_triangle_id
            = g_scene.emissive_geometry.triangle_id_to_active_triangle_id[uint(triangle_id)];
        if (active_triangle_id == ActiveEmissiveTriangleID::invalid)
            return 0.0;
        return eval_source_selection_pdf(EmitterType::emissive_triangle, uint(active_triangle_id))
            * eval_emissive_triangle_pdf(ctx, triangle_id, emitter_pos_ws, emitter_normal_ws);
    }
}

/// Uses power sampling for component lights and an EmissiveTriangleTree for emissive triangles.
public struct HierarchicalLightSampler : ILightSampler
{
    public AliasTable1D analytic_light_distribution;
    public AliasTable1D environment_light_distribution;
    public float analytic_light_power;
    public float environment_light_power;
    public float emissive_triangle_power;
    public EmissiveTriangleTree emissive_triangle_tree;

    public bool sample<SampleGenerator : ISampleGenerator>(
        const LightSampleContext ctx,
        inout SampleGenerator sg,
        out LightSample sample
    )
    {
        sample = {};
        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        if (emitter_type_pmf.analytic <= 0.0 && emitter_type_pmf.environment <= 0.0
            && emitter_type_pmf.emissive_triangle <= 0.0)
            return false;

        float light_sample = next_float(sg);
        float selection_pdf = 0.0;
        if (light_sample < emitter_type_pmf.analytic) {
            light_sample /= emitter_type_pmf.analytic;
            uint light_index;
            if (!analytic_light_distribution.sample(light_sample, light_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.analytic;
            if (selection_pdf <= 0.0)
                return false;

            const LightID light_id = LightID(light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, sg, sample))
                return false;
            sample.emitter_id = uint(light_id);
        } else if (light_sample < 1.0 - emitter_type_pmf.environment) {
            light_sample = (light_sample - emitter_type_pmf.analytic) / emitter_type_pmf.emissive_triangle;
            EmissiveTriangleID triangle_id;
            if (!emissive_triangle_tree.sample(ctx, light_sample, selection_pdf, triangle_id))
                return false;
            selection_pdf *= emitter_type_pmf.emissive_triangle;
            if (selection_pdf <= 0.0 || !sample_emissive_triangle(ctx.pos_ws, triangle_id, next_float2(sg), sample))
                return false;
        } else {
            light_sample = (light_sample - (1.0 - emitter_type_pmf.environment)) / emitter_type_pmf.environment;
            uint environment_light_index;
            if (!environment_light_distribution.sample(light_sample, environment_light_index, selection_pdf))
                return false;
            selection_pdf *= emitter_type_pmf.environment;
            if (selection_pdf <= 0.0)
                return false;

            const LightID light_id = LightID(uint(g_scene.light_system.environment_light_id) + environment_light_index);
            if (!g_scene.light_system.get_light(light_id).sample(ctx, sg, sample))
                return false;
            sample.emitter_id = uint(light_id);
            if (g_scene.light_system.environment_light_count > 1) {
                sample.pdf = eval_environment_mixture_pdf(ctx, sample.dir_ws);
                if (sample.pdf <= 0.0)
                    return false;
                sample.weight = g_scene.light_system.eval_environment_radiance(ctx, sample.dir_ws) / sample.pdf;
                selection_pdf = emitter_type_pmf.environment;
            }
        }

        sample.pdf *= selection_pdf;
        sample.weight /= selection_pdf;
        return true;
    }

    float eval_environment_mixture_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        float pdf = 0.0;
        for (uint i = 0; i < g_scene.light_system.environment_light_count; ++i)
            pdf += environment_light_distribution.eval_pdf(i)
                * g_scene.light_system.get_environment_light(i).eval_pdf(ctx, dir_ws);
        return pdf;
    }

    public float eval_environment_pdf(const LightSampleContext ctx, const float3 dir_ws)
    {
        const float environment_probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .environment;
        if (environment_probability <= 0.0)
            return 0.0;
        return environment_probability * eval_environment_mixture_pdf(ctx, dir_ws);
    }

    public float eval_emissive_pdf(
        const LightSampleContext ctx,
        const EmissiveTriangleID triangle_id,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        const float emissive_probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .emissive_triangle;
        if (emissive_probability <= 0.0)
            return 0.0;
        const float triangle_selection_pdf = emissive_triangle_tree.eval_pdf(ctx, triangle_id);
        if (triangle_selection_pdf <= 0.0)
            return 0.0;
        return emissive_probability * triangle_selection_pdf
            * eval_emissive_triangle_pdf(ctx, triangle_id, emitter_pos_ws, emitter_normal_ws);
    }

    public float eval_analytic_pdf(
        const LightSampleContext ctx,
        const uint light_index,
        const float3 emitter_pos_ws,
        const float3 emitter_normal_ws
    )
    {
        const float analytic_probability
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power)
                  .analytic;
        if (light_index >= g_scene.light_system.analytic_light_count || analytic_probability <= 0.0)
            return 0.0;
        return analytic_probability * analytic_light_distribution.eval_pdf(light_index)
            * detail::eval_analytic_light_pdf(ctx, light_index, emitter_pos_ws, emitter_normal_ws);
    }

    /// Emission selection: component lights by power as for next-event estimation; emissive triangles uniformly
    /// among the active ones (the light tree's importance needs a receiving point).
    public bool sample_emitter(const float u, out EmitterType type, out uint index, out float pmf)
    {
        type = EmitterType::unknown;
        index = 0;
        pmf = 0.0;
        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        float selection = 0.0;
        if (u < emitter_type_pmf.analytic) {
            if (!analytic_light_distribution.sample(u / emitter_type_pmf.analytic, index, selection))
                return false;
            type = EmitterType::analytic;
            pmf = emitter_type_pmf.analytic * selection;
        } else if (u < 1.0 - emitter_type_pmf.environment) {
            const uint count = g_scene.emissive_geometry.active_triangle_count;
            if (count == 0)
                return false;
            const float v = (u - emitter_type_pmf.analytic) / emitter_type_pmf.emissive_triangle;
            type = EmitterType::emissive_triangle;
            index = uint(g_scene.emissive_geometry.active_triangle_id_to_triangle_id[min(uint(v * count), count - 1)]);
            pmf = emitter_type_pmf.emissive_triangle / float(count);
        } else {
            if (!environment_light_distribution.sample(
                    (u - (1.0 - emitter_type_pmf.environment)) / emitter_type_pmf.environment, index, selection))
                return false;
            type = EmitterType::environment;
            pmf = emitter_type_pmf.environment * selection;
        }
        return pmf > 0.0;
    }

    public float eval_emitter_pmf(const EmitterType type, const uint index)
    {
        const detail::EmitterTypePMF emitter_type_pmf
            = detail::compute_emitter_type_pmf(analytic_light_power, environment_light_power, emissive_triangle_power);
        if (type == EmitterType::analytic)
            return emitter_type_pmf.analytic * analytic_light_distribution.eval_pdf(index);
        if (type == EmitterType::environment)
            return emitter_type_pmf.environment * environment_light_distribution.eval_pdf(index);
        if (type == EmitterType::emissive_triangle) {
            const uint count = g_scene.emissive_geometry.active_triangle_count;
            if (count == 0 || index >= g_scene.emissive_geometry.triangle_count
                || g_scene.emissive_geometry.triangle_id_to_active_triangle_id[index] == ActiveEmissiveTriangleID::invalid)
                return 0.0;
            return emitter_type_pmf.emissive_triangle / float(count);
        }
        return 0.0;
    }
}
