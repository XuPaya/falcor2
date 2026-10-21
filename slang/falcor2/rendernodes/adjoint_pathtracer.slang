// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Adjoint (light-tracing) companion of the reference path tracer.
//
// Light paths start at the scene's lights - an emitter chosen by the light sampler's emission selection, then a
// position and direction (render/light_emission.slang: analytic lights, emissive triangles and environment lights)
// - are traced by adjoint BSDF sampling and connected to the pinhole camera at every surface vertex (the t = 1
// strategy of bidirectional path tracing, Veach 1997, ch. 10). Contributions are splatted into a float buffer with
// atomics; resolve() adds them to the path tracer's frame.
//
// Combination with the path tracer: multiple importance sampling over the strategies that can produce a path,
// emission found by a camera path (s = 0, emitters that rays can hit: emissive triangles, environment), next-event
// estimation (s = 1) and light tracing (t = 1), with the balance heuristic. Each tracer evaluates the other
// strategies' densities of its own paths; the partial ratios are accumulated along the subpaths (as the dVC
// quantities of Georgiev 2012, "Implementing Vertex Connection and Merging", restricted to these strategies). A
// strategy that cannot produce a path (a delta lobe at the camera end, a direction outside the emission target, a
// scattering-medium event) has density zero, so its weight vanishes.
//
// The light-tracing integrand is the path tracer's: camera connections evaluate the BSDF with a material instance set
// up exactly as the path tracer would (wi = camera-side direction), so non-reciprocal BSDFs and shading-normal
// adjustments match.

module adjoint_pathtracer;

__exported import falcor2.render;
import falcor2.utils;

import slangpy;

/// Track closed-surface interiors (mirrors the path tracer's constant).
extern static const bool ENABLE_INTERIOR_TRACKING = true;
/// Homogeneous media in tracked interiors (mirrors the path tracer's constant).
extern static const bool ENABLE_HOMOGENEOUS_MEDIA = true;
/// Russian roulette on the per-bounce scattering weight.
extern static const bool ENABLE_RUSSIAN_ROULETTE = true;

public extern struct LightSampler : ILightSampler = PowerLightSampler;
public ParameterBlock<LightSampler> light_sampler;

/// The bound light sampler as a value (for the generic emission functions of render/light_emission.slang).
LightSampler scene_light_sampler() { return light_sampler; }

static const bool HANDLE_HOMOGENEOUS_MEDIA = ENABLE_INTERIOR_TRACKING && ENABLE_HOMOGENEOUS_MEDIA;
static const float RUSSIAN_ROULETTE_MAX_SURVIVAL = 0.95;

public enum class LightTracerMode : uint {
    /// Multiple importance sampling with the path tracer (combined image).
    mis = 0,
    /// The light tracer alone (weight one; validation, the path tracer must not add its image).
    all = 1,
};

public struct LightTracerSettings
{
    /// Maximum path depth with the path tracer's meaning: paths with up to max_depth - 1 surface vertices.
    uint max_depth = 3;
    /// Light-path depth at which Russian roulette starts.
    uint rr_depth = 3;
    /// LightTracerMode.
    uint mode = 0;
    /// Number of light paths traced this frame.
    uint light_path_count = 0;
    /// Scale of the splatted contributions (1 / light_path_count).
    float contribution_scale = 0.0;
    /// Light paths per frame over camera paths per pixel and frame (the MIS sample-count ratio).
    float mis_path_count = 0.0;
    LightEmissionSettings emission = {};
    /// Frame index (sample generator).
    uint iteration = 0;
    /// Camera depth range (rays from the pinhole ignore hits beyond this, like the path tracer's rays).
    float camera_ray_t_max = 1000.0;
};

public struct LightTracer
{
    LightTracerSettings settings;
    Camera camera;
    /// RGB float accumulation, 3 floats per pixel (row-major, y down).
    RWByteAddressBuffer accumulation;
};

public ParameterBlock<LightTracer> g_light_tracer;

/// Ray-sphere intersection distances (t0 <= t1); false if the line misses the sphere.
bool intersect_sphere(const float3 origin, const float3 dir, const float3 center, const float radius, out float t0, out float t1)
{
    const float3 oc = origin - center;
    const float b = dot(oc, dir);
    const float c = dot(oc, oc) - radius * radius;
    const float disc = b * b - c;
    t0 = 0.0;
    t1 = 0.0;
    if (disc < 0.0)
        return false;
    const float s = sqrt(disc);
    t0 = -b - s;
    t1 = -b + s;
    return true;
}

//------------------------------------------
// Path-tracer mirrors (keep in sync with reference_pathtracer.slang)

enum class HemisphereFlags {
    none = 0,
    upper = (1 << 0),
    lower = (1 << 1),
    both = upper | lower,
};

HemisphereFlags get_material_nee_hemispheres(const BSDFFlags lobe_types)
{
    HemisphereFlags result = HemisphereFlags::none;
    if (is_set(lobe_types, BSDFFlags::non_delta_reflection))
        result = result | HemisphereFlags::upper;
    if (is_set(lobe_types, BSDFFlags::non_delta_transmission))
        result = result | HemisphereFlags::lower;
    if (is_set(lobe_types, BSDFFlags::curve))
        result = result | HemisphereFlags::both;
    return result;
}

HemisphereFlags get_nee_visibility_supported_hemispheres(
    const bool is_entering,
    const MaterialID material_id,
    const bool is_thin_walled,
    const InteriorList interiors
)
{
    if (!HANDLE_HOMOGENEOUS_MEDIA)
        return HemisphereFlags::both;
    if (is_thin_walled)
        return interiors.is_empty() ? HemisphereFlags::both : HemisphereFlags::none;
    if (is_entering)
        return interiors.is_empty() ? HemisphereFlags::upper : HemisphereFlags::none;
    if (interiors.is_empty())
        return HemisphereFlags::upper;
    return interiors.contains_only(material_id) ? HemisphereFlags::lower : HemisphereFlags::none;
}

InteriorListUpdateResult cross_interior_boundary(
    const bool is_entering,
    const MaterialID material_id,
    const uint nested_priority,
    IMaterialInstance material_instance,
    inout InteriorList interior_list
)
{
    if (is_entering) {
        HomogeneousMediumProperties interior_medium = {};
        if (HANDLE_HOMOGENEOUS_MEDIA)
            interior_medium = material_instance.get_interior_medium();
        let entry = interior_list.make_entry(material_id, nested_priority, material_instance.get_interior_ior(), interior_medium);
        return interior_list.enter(entry);
    } else {
        return interior_list.exit(material_id);
    }
}

//------------------------------------------
// Light paths

/// A light-path surface vertex.
struct LightVertex
{
    PackedHitInfo hit;
    float3 pos_ws;
    /// Geometric normal (unoriented).
    float3 normal_ws;
    /// Interpolated shading normal (unoriented), for the adjoint correction.
    float3 shading_normal_ws;
    /// Direction toward the previous (light-side) vertex.
    float3 to_light_ws;
    /// True if the light arrives on the front (exterior) side.
    bool light_side_front;
    MaterialID material_id;
    MaterialHeader header;
};

/// MIS state of a light path: the other strategies' densities over the light tracer's (times the light-path count),
/// partial up to the current vertex. nee: next-event estimation at the vertex next to the light (s = 1); hit: a camera
/// path's BSDF sample hitting the emitter (s = 0). Both lack the camera-side direction density at the current vertex
/// (its camera-side incoming direction is not known yet); nee also lacks it nowhere at the first vertex (next-event
/// estimation does not sample the direction there).
struct LightPathMIS
{
    float nee;
    float hit;
};

/// Path-tracer view of a vertex: the interaction with wi = camera-side direction (the material instance is set
/// up from it exactly as the path tracer does).
SurfaceInteraction make_camera_side_interaction(const LightVertex v, const float3 wi_ws, const InteriorList camera_side_interiors)
{
    SurfaceInteraction si = g_scene.make_surface_interaction(v.hit.unpack(), wi_ws);
    if (ENABLE_INTERIOR_TRACKING)
        si.exterior_ior = camera_side_interiors.exterior_ior(v.material_id, si.front_facing);
    return si;
}

/// Whether the path tracer's next-event estimation at the first vertex (camera side cam_si) samples the light's
/// direction (its hemisphere rules), i.e. whether the s = 1 strategy exists.
bool camera_nee_possible(const LightVertex v, const SurfaceInteraction cam_si, const BSDFFlags lobe_types,
                         const InteriorList camera_side_interiors)
{
    const bool is_thin_walled = is_set(v.header.flags, MaterialFlags::thin_walled);
    const HemisphereFlags hemispheres = get_material_nee_hemispheres(lobe_types)
        & get_nee_visibility_supported_hemispheres(cam_si.front_facing, v.material_id, is_thin_walled, camera_side_interiors);
    const bool upper = dot(v.to_light_ws, cam_si.front_normal_ws()) >= 0.0;
    return upper ? (hemispheres & HemisphereFlags::upper) != HemisphereFlags::none
                 : (hemispheres & HemisphereFlags::lower) != HemisphereFlags::none;
}

void splat(const uint2 pixel, const float3 value)
{
    const uint2 dims = g_light_tracer.camera.dims;
    const uint address = (pixel.y * dims.x + pixel.x) * 12;
    g_light_tracer.accumulation.InterlockedAddF32(address + 0, value.x);
    g_light_tracer.accumulation.InterlockedAddF32(address + 4, value.y);
    g_light_tracer.accumulation.InterlockedAddF32(address + 8, value.z);
}

/// Connects a light-path vertex to the camera and splats w * beta * f * G * We.
void connect_to_camera(
    const LightVertex v,
    const float3 beta,
    const uint depth,
    const InteriorList interiors,
    const LightPathMIS mis,
    inout UniformSampleGenerator sg
)
{
    const LightTracerSettings s = g_light_tracer.settings;
    const Camera camera = g_light_tracer.camera;

    const float3 to_camera = camera.position - v.pos_ws;
    const float dist2 = dot(to_camera, to_camera);
    if (!(dist2 > 0.0))
        return;
    const float dist = sqrt(dist2);
    const float3 w_camera = to_camera / dist;

    // Pixel and pinhole importance.
    const float image_w_len = length(camera.image_w);
    const float cos_camera = dot(-w_camera, camera.image_w) / image_w_len;
    if (!(cos_camera > 0.0))
        return;
    const float2 pixel_pos = camera.project_world_to_pixel(v.pos_ws);
    if (!(all(pixel_pos >= 0.0) && all(pixel_pos < float2(camera.dims))))
        return;
    const uint2 pixel = min(uint2(pixel_pos), camera.dims - 1);
    if (dist > s.camera_ray_t_max)
        return;

    // One-sided materials are invisible from behind (the path tracer terminates there).
    const float cos_geom_camera = dot(w_camera, v.normal_ws);
    if (!is_set(v.header.flags, MaterialFlags::two_sided) && cos_geom_camera < 0.0)
        return;
    const float cos_geom_light = abs(dot(v.to_light_ws, v.normal_ws));
    if (!(cos_geom_light > 0.0))
        return;

    // The camera side of the vertex is in the medium the light arrives from (reflection) or past the
    // interface (transmission through a closed surface). The camera is assumed to be outside all objects.
    InteriorList camera_side_interiors = interiors;
    const bool transmits = dot(w_camera, v.normal_ws) * dot(v.to_light_ws, v.normal_ws) < 0.0;
    const bool is_thin_walled = is_set(v.header.flags, MaterialFlags::thin_walled);
    if (ENABLE_INTERIOR_TRACKING && transmits && !is_thin_walled) {
        if (v.light_side_front || camera_side_interiors.exit(v.material_id) != InteriorListUpdateResult::success)
            return;
    }

    const IHitInfo hit = v.hit.unpack();
    const SurfaceInteraction cam_si = make_camera_side_interaction(v, w_camera, camera_side_interiors);
    let cam_mi = g_scene.get_material_instance(hit, cam_si, ExplicitLodSampler(0), MaterialInstanceHints::adjust_shading_normal);
    const BSDFEval eval = cam_mi.eval_with_pdf(cam_si, v.to_light_ws, sg);
    if (!any(eval.value > 0.0))
        return;

    // Pixel importance: d_f^2 / (A_pixel cos^3) per solid angle, converted to the area at the vertex.
    const float pixel_area = 4.0 * length(camera.image_u) * length(camera.image_v) / float(camera.dims.x * camera.dims.y);
    const float importance = sqr(image_w_len) / (pixel_area * cos_camera * cos_camera * cos_camera * dist2);

    float weight = 1.0;
    if (s.mode == uint(LightTracerMode::mis)) {
        // The camera path's density of this vertex (area measure) and its directional density towards the previous
        // light vertex complete the other strategies' ratios.
        const float camera_area_pdf = importance * abs(cos_geom_camera);
        float nee = mis.nee * (depth >= 2 ? eval.pdf : 1.0);
        if (depth == 1 && !camera_nee_possible(v, cam_si, cam_mi.get_lobe_types(cam_si), camera_side_interiors))
            nee = 0.0;
        const float ratio = camera_area_pdf * (nee + mis.hit * eval.pdf);
        weight = 1.0 / (1.0 + ratio);
    }

    // Visibility from the camera, traced like the path tracer's primary ray (material opacity).
    float3 target = offset_ray_origin(v.pos_ws, cos_geom_camera >= 0.0 ? v.normal_ws : -v.normal_ws);
    float3 to_target = target - camera.position;
    const float t_max = length(to_target);
    if (!(t_max > 0.0))
        return;
    Ray camera_ray = Ray(camera.position, to_target / t_max, 0.0, t_max);
    OpacityEvaluator opacity_evaluator = OpacityEvaluator(sg.next_uint());
    // The vertex itself must be accepted by a camera ray (cut-outs).
    if (!opacity_evaluator.is_valid_hit(camera_ray, hit, MaterialInstanceHints::adjust_shading_normal))
        return;
    RayQuerySceneIntersector intersector = RayQuerySceneIntersector(opacity_evaluator);
    if (intersector.is_occluded(camera_ray, MaterialInstanceHints::adjust_shading_normal))
        return;

    const float3 contribution = beta * eval.value * (abs(cos_geom_camera) / cos_geom_light) * importance * weight
        * s.contribution_scale;
    if (all(isfinite(contribution)))
        splat(pixel, contribution);
}

/// Largest sampling pdf (1/sr) of a non-delta lobe weighted with the path tracer's own BSDF evaluation.
static const float EXACT_ADJOINT_MAX_PDF = 100.0;

/// Veach's adjoint factor for shading normals, |wi.ns| |wo.ng| / (|wi.ng| |wo.ns|), with the interpolated normal.
float adjoint_shading_correction(const LightVertex v, const float3 wo, const float cos_geom_in, const float cos_geom_out)
{
    const float cos_shading_in = abs(dot(v.to_light_ws, v.shading_normal_ws));
    const float cos_shading_out = abs(dot(wo, v.shading_normal_ws));
    if (!(cos_shading_out > 0.0))
        return 0.0;
    return (cos_shading_in * abs(cos_geom_out)) / (cos_geom_in * cos_shading_out);
}

/// Traces one light path per thread.
public void trace_light_paths(uint tid)
{
    const LightTracerSettings s = g_light_tracer.settings;
    if (tid >= s.light_path_count)
        return;
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid, 0x40000000u), s.iteration);

    EmissionSample e;
    if (!sample_light_emission(scene_light_sampler(), s.emission, sg, e))
        return;
    float3 beta = e.weight;
    const float emission_density = e.pdf_pos * e.pdf_dir;
    const bool point_light = e.type == EmitterType::analytic && all(e.normal_ws == 0.0);
    const bool hittable = e.type != EmitterType::analytic;

    // Start far light rays at the scene's bounding sphere (nothing lies outside it).
    float t_enter, t_exit;
    if (!intersect_sphere(e.pos_ws, e.dir_ws, s.emission.scene_center, s.emission.scene_radius, t_enter, t_exit)
        || !(t_exit > 0.0))
        return;
    // Rays from emissive triangles leave their surface (offset origin, no self-intersection).
    const float3 start = e.type == EmitterType::emissive_triangle ? offset_ray_origin(e.pos_ws, e.normal_ws)
                                                                  : e.pos_ws + e.dir_ws * max(t_enter, 0.0);
    Ray ray = Ray(start, e.dir_ws, 0.0, t_exit - max(t_enter, 0.0) + 1.0);

    InteriorList interiors = {};
    uint depth = 0;
    const uint max_vertices = s.max_depth > 1 ? s.max_depth - 1 : 0;
    LightPathMIS mis = {};

    while (depth < max_vertices) {
        OpacityEvaluator opacity_evaluator = OpacityEvaluator(sg.next_uint());
        RayQuerySceneIntersector intersector = RayQuerySceneIntersector(opacity_evaluator);
        const IHitInfo hit = intersector.intersect(ray, MaterialInstanceHints::adjust_shading_normal);
        if (!hit.valid)
            break;

        LightVertex v;
        v.hit = hit.pack();
        SurfaceInteraction si = g_scene.make_surface_interaction(hit, -ray.dir);
        v.pos_ws = si.position_ws;
        v.normal_ws = si.normal_ws;
        v.shading_normal_ws = si.shading_frame_ws.normal;
        v.to_light_ws = -ray.dir;
        v.light_side_front = si.front_facing;
        v.material_id = g_scene.get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        v.header = g_scene.material_system.get_material_data(v.material_id).header;

        // Media along the segment: absorption only (scattering media are left to the path tracer).
        if (HANDLE_HOMOGENEOUS_MEDIA && !interiors.is_empty()) {
            const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(interiors.active_medium());
            if (sampler.has_scattering())
                break;
            beta *= sampler.eval_transmittance(length(v.pos_ws - ray.origin));
        }

        if (is_set(v.header.flags, MaterialFlags::unlit))
            break;
        depth += 1;

        const float cos_geom_in = abs(dot(v.to_light_ws, v.normal_ws));
        if (!(cos_geom_in > 0.0))
            break;
        if (depth == 1) {
            // The camera paths' densities of the light vertex relative to the emission's.
            LightSampleContext ctx = LightSampleContext(si);
            if (e.type == EmitterType::environment) {
                const float light_pdf = light_sampler.eval_environment_pdf(ctx, -e.dir_ws);
                mis.nee = light_pdf / (s.mis_path_count * emission_density * cos_geom_in);
                mis.hit = 1.0 / (s.mis_path_count * emission_density * cos_geom_in);
            } else {
                const float3 d = v.pos_ws - e.pos_ws;
                const float d2 = dot(d, d);
                const float cos_light = point_light ? 1.0 : abs(dot(e.normal_ws, e.dir_ws));
                float light_area_pdf;
                if (e.type == EmitterType::emissive_triangle)
                    light_area_pdf = light_sampler.eval_emissive_pdf(ctx, EmissiveTriangleID(e.index), e.pos_ws, e.normal_ws)
                        * cos_light / d2;
                else if (point_light)
                    light_area_pdf = light_sampler.eval_analytic_pdf(ctx, e.index, e.pos_ws, e.normal_ws);
                else
                    light_area_pdf = light_sampler.eval_analytic_pdf(ctx, e.index, e.pos_ws, e.normal_ws) * cos_light / d2;
                mis.nee = light_area_pdf * d2 / (s.mis_path_count * emission_density * cos_geom_in);
                mis.hit = hittable ? cos_light / (s.mis_path_count * emission_density * cos_geom_in) : 0.0;
            }
        } else {
            mis.nee /= cos_geom_in;
            mis.hit /= cos_geom_in;
        }

        if (ENABLE_INTERIOR_TRACKING)
            si.exterior_ior = interiors.exterior_ior(v.material_id, si.front_facing);
        let mi = g_scene.get_material_instance(hit, si, ExplicitLodSampler(0), MaterialInstanceHints::adjust_shading_normal);

        connect_to_camera(v, beta, depth, interiors, mis, sg);

        if (depth >= max_vertices)
            break;

        // Adjoint scattering: sample the camera-side direction given the light-side direction.
        MaterialSample sample = {};
        if (!mi.sample(si, sg, sample))
            break;
        const float3 wo = sample.wo_ws;
        const float cos_geom_out = dot(wo, v.normal_ws);
        // One-sided materials are only seen from the front by the path tracer.
        if (!is_set(v.header.flags, MaterialFlags::two_sided) && cos_geom_out < 0.0)
            break;
        const bool transmits = cos_geom_out * dot(v.to_light_ws, v.normal_ws) < 0.0;
        const bool is_thin_walled = is_set(v.header.flags, MaterialFlags::thin_walled);

        InteriorList next_interiors = interiors;
        if (ENABLE_INTERIOR_TRACKING && transmits && !is_thin_walled) {
            if (cross_interior_boundary(si.front_facing, v.material_id, v.header.nested_priority, mi, next_interiors)
                != InteriorListUpdateResult::success)
                break;
        }

        const bool is_delta = sample.has_flag(BSDFFlags::delta);

        // Scattering weight. Non-delta lobes of moderate pdf: the path tracer's BSDF value (wi = camera side,
        // its shading-normal adjustment) per sampled pdf, so the light tracer integrates exactly the path
        // tracer's integrand: f_pt |wi.ns| |wo.ng| / (|wi.ng| pdf). Delta and near-specular lobes (where a
        // separately set-up evaluation of a lobe of width ~1e-3 is unreliable): the light-side sample weight with
        // Veach's adjoint shading-normal correction |wi.ns| |wo.ng| / (|wi.ng| |wo.ns|) (assumes reciprocity).
        // The camera-side evaluation also gives the camera paths' directional density here (MIS).
        float3 weight;
        if (!is_delta) {
            const SurfaceInteraction cam_si = make_camera_side_interaction(v, wo, next_interiors);
            let cam_mi = g_scene.get_material_instance(hit, cam_si, ExplicitLodSampler(0), MaterialInstanceHints::adjust_shading_normal);
            const BSDFEval cam = cam_mi.eval_with_pdf(cam_si, v.to_light_ws, sg);
            if (sample.pdf <= EXACT_ADJOINT_MAX_PDF)
                weight = cam.value * (abs(cos_geom_out) / (cos_geom_in * sample.pdf));
            else
                weight = sample.weight * adjoint_shading_correction(v, wo, cos_geom_in, cos_geom_out);
            if (depth == 1 && !camera_nee_possible(v, cam_si, cam_mi.get_lobe_types(cam_si), next_interiors))
                mis.nee = 0.0;
            mis.nee *= (depth >= 2 ? cam.pdf : 1.0) * abs(cos_geom_out) / sample.pdf;
            mis.hit *= cam.pdf * abs(cos_geom_out) / sample.pdf;
        } else {
            weight = sample.weight * adjoint_shading_correction(v, wo, cos_geom_in, cos_geom_out);
            // A delta vertex: the camera paths sample the same direction (ratio one); next-event estimation is
            // impossible there.
            if (depth == 1)
                mis.nee = 0.0;
            mis.nee *= abs(cos_geom_out);
            mis.hit *= abs(cos_geom_out);
        }
        if (!any(weight > 0.0))
            break;

        if (ENABLE_RUSSIAN_ROULETTE && depth >= s.rr_depth) {
            const float survival = min(max(weight.x, max(weight.y, weight.z)), RUSSIAN_ROULETTE_MAX_SURVIVAL);
            if (!(next_float(sg) < survival))
                break;
            weight /= survival;
        }
        beta *= weight;
        if (!any(beta > 0.0) || !all(isfinite(beta)))
            break;

        interiors = next_interiors;
        ray = Ray(offset_ray_origin(v.pos_ws, cos_geom_out >= 0.0 ? v.normal_ws : -v.normal_ws), wo);
    }
}

//------------------------------------------
// Resolve

/// Adds the light tracer's splats to the path tracer's frame (scale applied when splatting).
public void resolve(RaySampler ray_sampler, float4 path_tracer_color, bool add_path_tracer, out float4 output)
{
    const uint2 id = ray_sampler.id;
    const uint2 dims = ray_sampler.camera.dims;
    const uint address = (id.y * dims.x + id.x) * 12;
    const float3 light = asfloat(g_light_tracer.accumulation.Load3(address));
    const float3 base = add_path_tracer ? path_tracer_color.xyz : float3(0.0);
    output = float4(base + light, 1.0);
}

/// Clears the accumulation buffer (one thread per float).
public void clear_accumulation(uint tid, uint count)
{
    if (tid < count)
        g_light_tracer.accumulation.Store(tid * 4, 0u);
}
