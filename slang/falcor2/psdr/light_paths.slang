// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Light term: light paths from the scene's lights (an emitter chosen by the light sampler's emission selection, then a
// position and a direction: render/light_emission.slang) connected to the camera at every surface vertex - the t = 1
// strategy of bidirectional path tracing (Veach 1997, ch. 10) - with derivatives in material form. The light paths are
// falcor2's adjoint light tracer's (rendernodes/adjoint_pathtracer.slang trace_light_paths(): the same random numbers,
// opacity, interiors and absorbing media, Russian roulette and camera connections; keep in sync), so the primal image
// (camera paths + light splats of a frame) is the AdjointPathTracerNode's frame. Together with the camera paths'
// emission (s = 0) and next-event (s = 1) estimates the strategies are combined by multiple importance sampling
// (balance heuristic, as the adjoint tracer): each side evaluates the other strategies' densities of its paths (partial
// density ratios accumulated along the subpaths, paths.slang for the camera paths). The weights are detached.
//
// Derivatives: the vertices are fixed (no geometric derivatives along light paths: materials and the lights' emission);
// a connection's value C = Le x prod(scattering factors) x connection BSDF x detached geometry, pdfs and weight, so
// dC = C (dLe / Le + sum d f / f), each factor a ratio carrier of value one (light_path_factor(),
// light_emission_factor()). Backward mode replays the light path with the same random numbers (as the interior term
// replays camera paths) and back-propagates dL/dI at each connection's pixel.

/// Light tracing on: the light term runs and the camera paths' estimates are weighted against it.
public bool light_tracing;
/// Whether render_light computes derivatives (forward mode: into light_interior; backward mode always).
public bool light_derivatives;
/// Scale of each connection (1 / the number of light paths of the frame), as the adjoint tracer's.
public float light_contribution_scale;
/// Weight of a frame's light term in the derivatives (1 / frames per estimate).
public float light_sample_weight;
/// Light paths per frame over camera samples per pixel (the MIS sample-count ratio).
public float light_mis_path_count;
public LightEmissionSettings light_emission;
/// Splats of the current frame (primal) and (forward mode) derivative images, 3 floats per pixel.
public RWByteAddressBuffer light_primal;
public RWByteAddressBuffer light_interior;

/// Largest sampling pdf (1/sr) of a non-delta lobe weighted with the camera-side BSDF evaluation (the adjoint light
/// tracer's EXACT_ADJOINT_MAX_PDF).
static const float LIGHT_EXACT_MAX_PDF = 100.0;
/// Camera depth range of the adjoint tracer's camera connections.
static const float LIGHT_CAMERA_RAY_T_MAX = 1000.0;

/// The bound light sampler as a value (for the generic emission functions).
LightSampler scene_light_sampler() { return light_sampler; }

/// Camera pdf (area measure) of a point seen by the camera, as the camera path's primary sample.
float camera_area_pdf(float3 p, float3 n)
{
    float3 d = normalize(p - camera.position);
    return sensor_importance(p) * abs(dot(n, d));
}

//------------------------------------------
// Differentiable factors (values one)

[Differentiable]
float3 ratio3(float3 x, no_diff float3 d) { return float3(ratio(x.x, d.x), ratio(x.y, d.y), ratio(x.z, d.z)); }

/// The emitted radiance relative to its value: analytic lights' emission parameters, the environment map
/// (intensity, rotation, texels) along to_light, or the emissive material at a triangle point.
[Differentiable]
float3 light_emission_factor(float theta, no_diff uint type, no_diff uint index, no_diff HitRef h, no_diff float3 dir,
                             no_diff float3 primal)
{
    if (type == uint(EmitterType::analytic)) {
        no_diff float3 value = no_diff light_value[index];
        no_diff float3 inverse = 1.0 / max(value, 1e-20);
        return param3(theta, value, light_slot_base + LIGHT_SLOTS * index) * inverse;
    }
    if (type == uint(EmitterType::environment)) {
        float3 e = env_map_radiance(theta, -dir);
        return ratio3(e, detach(e));
    }
    Surface s = no_diff make_surface(h, dir);
    Vertex v = no_diff load_vertex(0.0, h);
    float3 e = eval_emission(theta, s, v.n, dir, primal);
    return ratio3(e, detach(e));
}

/// An interaction at `h` relative to its value: the BSDF f(wi, wo) |cos wo| evaluated camera-side (wi towards the
/// camera end of the path, wo towards the light), or a SpongeCake delta transmission along wi.
[Differentiable]
float3 light_path_factor(float theta, no_diff HitRef h, no_diff float3 wi, no_diff float3 wo, no_diff bool delta,
                         no_diff float exterior_ior)
{
    Surface s = no_diff make_surface(h, wi, exterior_ior);
    if (delta)
        return float3(delta_scale(theta, s, wi));
    float3 f1, f2;
    eval_bsdf(theta, s, wi, wo, float3(0), float3(1), float3(0), f1, f2);
    return ratio3(f1, detach(f1));
}

void add_light_image(RWByteAddressBuffer image, uint index, float3 value)
{
    [ForceUnroll]
    for (uint c = 0; c < 3; c++)
        atomic_add(image, 12 * index + 4 * c, value[c]);
}

/// HitRef of an emissive triangle point.
HitRef emissive_hit(uint triangle_id, float3 bary)
{
    EmissiveTriangle t = g_scene.emissive_geometry.get_triangle(EmissiveTriangleID(triangle_id));
    HitRef h;
    h.gid = t.geometry_instance_id;
    h.prim = triangle_id - uint(g_scene.emissive_geometry.geometry_instance_to_triangle_id[uint(t.geometry_instance_id)]);
    h.bary = bary.yz;
    return h;
}

//------------------------------------------
// The adjoint light tracer's light paths (keep in sync with adjoint_pathtracer.slang)

/// A light-path surface vertex.
struct LightVertex
{
    TriangleHitInfo hit;
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

SurfaceInteraction camera_side_interaction(LightVertex v, float3 wi_ws, InteriorList camera_side_interiors)
{
    SurfaceInteraction si = g_scene.make_triangle_surface_interaction(v.hit, wi_ws);
    si.exterior_ior = camera_side_interiors.exterior_ior(v.material_id, si.front_facing);
    return si;
}

/// Whether the path tracer's next-event estimation at the first vertex samples the light's direction.
bool camera_nee_possible(LightVertex v, SurfaceInteraction cam_si, BSDFFlags lobe_types, InteriorList camera_side_interiors)
{
    const bool is_thin_walled = is_set(v.header.flags, MaterialFlags::thin_walled);
    const uint hemispheres = material_nee_hemispheres(lobe_types)
        & nee_visibility_hemispheres(cam_si.front_facing, v.material_id, is_thin_walled, camera_side_interiors);
    const bool upper = dot(v.to_light_ws, cam_si.front_normal_ws()) >= 0.0;
    return upper ? (hemispheres & HEMISPHERE_UPPER) != 0 : (hemispheres & HEMISPHERE_LOWER) != 0;
}

/// Veach's adjoint factor for shading normals, |wi.ns| |wo.ng| / (|wi.ng| |wo.ns|), with the interpolated normal.
float adjoint_shading_correction(LightVertex v, float3 wo, float cos_geom_in, float cos_geom_out)
{
    const float cos_shading_in = abs(dot(v.to_light_ws, v.shading_normal_ws));
    const float cos_shading_out = abs(dot(wo, v.shading_normal_ws));
    if (!(cos_shading_out > 0.0))
        return 0.0;
    return (cos_shading_in * abs(cos_geom_out)) / (cos_geom_in * cos_shading_out);
}

/// A camera connection: its pixel, the camera direction, the camera-side exterior IOR and its value.
struct LightConnection
{
    bool connected;
    uint pixel;
    float3 dir;
    float exterior_ior;
    float3 value;
};

/// Connects a light-path vertex to the camera (the adjoint tracer's connect_to_camera()); splats into light_primal.
LightConnection connect_to_camera(LightVertex v, float3 beta, uint depth, InteriorList interiors, float mis_nee,
                                  float mis_hit, inout UniformSampleGenerator sg)
{
    LightConnection c = {};
    const float3 to_camera = camera.position - v.pos_ws;
    const float dist2 = dot(to_camera, to_camera);
    if (!(dist2 > 0.0))
        return c;
    const float dist = sqrt(dist2);
    const float3 w_camera = to_camera / dist;

    const float image_w_len = length(camera.image_w);
    const float cos_camera = dot(-w_camera, camera.image_w) / image_w_len;
    if (!(cos_camera > 0.0))
        return c;
    const float2 pixel_pos = camera.project_world_to_pixel(v.pos_ws);
    if (!(all(pixel_pos >= 0.0) && all(pixel_pos < float2(camera.dims))))
        return c;
    const uint2 pixel = min(uint2(pixel_pos), camera.dims - 1);
    if (dist > LIGHT_CAMERA_RAY_T_MAX)
        return c;

    const float cos_geom_camera = dot(w_camera, v.normal_ws);
    if (!is_set(v.header.flags, MaterialFlags::two_sided) && cos_geom_camera < 0.0)
        return c;
    const float cos_geom_light = abs(dot(v.to_light_ws, v.normal_ws));
    if (!(cos_geom_light > 0.0))
        return c;

    InteriorList camera_side_interiors = interiors;
    const bool transmits = dot(w_camera, v.normal_ws) * dot(v.to_light_ws, v.normal_ws) < 0.0;
    const bool is_thin_walled = is_set(v.header.flags, MaterialFlags::thin_walled);
    if (transmits && !is_thin_walled) {
        if (v.light_side_front || camera_side_interiors.exit(v.material_id) != InteriorListUpdateResult::success)
            return c;
    }

    const SurfaceInteraction cam_si = camera_side_interaction(v, w_camera, camera_side_interiors);
    let cam_mi = hit_material_instance(v.material_id, cam_si);
    const BSDFEval eval = cam_mi.eval_with_pdf(cam_si, v.to_light_ws, sg);
    if (!any(eval.value > 0.0))
        return c;

    const float pixel_area = 4.0 * length(camera.image_u) * length(camera.image_v) / float(camera.dims.x * camera.dims.y);
    const float importance = sqr(image_w_len) / (pixel_area * cos_camera * cos_camera * cos_camera * dist2);

    float weight = 1.0;
    {
        const float camera_area_pdf = importance * abs(cos_geom_camera);
        float nee = mis_nee * (depth >= 2 ? eval.pdf : 1.0);
        if (depth == 1 && !camera_nee_possible(v, cam_si, cam_mi.get_lobe_types(cam_si), camera_side_interiors))
            nee = 0.0;
        const float ratio = camera_area_pdf * (nee + mis_hit * eval.pdf);
        weight = 1.0 / (1.0 + ratio);
    }

    float3 target = offset_ray_origin(v.pos_ws, cos_geom_camera >= 0.0 ? v.normal_ws : -v.normal_ws);
    float3 to_target = target - camera.position;
    const float t_max = length(to_target);
    if (!(t_max > 0.0))
        return c;
    Ray camera_ray = Ray(camera.position, to_target / t_max, 0.0, t_max);
    OpacityEvaluator opacity = OpacityEvaluator(sg.next_uint());
    if (!opacity.is_valid_hit(camera_ray, v.hit, SHADING_HINTS))
        return c;
    if (occluded_opacity(camera_ray, opacity))
        return c;

    const float3 contribution = beta * eval.value * (abs(cos_geom_camera) / cos_geom_light) * importance * weight
        * light_contribution_scale;
    if (!all(isfinite(contribution)))
        return c;
    c.connected = true;
    c.pixel = pixel.y * camera.dims.x + pixel.x;
    c.dir = w_camera;
    c.exterior_ior = cam_si.exterior_ior;
    c.value = contribution;
    return c;
}

/// Splats a connection into an image buffer with float atomics in channel order (as the adjoint tracer's splat()).
void splat_light(RWByteAddressBuffer image, uint index, float3 value)
{
    const uint address = index * 12;
    image.InterlockedAddF32(address + 0, value.x);
    image.InterlockedAddF32(address + 4, value.y);
    image.InterlockedAddF32(address + 8, value.z);
}

/// One light path per thread of frame `frame` (the adjoint tracer's trace_light_paths() with s.iteration = frame).
public void render_light(uint tid, uint count, uint frame)
{
    if (!light_tracing || tid >= count)
        return;
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid, 0x40000000u), frame);
    EmissionSample e;
    if (!sample_light_emission(scene_light_sampler(), light_emission, sg, e))
        return;
    float3 beta = e.weight;
    const float emission_density = e.pdf_pos * e.pdf_dir;
    const bool point_light = e.type == EmitterType::analytic && all(e.normal_ws == 0.0);
    const bool hittable = e.type != EmitterType::analytic;
    HitRef light_hit = {};
    if (e.type == EmitterType::emissive_triangle)
        light_hit = emissive_hit(e.index, e.barycentrics);

    // Start far light rays at the scene's bounding sphere (nothing lies outside it).
    float t_enter, t_exit;
    if (!light_intersect_sphere(e.pos_ws, e.dir_ws, light_emission.scene_center, light_emission.scene_radius, t_enter,
                                t_exit) || !(t_exit > 0.0))
        return;
    // Rays from emissive triangles leave their surface (offset origin, no self-intersection).
    const float3 start = e.type == EmitterType::emissive_triangle ? offset_ray_origin(e.pos_ws, e.normal_ws)
                                                                  : e.pos_ws + e.dir_ws * max(t_enter, 0.0);
    Ray ray = Ray(start, e.dir_ws, 0.0, t_exit - max(t_enter, 0.0) + 1.0);

    // The path (hits, directions towards the light, sampled camera-side directions, delta flags, exterior IORs) and
    // its camera connections, for the derivatives.
    HitRef hits[MAX_DEPTH];
    float3 to_light[MAX_DEPTH];
    float3 next_dir[MAX_DEPTH];
    float next_ior[MAX_DEPTH];
    bool delta[MAX_DEPTH];
    LightConnection conn[MAX_DEPTH];
    uint n = 0;

    InteriorList interiors = {};
    uint depth = 0;
    const uint max_vertices = max_bounces;      // the path tracer's max_depth - 1
    float mis_nee = 0.0, mis_hit = 0.0;

    [loop]
    while (depth < max_vertices) {
        OpacityEvaluator opacity = OpacityEvaluator(sg.next_uint());
        TriangleHitInfo hit;
        if (!intersect_opacity(ray, opacity, hit))
            break;

        LightVertex v;
        v.hit = hit;
        SurfaceInteraction si = g_scene.make_triangle_surface_interaction(hit, -ray.dir);
        v.pos_ws = si.position_ws;
        v.normal_ws = si.normal_ws;
        v.shading_normal_ws = si.shading_frame_ws.normal;
        v.to_light_ws = -ray.dir;
        v.light_side_front = si.front_facing;
        v.material_id = g_scene.get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        v.header = g_scene.material_system.get_material_data(v.material_id).header;

        // Media along the segment: absorption only (scattering media are left to the path tracer).
        if (!interiors.is_empty()) {
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
                mis_nee = light_pdf / (light_mis_path_count * emission_density * cos_geom_in);
                mis_hit = 1.0 / (light_mis_path_count * emission_density * cos_geom_in);
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
                mis_nee = light_area_pdf * d2 / (light_mis_path_count * emission_density * cos_geom_in);
                mis_hit = hittable ? cos_light / (light_mis_path_count * emission_density * cos_geom_in) : 0.0;
            }
        } else {
            mis_nee /= cos_geom_in;
            mis_hit /= cos_geom_in;
        }

        si.exterior_ior = interiors.exterior_ior(v.material_id, si.front_facing);
        let mi = hit_material_instance(v.material_id, si);

        LightConnection c = connect_to_camera(v, beta, depth, interiors, mis_nee, mis_hit, sg);
        if (c.connected && !backward)
            splat_light(light_primal, c.pixel, c.value);
        hits[n].gid = hit.geometry_instance_id;
        hits[n].prim = hit.primitive_index;
        hits[n].bary = hit.barycentrics;
        to_light[n] = v.to_light_ws;
        next_dir[n] = float3(0);
        next_ior[n] = 1.0;
        delta[n] = false;
        conn[n] = c;
        n++;

        if (depth >= max_vertices)
            break;

        // Adjoint scattering: sample the camera-side direction given the light-side direction.
        MaterialSample sample = {};
        if (!mi.sample(si, sg, sample))
            break;
        const float3 wo = sample.wo_ws;
        const float cos_geom_out = dot(wo, v.normal_ws);
        if (!is_set(v.header.flags, MaterialFlags::two_sided) && cos_geom_out < 0.0)
            break;
        const bool transmits = cos_geom_out * dot(v.to_light_ws, v.normal_ws) < 0.0;
        const bool is_thin_walled = is_set(v.header.flags, MaterialFlags::thin_walled);

        InteriorList next_interiors = interiors;
        if (transmits && !is_thin_walled) {
            if (cross_interior(si.front_facing, v.material_id, v.header.nested_priority, mi, next_interiors)
                != InteriorListUpdateResult::success)
                break;
        }

        const bool is_delta = sample.has_flag(BSDFFlags::delta);
        float3 weight;
        float camera_ior = 1.0;
        if (!is_delta) {
            const SurfaceInteraction cam_si = camera_side_interaction(v, wo, next_interiors);
            camera_ior = cam_si.exterior_ior;
            let cam_mi = hit_material_instance(v.material_id, cam_si);
            const BSDFEval cam = cam_mi.eval_with_pdf(cam_si, v.to_light_ws, sg);
            if (sample.pdf <= LIGHT_EXACT_MAX_PDF)
                weight = cam.value * (abs(cos_geom_out) / (cos_geom_in * sample.pdf));
            else
                weight = sample.weight * adjoint_shading_correction(v, wo, cos_geom_in, cos_geom_out);
            if (depth == 1 && !camera_nee_possible(v, cam_si, cam_mi.get_lobe_types(cam_si), next_interiors))
                mis_nee = 0.0;
            mis_nee *= (depth >= 2 ? cam.pdf : 1.0) * abs(cos_geom_out) / sample.pdf;
            mis_hit *= cam.pdf * abs(cos_geom_out) / sample.pdf;
        } else {
            weight = sample.weight * adjoint_shading_correction(v, wo, cos_geom_in, cos_geom_out);
            // A delta vertex: the camera paths sample the same direction (ratio one); next-event estimation is
            // impossible there.
            if (depth == 1)
                mis_nee = 0.0;
            mis_nee *= abs(cos_geom_out);
            mis_hit *= abs(cos_geom_out);
        }
        if (!any(weight > 0.0))
            break;

        if (depth >= rr_depth) {
            const float survival = min(max(weight.x, max(weight.y, weight.z)), RUSSIAN_ROULETTE_MAX_SURVIVAL);
            if (!(next_float(sg) < survival))
                break;
            weight /= survival;
        }
        beta *= weight;
        if (!any(beta > 0.0) || !all(isfinite(beta)))
            break;

        interiors = next_interiors;
        delta[n - 1] = is_delta;
        next_dir[n - 1] = wo;
        next_ior[n - 1] = camera_ior;
        ray = Ray(offset_ray_origin(v.pos_ws, cos_geom_out >= 0.0 ? v.normal_ws : -v.normal_ws), wo);
    }
    if (!light_derivatives)
        return;

    // The emitted radiance's primal value at the light (emissive materials: for the ratio carrier).
    float3 emitted = float3(1.0);
    if (backward) {
        // Each factor's adjoint is the sum of the dL/dI-weighted connections that contain it.
        float3 suffix = 0.0;
        for (int k = int(n) - 1; k >= 0; k--) {
            if (k + 1 < int(n) && any(suffix != 0.0) && any(next_dir[k] != 0.0)) {
                var theta = diffPair(0.0);
                bwd_diff(light_path_factor)(theta, hits[k], next_dir[k], to_light[k], delta[k], next_ior[k], suffix);
            }
            if (conn[k].connected) {
                float3 a = dl_di[conn[k].pixel] * conn[k].value * light_sample_weight;
                var theta = diffPair(0.0);
                bwd_diff(light_path_factor)(theta, hits[k], conn[k].dir, to_light[k], false, conn[k].exterior_ior, a);
                suffix += a;
            }
        }
        if (any(suffix != 0.0)) {
            var theta = diffPair(0.0);
            bwd_diff(light_emission_factor)(theta, uint(e.type), e.index, light_hit, e.dir_ws, emitted, suffix);
        }
    } else {
        float3 prefix = fwd_diff(light_emission_factor)(diffPair(0.0, 1.0), uint(e.type), e.index, light_hit, e.dir_ws,
                                                        emitted).d;
        for (uint k = 0; k < n; k++) {
            if (conn[k].connected) {
                float3 d = fwd_diff(light_path_factor)(diffPair(0.0, 1.0), hits[k], conn[k].dir, to_light[k], false,
                                                       conn[k].exterior_ior).d;
                float3 value = conn[k].value * light_sample_weight * (prefix + d);
                if (all(isfinite(value)))
                    add_light_image(light_interior, conn[k].pixel, value);
            }
            if (k + 1 < n && any(next_dir[k] != 0.0))
                prefix += fwd_diff(light_path_factor)(diffPair(0.0, 1.0), hits[k], next_dir[k], to_light[k], delta[k],
                                                      next_ior[k]).d;
        }
    }
}

bool light_intersect_sphere(float3 origin, float3 dir, float3 center, float radius, out float t0, out float t1)
{
    float3 oc = origin - center;
    float b = dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    t0 = 0.0;
    t1 = 0.0;
    if (disc < 0.0)
        return false;
    float s = sqrt(disc);
    t0 = -b - s;
    t1 = -b + s;
    return true;
}
