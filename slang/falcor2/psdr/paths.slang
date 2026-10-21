// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Camera paths: records of their vertices and light samples, and the differentiable terms of their
// vertices.
//
// The camera paths are falcor2's reference path tracer (rendernodes/reference_pathtracer.slang) with its defaults and
// next-event estimation with multiple importance sampling (balance heuristic) on: the same random numbers in the same
// order (camera ray jitter, the opacity evaluator's seed per ray, light and BSDF samples, Russian roulette), material
// opacity (cut-outs), the next-event hemisphere rules, closed-surface interiors with their homogeneous media
// (volume scattering vertices) and, with light tracing, its MIS weights against the light tracer. trace_path() records
// the path; its radiance is computed as the reference path tracer computes it (rec.radiance), so that the primal image
// is the reference path tracer's image at the same frame index (pixel-identical up to the compiled arithmetic). Keep
// in sync with reference_pathtracer.slang.

/// Terms of path vertex k in material form: the emission towards the previous vertex, the next-event
/// estimate, and the throughput to vertex k + 1 or, at the last vertex, the environment radiance it sees
/// (step). See vertex_terms().
struct VertexTerms : IDifferentiable
{
    float3 emission;
    float3 nee;
    float3 step;
};

/// Vertex flags: a homogeneous-medium scattering vertex (no surface, no derivatives).
static const uint VERTEX_VOLUME = 1;

/// Detached path: vertex hits, sampling pdfs and MIS weights recorded by the primal tracer.
struct PathRecord
{
    uint length;
    float3 origin;
    HitRef hit[MAX_DEPTH + 1];
    /// VERTEX_VOLUME, else a surface vertex.
    uint flags[MAX_DEPTH + 1];
    /// World position of each vertex (volume vertices have no hit).
    float3 position[MAX_DEPTH + 1];
    /// Exterior IOR of the surface interaction (interior tracking).
    float exterior_ior[MAX_DEPTH + 1];
    /// Solid-angle pdf of the direction that generated vertex i (i > 0), or zero for delta lobes.
    float pdf[MAX_DEPTH + 1];
    /// Sample weight (BSDF value over pdf) of the direction that generated hit[i], or of the last
    /// sampled direction (i = length) when it escapes.
    float3 delta_weight[MAX_DEPTH + 1];
    /// Detached factors of the throughput from vertex i to vertex i + 1 besides the sample weight: Russian roulette
    /// (1 / survival) and the medium weight of the segment.
    float3 step_scale[MAX_DEPTH + 1];
    /// MIS weight of the emission found at hit[i].
    float mis[MAX_DEPTH + 1];
    /// Emitted radiance at hit[i] for other materials (see eval_bsdf).
    float3 emission[MAX_DEPTH + 1];
    LightRecord light[MAX_DEPTH + 1];
    /// Primal vertex terms from the material system's values, so that path values need no evaluation.
    VertexTerms terms[MAX_DEPTH + 1];
    /// Environment radiance (MIS-weighted) seen by the last sampled direction, and its factor (MIS and delta
    /// lobe weight) on the environment radiance.
    float3 miss_radiance;
    float3 miss_scale;
    float3 miss_dir;
    float miss_pdf;
    /// The path's radiance, accumulated as the reference path tracer does.
    float3 radiance;
    /// Whether the first vertex sampled a glossy reflection, for which the path has an antithetic partner (with
    /// antithetic sampling, see trace_path()).
    bool antithetic;
};

bool is_volume_vertex(PathRecord rec, uint k) { return (rec.flags[k] & VERTEX_VOLUME) != 0; }

/// Antithetic BSDF samples: the second number of a sample is turned by half a turn (u -> u + 0.5 mod 1), which keeps
/// the samples' distribution. StandardMaterial and OpenPBRMaterial choose the lobe with the first number and the
/// azimuth of a GGX visible normal with the second: the samples of a glossy lobe lie on either side of its center,
/// where its derivatives (e.g. with respect to the normal) have opposite signs.
struct AntitheticSampleGenerator : ISampleGenerator
{
    UniformSampleGenerator sg;
    uint count;

    __init(uint seed) { sg = UniformSampleGenerator(seed); count = 0; }
    __init(uint2 pixel, uint sample_index) { sg = UniformSampleGenerator(pixel, sample_index); count = 0; }
    __init(UniformSampleGenerator s) { sg = s; count = 0; }

    [mutating]
    uint next_uint()
    {
        uint u = sg.next_uint();
        return count++ == 1 ? u + 0x80000000u : u;
    }
};

float3 camera_forward() { return normalize(camera.image_w); }

float3 camera_dir(float2 film)
{
    float2 uv = film / float2(camera.dims) * 2 - 1;
    return normalize(uv.x * camera.image_u - uv.y * camera.image_v + camera.image_w);
}

/// The reference path tracer's camera ray of a pixel (render(): pixel jitter, and the lens sample with depth of field).
Ray primary_ray(uint2 pixel, inout UniformSampleGenerator sg)
{
    float2 pixel_offset = float2(0.5, 0.5);
    pixel_offset += (next_float2(sg) - 0.5) * 1.0;
    pixel_offset = saturate(pixel_offset + float2(0.0, 0.0));
    float2 uv = (float2(pixel) + pixel_offset) / float2(camera.dims);
    uv.y = 1 - uv.y;
    Ray ray = camera.get_ray(uv);
    if (camera.aperture_radius > 0.0 && camera.focus_distance > 0.0)
        ray = camera.get_ray(uv, next_float2(sg));
    return ray;
}

//------------------------------------------
// Reference path tracer rules (keep in sync with reference_pathtracer.slang)

static const uint HEMISPHERE_UPPER = 1;
static const uint HEMISPHERE_LOWER = 2;
static const uint HEMISPHERE_BOTH = 3;
static const float RUSSIAN_ROULETTE_MAX_SURVIVAL = 0.95;

uint material_nee_hemispheres(BSDFFlags lobe_types)
{
    uint result = 0;
    if (is_set(lobe_types, BSDFFlags::non_delta_reflection))
        result |= HEMISPHERE_UPPER;
    if (is_set(lobe_types, BSDFFlags::non_delta_transmission))
        result |= HEMISPHERE_LOWER;
    if (is_set(lobe_types, BSDFFlags::curve))
        result |= HEMISPHERE_BOTH;
    return result;
}

uint nee_visibility_hemispheres(bool is_entering, MaterialID material_id, bool is_thin_walled, InteriorList interiors)
{
    if (is_thin_walled)
        return interiors.is_empty() ? HEMISPHERE_BOTH : 0;
    if (is_entering)
        return interiors.is_empty() ? HEMISPHERE_UPPER : 0;
    if (interiors.is_empty())
        return HEMISPHERE_UPPER;
    return interiors.contains_only(material_id) ? HEMISPHERE_LOWER : 0;
}

bool in_sampled_hemispheres(float3 normal_ws, uint hemispheres, float3 dir_ws)
{
    bool upper = dot(dir_ws, normal_ws) >= 0.0;
    return upper ? (hemispheres & HEMISPHERE_UPPER) != 0 : (hemispheres & HEMISPHERE_LOWER) != 0;
}

/// Light sample context whose receiver normal covers exactly one sampled hemisphere (zero: unrestricted).
LightSampleContext hemisphere_light_context(SurfaceInteraction si, uint hemispheres)
{
    LightSampleContext ctx = LightSampleContext(si);
    float3 upper_normal_ws = si.front_normal_ws();
    if (hemispheres == HEMISPHERE_UPPER)
        ctx.normal_ws = upper_normal_ws;
    else if (hemispheres == HEMISPHERE_LOWER)
        ctx.normal_ws = -upper_normal_ws;
    else
        ctx.normal_ws = {};
    return ctx;
}

/// The scattering vertex as the reference path tracer's make_scatter_interaction() sees it.
SurfaceInteraction scatter_interaction(float3 ray_origin, float3 scatter_normal, float3 ray_dir)
{
    SurfaceInteraction si = {};
    si.position_ws = ray_origin;
    si.normal_ws = scatter_normal;
    si.front_facing = true;
    si.wi_ws = -ray_dir;
    return si;
}

/// The scatter normal as the reference path tracer stores it in its path state (octahedral, zero kept).
float3 stored_scatter_normal(float3 n) { return any(n != 0.0) ? oct_to_ndir_snorm(ndir_to_oct_snorm(n)) : float3(0.0); }

InteriorListUpdateResult cross_interior(bool is_entering, MaterialID material_id, uint nested_priority,
                                        IMaterialInstance mi, inout InteriorList interiors)
{
    if (is_entering) {
        HomogeneousMediumProperties medium = mi.get_interior_medium();
        let entry = interiors.make_entry(material_id, nested_priority, mi.get_interior_ior(), medium);
        return interiors.enter(entry);
    }
    return interiors.exit(material_id);
}

/// Russian roulette as the reference path tracer's (depth: path vertices so far): false terminates the path,
/// otherwise the throughput is divided by the survival probability (returned in `factor`).
bool russian_roulette(uint depth, inout float3 throughput, inout UniformSampleGenerator sg, out float factor)
{
    factor = 1.0;
    if (depth < rr_depth)
        return true;
    float3 t = detach(throughput);
    float survival = min(max(t.x, max(t.y, t.z)), RUSSIAN_ROULETTE_MAX_SURVIVAL);
    if (next_float(sg) > survival)
        return false;
    throughput /= survival;
    factor = 1.0 / survival;
    return true;
}

/// Closest hit along a ray with the scene's opacity evaluation (falcor2's RayQuerySceneIntersector for triangles; the
/// reference path tracer's any-hit shader evaluates the same opacity per candidate).
bool intersect_opacity(Ray ray, OpacityEvaluator opacity, out TriangleHitInfo hit)
{
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

bool occluded_opacity(Ray ray, OpacityEvaluator opacity)
{
    RayQuerySceneIntersector intersector = RayQuerySceneIntersector(opacity);
    return intersector.is_occluded(ray, SHADING_HINTS);
}

/// Visibility ray towards a light sample (the reference path tracer's create_visibility_ray()).
Ray light_visibility_ray(SurfaceInteraction si, LightSample ls)
{
    Ray ray;
    ray.origin = offset_ray_origin(si.position_ws, dot(ls.dir_ws, si.normal_ws) >= 0.0 ? si.normal_ws : -si.normal_ws);
    ray.dir = ls.dir_ws;
    ray.t_min = 0.0;
    ray.t_max = ls.distance;
    if (ls.emitter_type == EmitterType::emissive_triangle) {
        float3 target = offset_ray_origin(ls.pos_ws, dot(-ls.dir_ws, ls.normal_ws) >= 0.0 ? ls.normal_ws : -ls.normal_ws);
        float3 to_light = target - ray.origin;
        ray.t_max = length(to_light);
        ray.dir = to_light / ray.t_max;
    }
    return ray;
}

IMaterialInstance hit_material_instance(MaterialID material_id, SurfaceInteraction si)
{
    return g_scene.material_system.get_material(material_id).setup_material_instance(si, ExplicitLodSampler(0), SHADING_HINTS);
}

/// The light tracer's density of sampling the direction towards the previous path vertex at the current vertex, given
/// light arriving from to_light (solid angle); 1 at the primary hit (depth: path vertices so far).
float light_tracer_reverse_pdf(TriangleHitInfo hit, MaterialID material_id, SurfaceInteraction si, float3 to_light,
                               uint depth)
{
    if (depth <= 1)
        return 1.0;
    SurfaceInteraction light_si = g_scene.make_triangle_surface_interaction(hit, to_light);
    light_si.exterior_ior = si.exterior_ior;
    let light_mi = hit_material_instance(material_id, light_si);
    UniformSampleGenerator eval_sg = UniformSampleGenerator(0);
    return light_mi.eval_with_pdf(light_si, si.wi_ws, eval_sg).pdf;
}

/// p_t1 / p_s1 of a next-event sample (the reference path tracer's light_tracer_nee_ratio()).
float light_tracer_nee_ratio(TriangleHitInfo hit, MaterialID material_id, SurfaceInteraction si, LightSample ls,
                             bool unreachable, float lt_ratio, uint depth)
{
    if (!light_tracing || unreachable || !(lt_ratio > 0.0))
        return 0.0;
    const float cos_here = abs(dot(si.normal_ws, ls.dir_ws));
    float ratio;
    if (ls.emitter_type == EmitterType::environment) {
        const float pe = environment_emission_pdf(scene_light_sampler(), light_emission, ls.dir_ws);
        ratio = pe * cos_here / ls.pdf;
    } else {
        const EmitterType type = ls.emitter_type;
        const float pe = emission_pdf(scene_light_sampler(), light_emission, type, ls.emitter_id, ls.pos_ws, ls.normal_ws,
                                      -ls.dir_ws);
        if (!(pe > 0.0))
            return 0.0;
        const float d2 = ls.distance * ls.distance;
        const bool point = type == EmitterType::analytic && all(ls.normal_ws == 0.0);
        const float light_area_pdf = point ? ls.pdf : ls.pdf * abs(dot(ls.normal_ws, ls.dir_ws)) / d2;
        if (!(light_area_pdf > 0.0))
            return 0.0;
        ratio = pe * cos_here / d2 / light_area_pdf;
    }
    return lt_ratio * light_tracer_reverse_pdf(hit, material_id, si, ls.dir_ws, depth) * ratio;
}

/// The light sample as a LightRecord (differentiable next-event terms, see light_radiance()).
LightRecord light_record(LightSample ls, float mis, float3 bsdf)
{
    LightRecord l = {};
    l.pdf = ls.pdf;
    l.weight = ls.weight;
    l.mis = mis;
    l.slot = NO_SLOT;
    l.emission = ls.weight * ls.pdf;
    l.bsdf = bsdf;
    bool directional = ls.distance >= float.MAX;
    if (ls.emitter_type == EmitterType::emissive_triangle) {
        EmissiveTriangle t = g_scene.emissive_geometry.get_triangle(EmissiveTriangleID(ls.emitter_id));
        l.kind = LIGHT_TRIANGLE;
        l.hit.gid = t.geometry_instance_id;
        l.hit.prim = ls.emitter_id - uint(g_scene.emissive_geometry.geometry_instance_to_triangle_id[uint(t.geometry_instance_id)]);
        float3 e1 = t.pos_ws[1] - t.pos_ws[0], e2 = t.pos_ws[2] - t.pos_ws[0], q = ls.pos_ws - t.pos_ws[0];
        float d11 = dot(e1, e1), d12 = dot(e1, e2), d22 = dot(e2, e2), q1 = dot(q, e1), q2 = dot(q, e2);
        float det = d11 * d22 - d12 * d12;
        l.hit.bary = float2(d22 * q1 - d12 * q2, d11 * q2 - d12 * q1) / det;
        return l;
    }
    if (ls.emitter_type == EmitterType::analytic)
        l.slot = light_slot_base + LIGHT_SLOTS * ls.emitter_id;
    l.kind = directional ? LIGHT_DIRECTIONAL : (dot(ls.normal_ws, ls.normal_ws) > 0 ? LIGHT_AREA : LIGHT_POINT);
    l.pos = directional ? ls.dir_ws : ls.pos_ws;
    l.normal = ls.normal_ws;
    return l;
}

bool visible_dir(float3 pos, float3 dir)
{
    HitRef h;
    return !intersect(pos, dir, 1e30, h);
}

/// Multiplies the medium weight of the segment that reached vertex k into the throughput terms of vertex k - 1.
void scale_segment(inout PathRecord rec, uint k, float3 factor)
{
    if (k == 0)
        return;
    rec.step_scale[k - 1] *= factor;
    rec.terms[k - 1].step *= factor;
}

/// Traces and records a camera path from `ray0` (the reference path tracer's path from that ray; the first segment is
/// a camera ray).
///
/// With `antithetic`, the BSDF of the first vertex is sampled twice, with sg and with an AntitheticSampleGenerator,
/// and `dual` selects the antithetic sample. The two samples choose the same lobe; rec.antithetic tells whether it is
/// a glossy reflection, for which the two paths (which share all other random numbers) are averaged. The choice is
/// the same for both paths, which keeps the average unbiased. Returns whether the camera ray hit a surface.
bool trace_path(Ray ray0, inout UniformSampleGenerator sg, out PathRecord rec, bool antithetic = false, bool dual = false)
{
    rec = {};
    rec.origin = ray0.origin;
    [ForceUnroll]
    for (uint i = 0; i <= MAX_DEPTH; i++)
        rec.step_scale[i] = float3(1.0);
    const uint max_depth = max_bounces + 1;

    // Path state (the reference path tracer's PathState).
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    float3 ray_origin = ray0.origin, ray_dir = ray0.dir;
    float scatter_pdf = 0.0;
    float3 scatter_normal = float3(0.0);
    bool delta = false;
    uint sampled_hemispheres = 0;
    InteriorList interiors = {};
    uint depth = 0;
    // Light tracing: N p_t1 / p_camera of the path prefix, without the light tracer's density of the direction towards
    // the previous vertex at the current one (see light_paths.slang); unreachable: the light tracer cannot generate it.
    float lt_ratio = 0.0;
    bool lt_unreachable = false;
    bool hit_surface = false;

    [loop]
    while (true) {
        // prepare_path_segment(): a distance to the next scattering event inside a homogeneous medium. (The first
        // segment keeps ray0's t_min: boundary-term paths start on an edge of the scene.)
        Ray ray = Ray(ray_origin, ray_dir, depth == 0 ? ray0.t_min : 0.0);
        bool volume_event = false;
        if (!interiors.is_empty()) {
            const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(interiors.active_medium());
            float distance;
            if (sampler.sample_distance(throughput, sg, distance) && isfinite(distance) && distance < ray.t_max) {
                ray.t_max = distance;
                volume_event = true;
            }
        }
        OpacityEvaluator opacity = OpacityEvaluator(sg.next_uint());
        TriangleHitInfo hit;
        if (!intersect_opacity(ray, opacity, hit)) {
            if (volume_event) {
                // handle_volume_event(): a scattering vertex in the medium.
                const HomogeneousMediumProperties medium = interiors.active_medium();
                const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(medium);
                const float3 event_weight = sampler.eval_event_weight(throughput, ray.t_max);
                throughput *= event_weight;
                ray_origin += ray_dir * ray.t_max;
                sampled_hemispheres = 0;
                delta = false;
                lt_unreachable = true;
                scatter_normal = float3(0.0);
                depth += 1;
                uint kv = depth - 1;
                rec.length = depth;
                rec.flags[kv] = VERTEX_VOLUME;
                rec.position[kv] = ray_origin;
                scale_segment(rec, kv, event_weight);
                if (!any(throughput > 0.0) || depth >= max_depth)
                    break;
                const HenyeyGreensteinPhaseFunction phase = HenyeyGreensteinPhaseFunction(clamp(medium.anisotropy, -0.999, 0.999));
                PhaseFunctionSample ps = {};
                if (!phase.sample(-ray_dir, sg, ps))
                    break;
                throughput *= ps.weight;
                scatter_pdf = ps.pdf;
                ray_dir = ps.wo_ws;
                rec.terms[kv].step = ps.weight;
                rec.delta_weight[kv + 1] = ps.weight;
                float rr;
                if (!any(throughput > 0.0) || !russian_roulette(depth, throughput, sg, rr)) {
                    rec.terms[kv].step = float3(0.0);
                    break;
                }
                rec.step_scale[kv] = float3(rr);
                rec.terms[kv].step *= rr;
                continue;
            }
            // handle_miss()
            uint k = depth;     // the escaped segment leaves vertex k - 1
            if (!interiors.is_empty()) {
                const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(interiors.active_medium());
                const float3 w = sampler.eval_surface_weight_to_infinity(throughput);
                throughput *= w;
                scale_segment(rec, k, w);
            }
            const bool primary_miss = depth == 0;
            const bool has_environment = g_scene.has_environment_lights();
            const bool nee_sampled = !delta && sampled_hemispheres != 0
                && in_sampled_hemispheres(scatter_normal, sampled_hemispheres, ray_dir);
            const bool handle_environment = has_environment && !primary_miss;
            rec.miss_dir = ray_dir;
            if ((has_environment && primary_miss) || handle_environment) {
                SurfaceInteraction scatter_si = scatter_interaction(ray_origin, scatter_normal, ray_dir);
                LightSampleContext ctx = hemisphere_light_context(scatter_si, sampled_hemispheres);
                float mis = 1.0;
                if (handle_environment && nee_sampled) {
                    float light_pdf = light_sampler.eval_environment_pdf(ctx, ray_dir);
                    mis = mis_heuristic(1, scatter_pdf, 1, light_pdf);
                }
                if (light_tracing && handle_environment) {
                    float ratio = nee_sampled && !delta ? light_sampler.eval_environment_pdf(ctx, ray_dir) / scatter_pdf : 0.0;
                    if (!lt_unreachable)
                        ratio += lt_ratio * environment_emission_pdf(scene_light_sampler(), light_emission, ray_dir);
                    mis = 1.0 / (1.0 + ratio);
                }
                const float3 e = g_scene.light_system.eval_environment_radiance(ctx, ray_dir);
                radiance += throughput * e * mis;
                if (primary_miss) {
                    rec.miss_scale = float3(1.0);
                    rec.miss_radiance = e;
                } else {
                    rec.miss_scale = mis * (delta ? rec.delta_weight[k] : float3(1.0));
                    rec.miss_radiance = e * rec.miss_scale;
                    rec.miss_pdf = delta ? 0.0 : scatter_pdf;
                    rec.terms[k - 1].step *= e * mis;
                }
            } else if (!primary_miss) {
                rec.terms[k - 1].step = float3(0.0);
            }
            break;
        }

        // handle_hit()
        SurfaceInteraction si = g_scene.make_triangle_surface_interaction(hit, -ray_dir);
        {
            const float segment_length = length(si.position_ws - ray_origin);
            const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(interiors.active_medium());
            float3 w;
            if (sampler.has_scattering())
                w = sampler.eval_surface_weight(throughput, segment_length);
            else
                w = sampler.eval_transmittance(segment_length);
            throughput *= w;
            scale_segment(rec, depth, w);
        }
        const MaterialID material_id = g_scene.get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        const MaterialHeader header = g_scene.material_system.get_material_data(material_id).header;
        // Back faces of one-sided materials terminate the path.
        if (!(si.front_facing || is_set(header.flags, MaterialFlags::two_sided))) {
            if (depth > 0)
                rec.terms[depth - 1].step = float3(0.0);
            break;
        }
        si.exterior_ior = interiors.exterior_ior(material_id, si.front_facing);
        IMaterialInstance mi = hit_material_instance(material_id, si);
        const bool is_thin_walled = is_set(header.flags, MaterialFlags::thin_walled);
        const bool is_unlit = is_set(header.flags, MaterialFlags::unlit);

        depth += 1;
        const bool is_primary_hit = depth == 1;
        const uint k = depth - 1;
        hit_surface = true;
        rec.length = depth;
        rec.hit[k].gid = hit.geometry_instance_id;
        rec.hit[k].prim = hit.primitive_index;
        rec.hit[k].bary = hit.barycentrics;
        rec.position[k] = si.position_ws;
        rec.exterior_ior[k] = si.exterior_ior;

        if (light_tracing) {
            // Area measure at this vertex: the camera's density of the primary hit, or the cosine of the arrival.
            const float cos_in = abs(dot(si.normal_ws, ray_dir));
            if (is_primary_hit) {
                const float3 d = si.position_ws - camera.position;
                const float dist2 = dot(d, d);
                const float w2 = dot(camera.image_w, camera.image_w);
                const float c = dot(normalize(d), camera.image_w) * rsqrt(w2);
                const float pixel_area = 4.0 * length(camera.image_u) * length(camera.image_v)
                    / float(camera.dims.x * camera.dims.y);
                const float camera_area_pdf = w2 / (pixel_area * c * c * c * dist2) * cos_in;
                lt_ratio = camera_area_pdf > 0.0 ? light_mis_path_count / camera_area_pdf : 0.0;
                if (camera.aperture_radius > 0.0 && camera.focus_distance > 0.0)
                    lt_unreachable = true;
            } else {
                lt_ratio = cos_in > 0.0 ? lt_ratio / cos_in : 0.0;
            }
        }

        if (is_unlit) {
            const float3 le = mi.eval_emissive_radiance();
            radiance += throughput * le;
            rec.emission[k] = le;
            rec.mis[k] = 1.0;
            rec.terms[k].emission = le;
            break;
        }

        // Emission found by the path.
        const bool nee_sampled = !delta && sampled_hemispheres != 0
            && in_sampled_hemispheres(scatter_normal, sampled_hemispheres, ray_dir);
        const bool apply_emitter_mis = !is_primary_hit && nee_sampled;
        if (g_scene.has_emissive_triangles() && any(mi.is_emissive())) {
            float mis = 1.0;
            if (apply_emitter_mis) {
                SurfaceInteraction scatter_si = scatter_interaction(ray_origin, scatter_normal, ray_dir);
                LightSampleContext ctx = hemisphere_light_context(scatter_si, sampled_hemispheres);
                EmissiveTriangleID triangle_id = g_scene.emissive_geometry.get_triangle_id(hit.geometry_instance_id,
                                                                                           hit.primitive_index);
                float light_pdf = light_sampler.eval_emissive_pdf(ctx, triangle_id, si.position_ws, si.normal_ws);
                mis = mis_heuristic(1, scatter_pdf, 1, light_pdf);
                if (light_tracing)
                    mis = 1.0 / (1.0 + light_pdf / scatter_pdf + light_tracer_emitter_ratio(hit, si, ray_dir, lt_ratio,
                                                                                           lt_unreachable));
            } else if (light_tracing && !is_primary_hit) {
                mis = 1.0 / (1.0 + light_tracer_emitter_ratio(hit, si, ray_dir, lt_ratio, lt_unreachable));
            }
            const float3 le = mi.eval_emissive_radiance();
            radiance += throughput * le * mis;
            rec.emission[k] = le;
            rec.mis[k] = mis;
            rec.terms[k].emission = le * mis;
        }

        if (depth >= max_depth)
            break;

        // Next-event estimation (sample_direct_lighting()).
        sampled_hemispheres = material_nee_hemispheres(mi.get_lobe_types(si))
            & nee_visibility_hemispheres(si.front_facing, material_id, is_thin_walled, interiors);
        if (sampled_hemispheres != 0) {
            LightSample ls = {};
            LightSampleContext ctx = hemisphere_light_context(si, sampled_hemispheres);
            if (light_sampler.sample(ctx, sg, ls)
                && in_sampled_hemispheres(si.front_facing ? si.normal_ws : -si.normal_ws, sampled_hemispheres, ls.dir_ws)) {
                OpacityEvaluator shadow_opacity = OpacityEvaluator(sg.next_uint());
                if (!occluded_opacity(light_visibility_ray(si, ls), shadow_opacity)) {
                    float mis = 1.0;
                    float3 f;
                    if (!ls.nee_only) {
                        const BSDFEval be = mi.eval_with_pdf(si, ls.dir_ws, sg);
                        f = be.value;
                        if (light_tracing) {
                            float ratio = be.pdf / ls.pdf
                                + light_tracer_nee_ratio(hit, material_id, si, ls, lt_unreachable, lt_ratio, depth);
                            mis = 1.0 / (1.0 + ratio);
                        } else {
                            mis = mis_heuristic(1, ls.pdf, 1, be.pdf);
                        }
                    } else {
                        f = mi.eval(si, ls.dir_ws, sg);
                        if (light_tracing)
                            mis = 1.0 / (1.0 + light_tracer_nee_ratio(hit, material_id, si, ls, lt_unreachable, lt_ratio,
                                                                     depth));
                    }
                    radiance += throughput * f * ls.weight * mis;
                    rec.light[k] = light_record(ls, mis, f);
                    rec.terms[k].nee = f * ls.weight * mis;
                }
            }
        }

        // BSDF sample.
        MaterialSample ms = {};
        bool sampled;
        if (k == 0 && antithetic) {
            AntitheticSampleGenerator other_sg = AntitheticSampleGenerator(sg);
            MaterialSample other = {};
            bool other_sampled = mi.sample(si, other_sg, other);
            sampled = mi.sample(si, sg, ms);
            MaterialSample lobe = sampled ? ms : other, other_lobe = other_sampled ? other : ms;
            // Symmetric in the two samples, so that the pair stays unbiased also when the sampled direction (not only
            // the lobe) decides between reflection and transmission (SpongeCakeMaterial's single scattering).
            rec.antithetic = (sampled || other_sampled) && lobe.has_flag(BSDFFlags::glossy_reflection)
                && other_lobe.has_flag(BSDFFlags::glossy_reflection);
            if (dual) {
                ms = other;
                sampled = other_sampled;
                sg = other_sg.sg;
            }
        } else {
            sampled = mi.sample(si, sg, ms);
        }
        if (!sampled)
            break;
        if (!is_thin_walled && ms.has_flag(BSDFFlags::transmission)) {
            if (cross_interior(si.front_facing, material_id, header.nested_priority, mi, interiors)
                != InteriorListUpdateResult::success)
                break;
        }
        if (light_tracing) {
            // Light-tracing ratio of the next vertex (divided by its arrival cosine there).
            const float cos_out = abs(dot(si.normal_ws, ms.wo_ws));
            if (ms.has_flag(BSDFFlags::delta)) {
                lt_ratio *= cos_out;
                if (is_primary_hit)
                    lt_unreachable = true;     // the light tracer cannot connect a delta vertex
            } else {
                lt_ratio *= light_tracer_reverse_pdf(hit, material_id, si, ms.wo_ws, depth) * cos_out / ms.pdf;
            }
        }

        // generate_scatter_ray()
        throughput *= ms.weight;
        scatter_pdf = ms.pdf;
        scatter_normal = stored_scatter_normal(si.front_normal_ws());
        delta = ms.has_flag(BSDFFlags::delta);
        ray_origin = offset_ray_origin(si.position_ws, dot(ms.wo_ws, si.normal_ws) >= 0.0 ? si.normal_ws : -si.normal_ws);
        ray_dir = ms.wo_ws;
        rec.delta_weight[k + 1] = ms.weight;
        rec.pdf[k + 1] = delta ? 0.0 : ms.pdf;
        rec.terms[k].step = ms.weight;
        float rr;
        if (!any(throughput > 0.0) || !russian_roulette(depth, throughput, sg, rr)) {
            rec.terms[k].step = float3(0.0);
            break;
        }
        rec.step_scale[k] = float3(rr);
        rec.terms[k].step *= rr;
    }
    rec.radiance = radiance;
    return hit_surface;
}

/// p_t1 / p_s0 of an emissive triangle found by a scattered ray (lt_ratio is already divided by the arrival cosine).
float light_tracer_emitter_ratio(TriangleHitInfo hit, SurfaceInteraction si, float3 ray_dir, float lt_ratio,
                                 bool unreachable)
{
    if (!light_tracing || unreachable)
        return 0.0;
    const EmissiveTriangleID triangle_id = g_scene.emissive_geometry.get_triangle_id(hit.geometry_instance_id,
                                                                                     hit.primitive_index);
    const float pe = emission_pdf(scene_light_sampler(), light_emission, EmitterType::emissive_triangle,
                                  uint(triangle_id), si.position_ws, si.normal_ws, -ray_dir);
    return lt_ratio * pe;
}

/// A path from `origin` along `dir` (boundary terms: origin may lie on an edge of the scene, so the first segment
/// starts at ray_epsilon; a camera ray of the reference path tracer's length).
bool trace_path(float3 origin, float3 dir, inout UniformSampleGenerator sg, out PathRecord rec)
{
    return trace_path(Ray(origin, dir, ray_epsilon), sg, rec);
}

/// Importance of a sensor point, excluding the surface cosine (pinhole, box filter).
float sensor_importance(float3 p)
{
    float3 d = p - camera.position;
    float c = dot(normalize(d), camera_forward());
    float w = length(camera.image_w);
    float pixel_area = 4 * length(camera.image_u) * length(camera.image_v) / (w * w * camera.dims.x * camera.dims.y);
    return 1 / (pixel_area * dot(d, d) * c * c * c);
}

[Differentiable]
float camera_weight(Vertex v)
{
    float3 d = v.p - camera.position;
    float c = dot(normalize(d), camera_forward());
    float g = abs(dot(v.n, normalize(d))) / (dot(d, d) * c * c * c);
    return ratio(g, detach(g)) * v.J;
}

/// Differentiable terms of path vertex k (their values are rec.terms[k]). With from_camera, vertex 0
/// includes the camera Jacobian (value 1). The vertex's surface and BSDF are set up once, and the BSDF has
/// a single call site, so its derivative code is instantiated once. Volume vertices (and the throughput through media,
/// Russian roulette) are detached factors.
[Differentiable]
VertexTerms vertex_terms(float theta, no_diff PathRecord rec, no_diff uint k, no_diff bool from_camera)
{
    if (is_volume_vertex(rec, k))
        return { float3(0), float3(0), rec.terms[k].step };
    Vertex prev;
    if (k == 0 || is_volume_vertex(rec, k - 1)) {
        prev.p = k == 0 ? rec.origin : rec.position[k - 1];
        prev.n = float3(0);
        prev.J = 1;
    } else {
        prev = load_vertex(theta, rec.hit[k - 1]);
    }
    Vertex cur = load_vertex(theta, rec.hit[k]);
    float scale = k == 0 && from_camera ? camera_weight(cur) : 1.0;
    Surface s = no_diff make_surface(rec.hit[k], normalize(detach(prev.p - cur.p)), rec.exterior_ior[k]);
    VertexTerms t;
    t.emission = eval_emission(theta, s, cur.n, prev.p - cur.p, rec.emission[k]) * rec.mis[k] * scale;
    // The BSDF is evaluated towards the light sample and towards the next vertex or the environment.
    float3 wo_light = float3(0), w_light = float3(0);
    if (rec.light[k].kind != LIGHT_NONE)
        w_light = light_radiance(theta, cur, rec.light[k], wo_light);
    float3 wo_next = float3(0), w_next = float3(0), other_next = float3(0), step = float3(0);
    bool scattered = false;
    if (k + 1 < rec.length) {
        if (rec.pdf[k + 1] > 0) {
            if (is_volume_vertex(rec, k + 1)) {
                // Scattered into a medium: the direction is fixed (no geometric derivative).
                wo_next = normalize(rec.position[k + 1] - detach(cur.p));
                w_next = float3(1.0 / rec.pdf[k + 1]);
            } else {
                Vertex next = load_vertex(theta, rec.hit[k + 1]);
                float g = geometric(cur.p, next);
                wo_next = normalize(next.p - cur.p);
                w_next = ratio(g * next.J, detach(rec.pdf[k + 1] * g));
            }
            other_next = rec.delta_weight[k + 1] * rec.pdf[k + 1];
            scattered = true;
        } else {
            step = rec.delta_weight[k + 1] * delta_scale(theta, s, normalize(prev.p - cur.p));
        }
    } else if (any(rec.miss_scale > 0)) {
        // The environment radiance seen by the escaped direction changes with the environment map.
        float3 e = env_map_radiance(theta, rec.miss_dir);
        float3 miss = rec.miss_radiance + (e - detach(e)) * rec.miss_scale;
        if (rec.miss_pdf > 0) {
            wo_next = rec.miss_dir;
            w_next = miss / rec.miss_pdf;
            other_next = rec.delta_weight[k + 1] * rec.miss_pdf;
            scattered = true;
        } else {
            step = miss * delta_scale(theta, s, normalize(prev.p - cur.p));
        }
    }
    float3 f_light, f_next;
    eval_bsdf(theta, s, normalize(prev.p - cur.p), wo_light, wo_next, rec.light[k].bsdf, other_next, f_light, f_next);
    t.nee = f_light * w_light * scale;
    t.step = (scattered ? f_next * w_next : step) * rec.step_scale[k] * scale;
    return t;
}

/// suffix[k] = emission + nee + step * suffix[k + 1] with suffix[length] = 1: the radiance that vertices
/// k, k + 1, ... send towards vertex k - 1, relative to the throughput reaching vertex k. Returns the path
/// value.
float3 path_terms(PathRecord rec, out float3 suffix[MAX_DEPTH + 2])
{
    suffix[rec.length] = 1;
    for (int k = int(rec.length) - 1; k >= 0; k--)
        suffix[k] = rec.terms[k].emission + rec.terms[k].nee + rec.terms[k].step * suffix[k + 1];
    return rec.length == 0 ? rec.miss_radiance : suffix[0];
}

float3 path_value(PathRecord rec)
{
    float3 suffix[MAX_DEPTH + 2];
    return path_terms(rec, suffix);
}

/// Forward mode returns (value, derivative); backward mode accumulates `weight * dL/dvalue`. The path
/// value is a sum of products of vertex terms, so each vertex is differentiated on its own, weighted by
/// the throughput before it and, for the throughput term, the radiance after it.
float3 differentiate_path(PathRecord rec, bool from_camera, float3 weight, out float3 derivative)
{
    float3 suffix[MAX_DEPTH + 2];
    float3 value = path_terms(rec, suffix);
    derivative = 0;
    if (rec.length == 0 && from_camera && any(rec.miss_scale > 0)) {
        // The camera sees the environment map.
        if (backward) {
            var theta = diffPair(0.0);
            bwd_diff(env_map_radiance)(theta, rec.miss_dir, weight);
        } else {
            derivative = fwd_diff(env_map_radiance)(diffPair(0.0, 1.0), rec.miss_dir).d;
        }
    }
    // Bound this sample's vertex gradients relative to its contribution (see max_relative_vertex_derivative).
    vertex_derivative_limit = backward && max_relative_vertex_derivative > 0
        ? max(max_relative_vertex_derivative * length(weight * value), 1e-30) : 0.0;
    float3 prefix = 1;
    for (uint k = 0; k < rec.length; k++) {
        if (backward) {
            VertexTerms.Differential d;
            d.emission = weight * prefix;
            d.nee = weight * prefix;
            d.step = weight * prefix * suffix[k + 1];
            var theta = diffPair(0.0);
            bwd_diff(vertex_terms)(theta, rec, k, from_camera, d);
        } else {
            let d = fwd_diff(vertex_terms)(diffPair(0.0, 1.0), rec, k, from_camera).d;
            derivative += prefix * (d.emission + d.nee + d.step * suffix[k + 1]);
        }
        prefix *= rec.terms[k].step;
    }
    vertex_derivative_limit = 0.0;
    return value;
}
