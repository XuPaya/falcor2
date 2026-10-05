// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Camera paths: records of their vertices and light samples, and the differentiable terms of their
// vertices.

/// Terms of path vertex k in material form: the emission towards the previous vertex, the next-event
/// estimate, and the throughput to vertex k + 1 or, at the last vertex, the environment radiance it sees
/// (step). See vertex_terms().
struct VertexTerms : IDifferentiable
{
    float3 emission;
    float3 nee;
    float3 step;
};

/// Detached path: vertex hits, sampling pdfs and MIS weights recorded by the primal tracer.
struct PathRecord
{
    uint length;
    float3 origin;
    HitRef hit[MAX_DEPTH + 1];
    /// Solid-angle pdf of the direction that generated hit[i] (i > 0), or zero for delta lobes.
    float pdf[MAX_DEPTH + 1];
    /// Sample weight (BSDF value over pdf) of the direction that generated hit[i], or of the last
    /// sampled direction (i = length) when it escapes.
    float3 delta_weight[MAX_DEPTH + 1];
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
    /// Whether the first vertex sampled a glossy reflection, for which the path has an antithetic partner (with
    /// antithetic sampling, see trace_path()).
    bool antithetic;
};

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

LightRecord sample_light(Surface s, IMaterialInstance mi, float3 pos, inout UniformSampleGenerator sg)
{
    LightRecord l = {};
    LightSample ls;
    if (!light_sampler.sample(LightSampleContext(s.si), sg, ls) || ls.pdf <= 0)
        return l;
    bool directional = ls.distance >= float.MAX;
    if (directional ? !visible_dir(pos, ls.dir_ws) : !visible(pos, ls.pos_ws))
        return l;
    UniformSampleGenerator eval_sg = UniformSampleGenerator(0);
    BSDFEval bsdf = mi.eval_with_pdf(s.si, ls.dir_ws, eval_sg);
    l.mis = mis_weight(ls.pdf, ls.nee_only ? 0 : bsdf.pdf);
    l.pdf = ls.pdf;
    l.weight = ls.weight;
    l.slot = NO_SLOT;
    l.emission = ls.weight * ls.pdf;
    l.bsdf = bsdf.value;
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

/// Traces and records a path whose first hit is found from `origin` along `dir`.
///
/// With `antithetic`, the BSDF of the first vertex is sampled twice, with sg and with an AntitheticSampleGenerator,
/// and `dual` selects the antithetic sample. The two samples choose the same lobe; rec.antithetic tells whether it is
/// a glossy reflection, for which the two paths (which share all other random numbers) are averaged. The choice is
/// the same for both paths, which keeps the average unbiased.
bool trace_path(float3 origin, float3 dir, inout UniformSampleGenerator sg, out PathRecord rec, bool antithetic = false,
                bool dual = false)
{
    rec = {};
    rec.origin = origin;
    HitRef h;
    if (!intersect(origin, dir, 1e30, h)) {
        LightSampleContext ctx = {};
        ctx.pos_ws = origin;
        rec.miss_radiance = g_scene.light_system.eval_environment_radiance(ctx, dir);
        rec.miss_scale = 1;
        rec.miss_dir = dir;
        return false;
    }
    float3 prev = origin;
    rec.mis[0] = 1;
    for (uint depth = 0; depth <= max_bounces; depth++) {
        rec.hit[depth] = h;
        rec.length = depth + 1;
        Vertex v = load_vertex(0, h);
        Surface s = make_surface(h, normalize(prev - v.p));
        IMaterialInstance mi = material_instance(s);
        rec.emission[depth] = mi.eval_emissive_radiance();
        rec.terms[depth].emission = dot(v.n, prev - v.p) > 0 ? rec.emission[depth] * rec.mis[depth] : float3(0);
        if (depth == max_bounces)
            break;
        LightRecord l = sample_light(s, mi, v.p, sg);
        rec.light[depth] = l;
        rec.terms[depth].nee = l.bsdf * l.weight * l.mis;
        MaterialSample ms = {};
        bool sampled;
        if (depth == 0 && antithetic) {
            AntitheticSampleGenerator other_sg = AntitheticSampleGenerator(sg);
            MaterialSample other = {};
            bool other_sampled = mi.sample(s.si, other_sg, other);
            sampled = mi.sample(s.si, sg, ms);
            MaterialSample lobe = sampled ? ms : other;
            rec.antithetic = (sampled || other_sampled) && lobe.has_flag(BSDFFlags::glossy_reflection);
            if (dual) {
                ms = other;
                sampled = other_sampled;
                sg = other_sg.sg;
            }
        } else {
            sampled = mi.sample(s.si, sg, ms);
        }
        if (!sampled)
            break;
        bool delta = ms.has_flag(BSDFFlags::delta);
        HitRef next;
        rec.delta_weight[depth + 1] = ms.weight;
        if (!intersect(v.p, ms.wo_ws, 1e30, next)) {
            LightSampleContext ctx = LightSampleContext(s.si);
            float light_pdf = delta ? 0 : light_sampler.eval_environment_pdf(ctx, ms.wo_ws);
            rec.miss_scale = mis_weight(ms.pdf, light_pdf) * (delta ? ms.weight : float3(1));
            rec.miss_radiance = g_scene.light_system.eval_environment_radiance(ctx, ms.wo_ws) * rec.miss_scale;
            rec.miss_dir = ms.wo_ws;
            rec.miss_pdf = delta ? 0 : ms.pdf;
            rec.terms[depth].step = delta ? rec.miss_radiance : ms.weight * rec.miss_radiance;
            break;
        }
        rec.terms[depth].step = ms.weight;
        Vertex n = load_vertex(0, next);
        EmissiveTriangleID tri = g_scene.emissive_geometry.get_triangle_id(next.gid, next.prim);
        float light_pdf = delta ? 0 : light_sampler.eval_emissive_pdf(LightSampleContext(s.si), tri, n.p, n.n);
        rec.pdf[depth + 1] = delta ? 0 : ms.pdf;
        rec.mis[depth + 1] = mis_weight(ms.pdf, light_pdf);
        prev = v.p;
        h = next;
    }
    return true;
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
/// a single call site, so its derivative code is instantiated once.
[Differentiable]
VertexTerms vertex_terms(float theta, no_diff PathRecord rec, no_diff uint k, no_diff bool from_camera)
{
    Vertex prev;
    if (k == 0) {
        prev.p = rec.origin;
        prev.n = float3(0);
        prev.J = 1;
    } else {
        prev = load_vertex(theta, rec.hit[k - 1]);
    }
    Vertex cur = load_vertex(theta, rec.hit[k]);
    float scale = k == 0 && from_camera ? camera_weight(cur) : 1.0;
    Surface s = no_diff make_surface(rec.hit[k], normalize(detach(prev.p - cur.p)));
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
            Vertex next = load_vertex(theta, rec.hit[k + 1]);
            float g = geometric(cur.p, next);
            wo_next = normalize(next.p - cur.p);
            w_next = ratio(g * next.J, detach(rec.pdf[k + 1] * g));
            other_next = rec.delta_weight[k + 1] * rec.pdf[k + 1];
            scattered = true;
        } else {
            step = rec.delta_weight[k + 1];
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
            step = miss;
        }
    }
    float3 f_light, f_next;
    eval_bsdf(theta, s, normalize(prev.p - cur.p), wo_light, wo_next, rec.light[k].bsdf, other_next, f_light, f_next);
    t.nee = f_light * w_light * scale;
    t.step = (scattered ? f_next * w_next : step) * scale;
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
    if (rec.length == 0 && from_camera) {
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
