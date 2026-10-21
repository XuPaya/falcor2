// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// Emission sampling of the scene's lights for light tracing (the light subpaths of bidirectional path tracing,
// Veach 1997, ch. 10), and the evaluation of its densities for multiple importance sampling against the camera
// paths' strategies. An emitter is chosen with the light sampler's emission selection (ILightSampler.sample_emitter),
// then a position and a direction:
// - finite analytic lights (disk, rect, sphere): position uniform over the area; direction uniform over the
//   projected disk of an emission target sphere when the light lies outside it (as pbrt samples distant lights over
//   the disk of the scene's bounding sphere; the target defaults to the scene's bounding sphere), else
//   cosine-weighted about the normal. Point lights: the position is a delta (density 1).
// - emissive triangles: position uniform, direction cosine-weighted about the front normal;
// - environment lights: direction from the light's own sampling, position uniform over the disk of the scene's
//   bounding sphere perpendicular to it (pbrt's infinite-light emission).
// Densities: pdf_pos (selection x area density; environment: selection / (pi R^2); point lights: selection) and
// pdf_dir (solid angle of the emitted direction; environment: of the direction towards the light).

public struct LightEmissionSettings
{
    /// Emission target sphere (finite lights outside it aim at it).
    public float3 target_center;
    public float target_radius;
    /// Bounding sphere of the scene (environment emission disk; far light rays start on it).
    public float3 scene_center;
    public float scene_radius;
};

public struct EmissionSample
{
    /// Start of the emitted ray (environment: on the scene disk).
    public float3 pos_ws;
    /// Emitter normal (zero for point and environment lights).
    public float3 normal_ws;
    /// Emitted direction (away from the light).
    public float3 dir_ws;
    /// Emitted radiance x |cos at the light| / (pdf_pos pdf_dir).
    public float3 weight;
    public float pdf_pos;
    public float pdf_dir;
    public EmitterType type;
    public uint index;
    /// Emissive triangles: barycentrics of pos_ws.
    public float3 barycentrics;
};

/// Density (solid angle) of the emission direction `dir` from `origin` (see the file comment).
public float emission_direction_pdf(LightEmissionSettings s, float3 origin, float3 normal, float3 dir)
{
    float3 to_center = s.target_center - origin;
    float d2 = dot(to_center, to_center);
    float r = s.target_radius;
    if (d2 > sqr(1.0001 * r)) {
        float3 axis = to_center * rsqrt(d2);
        float cos_axis = dot(dir, axis);
        if (!(cos_axis > 0.0))
            return 0.0;
        // Intersection with the target's disk (through its center, perpendicular to the axis).
        float t = dot(to_center, axis) / cos_axis;
        float3 p = origin + t * dir - s.target_center;
        if (dot(p, p) > r * r)
            return 0.0;
        return t * t / (float.PI * r * r * cos_axis);
    }
    if (any(normal != 0.0))
        return max(dot(normal, dir), 0.0) * float.INV_PI;
    return 0.25 * float.INV_PI;
}

bool sample_emission_direction(LightEmissionSettings s, float3 origin, float3 normal, float2 u, out float3 dir,
                               out float pdf)
{
    float3 to_center = s.target_center - origin;
    float d2 = dot(to_center, to_center);
    float r = s.target_radius;
    dir = {};
    pdf = 0.0;
    if (d2 > sqr(1.0001 * r)) {
        float3 axis = to_center * rsqrt(d2);
        Frame frame = Frame(axis);
        float2 disk = square_to_uniform_disk_concentric(u) * r;
        float3 v = s.target_center + disk.x * frame.tangent + disk.y * frame.bitangent - origin;
        float dist2 = dot(v, v);
        if (!(dist2 > 0.0))
            return false;
        dir = v * rsqrt(dist2);
        float cos_axis = dot(dir, axis);
        if (!(cos_axis > 0.0))
            return false;
        pdf = dist2 / (float.PI * r * r * cos_axis);
        return true;
    }
    if (any(normal != 0.0)) {
        float3 local = square_to_cosine_hemisphere_concentric(u);
        Frame frame = Frame(normal);
        dir = normalize(local.x * frame.tangent + local.y * frame.bitangent + local.z * frame.normal);
        pdf = local.z * float.INV_PI;
        return pdf > 0.0;
    }
    dir = square_to_uniform_sphere(u);
    pdf = 0.25 * float.INV_PI;
    return true;
}

LightType emission_light_type(LightData data) { return LightType(reinterpret<uint>(data.header)); }

/// Area of a finite analytic light (0: point or unsupported).
float analytic_light_area(uint light)
{
    LightData data = g_scene.light_system.light_data[light];
    LightType type = emission_light_type(data);
    if (type == LightType::disk_light)
        return reinterpret<DiskLight>(data).surface_area;
    if (type == LightType::rect_light)
        return reinterpret<RectLight>(data).surface_area;
    if (type == LightType::sphere_light)
        return reinterpret<SphereLight>(data).surface_area;
    return 0.0;
}

/// Samples an emitted ray of an emitter chosen by the light sampler.
public bool sample_light_emission<LS : ILightSampler>(LS sampler, LightEmissionSettings s, inout UniformSampleGenerator sg,
                                                      out EmissionSample e)
{
    e = {};
    EmitterType type;
    uint index;
    float pmf;
    if (!sampler.sample_emitter(next_float(sg), type, index, pmf) || !(pmf > 0.0))
        return false;
    e.type = type;
    e.index = index;
    float2 u_pos = next_float2(sg);
    float2 u_dir = next_float2(sg);
    if (type == EmitterType::analytic) {
        LightData data = g_scene.light_system.light_data[index];
        LightType light_type = emission_light_type(data);
        float3 radiance = float3(0.0);
        float area = 0.0;
        LightShaping shaping;
        float3 shaping_dir = float3(0.0);
        if (light_type == LightType::disk_light) {
            DiskLight disk = reinterpret<DiskLight>(data);
            float2 p = square_to_uniform_disk_polar(u_pos.yx);
            e.pos_ws = disk.center_ws + p.x * disk.radius_u_ws + p.y * disk.radius_v_ws;
            e.normal_ws = disk.normal_ws;
            radiance = disk.radiance;
            area = disk.surface_area;
            shaping = disk.shaping;
            shaping_dir = disk.normal_ws;
        } else if (light_type == LightType::rect_light) {
            RectLight rect = reinterpret<RectLight>(data);
            float2 o = 2.0 * u_pos - 1.0;
            e.pos_ws = rect.center_ws + o.x * rect.half_extent_u_ws + o.y * rect.half_extent_v_ws;
            e.normal_ws = rect.normal_ws;
            radiance = rect.radiance;
            area = rect.surface_area;
            shaping = rect.shaping;
            shaping_dir = rect.normal_ws;
        } else if (light_type == LightType::sphere_light) {
            SphereLight sphere = reinterpret<SphereLight>(data);
            float3 n = square_to_uniform_sphere(u_pos.yx);
            e.pos_ws = sphere.center_ws + sphere.radius_ws * n;
            e.normal_ws = n;
            radiance = sphere.radiance;
            area = sphere.surface_area;
            shaping = sphere.shaping;
            shaping_dir = sphere.dir_ws;
        } else if (light_type == LightType::point_light) {
            PointLight point = reinterpret<PointLight>(data);
            e.pos_ws = point.pos_ws;
            if (!sample_emission_direction(s, e.pos_ws, float3(0.0), u_dir, e.dir_ws, e.pdf_dir))
                return false;
            e.pdf_pos = pmf;
            e.weight = point.intensity * point.shaping.eval(point.dir_ws, e.dir_ws) / (pmf * e.pdf_dir);
            return any(e.weight > 0.0);
        } else {
            return false;
        }
        if (!(area > 0.0))
            return false;
        if (!sample_emission_direction(s, e.pos_ws, e.normal_ws, u_dir, e.dir_ws, e.pdf_dir))
            return false;
        float cos_light = dot(e.normal_ws, e.dir_ws);
        if (!(cos_light > 0.0))
            return false;
        e.pdf_pos = pmf / area;
        e.weight = radiance * shaping.eval(shaping_dir, e.dir_ws) * cos_light / (e.pdf_pos * e.pdf_dir);
        return any(e.weight > 0.0);
    }
    if (type == EmitterType::emissive_triangle) {
        EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(EmissiveTriangleID(index));
        if (!(triangle.area_ws > 0.0))
            return false;
        float3 bary = sample_triangle(u_pos);
        e.barycentrics = bary;
        e.pos_ws = triangle.get_pos_ws(bary);
        e.normal_ws = triangle.normal_ws;
        float3 local = square_to_cosine_hemisphere_concentric(u_dir);
        Frame frame = Frame(triangle.normal_ws);
        e.dir_ws = normalize(local.x * frame.tangent + local.y * frame.bitangent + local.z * frame.normal);
        e.pdf_dir = local.z * float.INV_PI;
        if (!(e.pdf_dir > 0.0))
            return false;
        e.pdf_pos = pmf / triangle.area_ws;
        let material = g_scene.get_material(triangle.material_id);
        SurfaceInteraction si = {};
        si.wi_ws = e.dir_ws;
        si.uv = triangle.get_uv(bary);
        float3 radiance = material.eval_emissive_radiance(si, ExplicitLodSampler());
        e.weight = radiance * local.z / (e.pdf_pos * e.pdf_dir);
        return any(e.weight > 0.0);
    }
    if (type == EmitterType::environment) {
        ILight light = g_scene.light_system.get_environment_light(index);
        LightSampleContext ctx = {};
        ctx.pos_ws = s.scene_center;
        LightSample ls = {};
        if (!light.sample(ctx, sg, ls) || !(ls.pdf > 0.0))
            return false;
        // ls.pdf is this light's own density; the emission density mixes the environment lights by selection.
        float3 to_light = ls.dir_ws;
        float3 radiance = g_scene.light_system.eval_environment_radiance(ctx, to_light);
        e.pdf_dir = environment_emission_dir_pdf(sampler, to_light);
        if (!(e.pdf_dir > 0.0))
            return false;
        float R = s.scene_radius;
        Frame frame = Frame(to_light);
        float2 disk = square_to_uniform_disk_concentric(u_pos) * R;
        e.pos_ws = s.scene_center + R * to_light + disk.x * frame.tangent + disk.y * frame.bitangent;
        e.dir_ws = -to_light;
        e.pdf_pos = 1.0 / (float.PI * R * R);
        e.weight = radiance / (e.pdf_pos * e.pdf_dir);
        return any(e.weight > 0.0);
    }
    return false;
}

/// Emission density of the direction towards the environment (selection-weighted mixture of the environment
/// lights' own densities).
public float environment_emission_dir_pdf<LS : ILightSampler>(LS sampler, float3 to_light)
{
    LightSampleContext ctx = {};
    float pdf = 0.0;
    for (uint i = 0; i < g_scene.light_system.environment_light_count; ++i)
        pdf += sampler.eval_emitter_pmf(EmitterType::environment, i)
            * g_scene.light_system.get_environment_light(i).eval_pdf(ctx, to_light);
    return pdf;
}

/// Emission densities pdf_pos x pdf_dir of a light vertex found by a camera path: finite analytic light `index`
/// (LightID) or emissive triangle `index` (EmissiveTriangleID) at pos with normal, emitting along dir_emitted
/// (towards the camera path). Point lights: pdf_pos = selection.
public float emission_pdf<LS : ILightSampler>(LS sampler, LightEmissionSettings s, EmitterType type, uint index,
                                               float3 pos_ws, float3 normal_ws, float3 dir_emitted)
{
    float pmf = sampler.eval_emitter_pmf(type, index);
    if (!(pmf > 0.0))
        return 0.0;
    if (type == EmitterType::analytic) {
        float area = analytic_light_area(index);
        LightData data = g_scene.light_system.light_data[index];
        if (emission_light_type(data) == LightType::point_light)
            return pmf * emission_direction_pdf(s, pos_ws, float3(0.0), dir_emitted);
        if (!(area > 0.0))
            return 0.0;
        return pmf / area * emission_direction_pdf(s, pos_ws, normal_ws, dir_emitted);
    }
    if (type == EmitterType::emissive_triangle) {
        EmissiveTriangle triangle = g_scene.emissive_geometry.get_triangle(EmissiveTriangleID(index));
        if (!(triangle.area_ws > 0.0))
            return 0.0;
        return pmf / triangle.area_ws * max(dot(triangle.normal_ws, dir_emitted), 0.0) * float.INV_PI;
    }
    return 0.0;
}

/// Emission densities pdf_pos x pdf_dir of the environment along `to_light` (the direction towards it).
public float environment_emission_pdf<LS : ILightSampler>(LS sampler, LightEmissionSettings s, float3 to_light)
{
    return environment_emission_dir_pdf(sampler, to_light) / (float.PI * s.scene_radius * s.scene_radius);
}
