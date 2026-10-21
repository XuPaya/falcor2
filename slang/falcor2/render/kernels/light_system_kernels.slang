// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;

/// Flux a finite analytic light can deliver into the sphere (center, radius) when it lies outside it, as pbrt's
/// DistantLight::Phi = pi R^2 L for a light far away: the radiance (or intensity) times the light's projected area times
/// the solid angle of the sphere seen from the light (an upper bound: the largest cosine at the light over the sphere).
/// Returns `power` (the total emitted flux) for lights near or inside the sphere and for other lights.
static const float SCENE_BOUNDED_PI = 3.14159265358979;

float lum(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

float scene_bounded_power(LightSystem light_system, uint index, float power, float3 center, float radius)
{
    if (!(radius > 0.0))
        return power;
    LightData data = light_system.light_data[index];
    LightType type = LightType(reinterpret<uint>(data.header));
    float3 pos;
    float extent = 0.0;
    float emitted;      // luminance x projected area (area lights: x cosine bound below), or intensity
    float3 normal = float3(0.0);
    if (type == LightType::disk_light) {
        DiskLight disk = reinterpret<DiskLight>(data);
        pos = disk.center_ws;
        normal = disk.normal_ws;
        extent = max(length(disk.radius_u_ws), length(disk.radius_v_ws));
        emitted = lum(disk.radiance) * disk.surface_area;
    } else if (type == LightType::rect_light) {
        RectLight rect = reinterpret<RectLight>(data);
        pos = rect.center_ws;
        normal = rect.normal_ws;
        extent = length(rect.half_extent_u_ws) + length(rect.half_extent_v_ws);
        emitted = lum(rect.radiance) * rect.surface_area;
    } else if (type == LightType::sphere_light) {
        SphereLight sphere = reinterpret<SphereLight>(data);
        pos = sphere.center_ws;
        extent = sphere.radius_ws;
        emitted = lum(sphere.radiance) * SCENE_BOUNDED_PI * sphere.radius_ws * sphere.radius_ws;
    } else if (type == LightType::point_light) {
        pos = reinterpret<PointLight>(data).pos_ws;
        emitted = lum(reinterpret<PointLight>(data).intensity);
    } else {
        return power;
    }
    float d = length(center - pos);
    if (d <= radius + extent)
        return power;
    float sin2 = min(radius * radius / (d * d), 1.0);
    float solid_angle = 2.0 * SCENE_BOUNDED_PI * (1.0 - sqrt(1.0 - sin2));
    if (any(normal != 0.0))
        emitted *= saturate(dot(normal, (center - pos) / d) + radius / d);
    return min(power, emitted * solid_angle);
}

/// Compute the power estimate of each component light for importance sampling. With a scene bounding sphere
/// (scene_radius > 0), finite lights far outside it report the flux they can deliver into it (scene_bounded_power()).
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_light_powers(
    uint3 tid: SV_DispatchThreadID,
    uniform LightSystem light_system,
    uniform uint light_count,
    uniform float3 scene_center,
    uniform float scene_radius,
    RWStructuredBuffer<float> powers
)
{
    if (tid.x >= light_count)
        return;

    float power = light_system.get_light(LightID(tid.x)).power_estimate();
    power = scene_bounded_power(light_system, tid.x, power, scene_center, scene_radius);
    powers[tid.x] = power;
}
