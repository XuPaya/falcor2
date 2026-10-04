// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Light samples and their differentiable radiance: emissive triangles, analytic lights and the
// environment map.

static const uint LIGHT_NONE = 0;
static const uint LIGHT_TRIANGLE = 1;
static const uint LIGHT_AREA = 2;
static const uint LIGHT_POINT = 3;
static const uint LIGHT_DIRECTIONAL = 4;

struct LightRecord
{
    uint kind;
    /// Emissive triangle sample with fixed barycentrics.
    HitRef hit;
    /// Analytic sample position and normal, or direction for directional lights.
    float3 pos;
    float3 normal;
    /// Analytic lights: Li / pdf. Emissive triangles: pdf w.r.t. solid angle.
    float3 weight;
    float pdf;
    float mis;
    uint slot;
    /// Values for other materials (see eval_bsdf): emitted radiance of the sample and BSDF value.
    float3 emission;
    float3 bsdf;
};

float mis_weight(float a, float b) { return a > 0 || b > 0 ? a * a / (a * a + b * b) : 1.0; }

/// x / detach(d) for a detached positive denominator; zero where the denominator vanishes.
[Differentiable]
float ratio(float x, no_diff float d) { return d > 0 ? x / d : 0.0; }

[Differentiable]
float geometric(float3 x, Vertex y)
{
    float3 d = y.p - x;
    float d2 = dot(d, d);
    return abs(dot(y.n, d)) / (d2 * sqrt(d2));
}

EnvMapLight env_map_light(uint id) { return reinterpret<EnvMapLight>(g_scene.light_system.light_data[id]); }

/// Radiance of the environment map along dir (EnvMapLight.eval_environment_radiance()) with derivatives for
/// its intensity, rotation and texels; zero without an environment map.
[Differentiable]
float3 env_map_radiance(float theta, no_diff float3 dir)
{
    if (env_map_id == NO_SLOT)
        return float3(0);
    EnvMapLight light = no_diff env_map_light(env_map_id);
    uint slot = light_slot_base + LIGHT_SLOTS * env_map_id;
    // Rotating the map by a small rotation vector r turns the direction into the map by -r.
    float3 d = dir - cross(param3(theta, float3(0), slot + 3), dir);
    float2 uv = world_to_latlong_map(mul(transpose(light.world_from_object), d));
    // The radiance is proportional to the intensity property.
    no_diff float3 intensity = no_diff light_value[env_map_id];
    float3 scale = param3(theta, intensity, slot) / max(intensity, 1e-20) * light.scale;
    return texel_bilinear(theta, light.env_map_texture_handle, uv, float4(0.0)).rgb * scale;
}

/// Next-event estimate at `cur` without the BSDF: the direction `wo` to the light sample and its
/// MIS-weighted radiance over the sampling pdf.
[Differentiable]
float3 light_radiance(float theta, Vertex cur, no_diff LightRecord l, out float3 wo)
{
    if (l.kind == LIGHT_DIRECTIONAL) {
        // Environment lights: the sample's radiance (the sum of the environment lights) changes with the
        // environment map, its direction does not.
        wo = l.pos;
        float3 e = env_map_radiance(theta, l.pos);
        return (l.weight + (e - detach(e)) / l.pdf) * l.mis;
    }
    if (l.kind == LIGHT_TRIANGLE) {
        Vertex y = load_vertex(theta, l.hit);
        float g = geometric(cur.p, y);
        Surface light_surface = no_diff make_surface(l.hit, normalize(detach(cur.p - y.p)));
        wo = normalize(y.p - cur.p);
        return eval_emission(theta, light_surface, y.n, cur.p - y.p, l.emission) * ratio(g * y.J, detach(l.pdf * g))
            * l.mis;
    }
    // The sample moves with the light's translation (zero-valued parameters).
    float3 pos = l.pos + param3(theta, float3(0), offset_slot(l.slot, 3));
    wo = normalize(pos - cur.p);
    float falloff = 1;
    if (l.kind == LIGHT_AREA) {
        float3 d = pos - cur.p;
        falloff = abs(dot(l.normal, d)) / pow(dot(d, d), 1.5);
    } else if (l.kind == LIGHT_POINT) {
        falloff = 1 / dot(pos - cur.p, pos - cur.p);
    }
    float3 scale = 1;
    if (l.slot != NO_SLOT) {
        // Li is proportional to the light's emission property.
        no_diff float3 value = no_diff light_value[(l.slot - light_slot_base) / LIGHT_SLOTS];
        scale = param3(theta, value, l.slot) / max(value, 1e-20);
    }
    return l.weight * scale * ratio(falloff, detach(falloff)) * l.mis;
}
