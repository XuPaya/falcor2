// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Scene parameters with derivative slots: material and light factors (param()) and texels
// (texel_bilinear()). Forward mode reads their tangents, backward mode accumulates their gradients.

/// Lanes that add to the same address (shared parameters) are summed first, avoiding atomic contention.
void atomic_add(RWByteAddressBuffer buffer, uint address, float value)
{
    value = isfinite(value) ? value : 0.0;
    if (WaveActiveAllEqual(address)) {
        value = WaveActiveSum(value);
        if (!WaveIsFirstLane())
            return;
    }
    if (value != 0)
        buffer.InterlockedAddF32(address, value);
}

uint offset_slot(uint slot, uint k) { return slot == NO_SLOT ? NO_SLOT : slot + k; }

/// Scene parameter (material or light) with derivative slot.
[Differentiable, ForwardDerivative(fwd_param), BackwardDerivative(bwd_param)]
float param(float theta, no_diff float value, no_diff uint slot)
{
    return value;
}

DifferentialPair<float> fwd_param(DifferentialPair<float> theta, float value, uint slot)
{
    return diffPair(value, slot == NO_SLOT ? 0.0 : theta.d * param_tangent[slot]);
}

void bwd_param(inout DifferentialPair<float> theta, float value, uint slot, float d)
{
    if (slot != NO_SLOT)
        atomic_add(param_grad, 4 * slot, d);
    theta = diffPair(theta.p, 0.0);
}

[Differentiable]
float3 param3(float theta, no_diff float3 value, no_diff uint slot)
{
    return float3(
        param(theta, value.x, slot),
        param(theta, value.y, offset_slot(slot, 1)),
        param(theta, value.z, offset_slot(slot, 2))
    );
}

/// Texture handle without UDIM tiles and uv origin correction: texel (x, y) covers uv in [x, x + 1] / width x
/// [y, y + 1] / height.
typedef TextureHandle<float4, uint(TextureHandleFlags::none)> RawTextureHandle;

/// Texel slots and size of a texture: (first slot or NO_SLOT, width, height).
uint3 texel_layout(RawTextureHandle h)
{
    if (!h.is_valid() || h.texture_index >= texel_slot_count || texel_slots[h.texture_index] == NO_SLOT)
        return uint3(NO_SLOT, 0, 0);
    uint width, height, mip_count;
    h.get_dimensions(0, width, height, mip_count);
    return uint3(texel_slots[h.texture_index], width, height);
}

/// Texture value at uv, as sampled at mip level 0, with the derivatives of bilinear filtering of a wrapped texture
/// (falcor2's default sampler) for its texels and for uv: every texel channel is a parameter (texel_slots).
/// Textures without texel parameters have no derivatives (`fallback` for invalid handles).
[Differentiable]
float4 texel_bilinear(float theta, no_diff RawTextureHandle h, float2 uv, no_diff float4 fallback)
{
    float4 sampled = no_diff h.sample(ExplicitLodSampler(), detach(uv), fallback);
    uint3 layout = no_diff texel_layout(h);
    if (layout.x == NO_SLOT)
        return sampled;
    int2 size = int2(layout.yz);
    float2 p = uv * float2(size) - 0.5;
    int2 i = int2(floor(detach(p)));
    float2 f = p - float2(i);
    float4 result = float4(0.0);
    [ForceUnroll]
    for (uint k = 0; k < 4; k++) {
        int2 t = ((i + int2(k & 1, k >> 1)) % size + size) % size;
        float4 value = no_diff h.load(t);
        uint slot = layout.x + 4 * uint(t.y * size.x + t.x);
        float weight = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        result += weight * float4(param(theta, value.x, slot), param(theta, value.y, slot + 1),
                                  param(theta, value.z, slot + 2), param(theta, value.w, slot + 3));
    }
    // The sampler's value (its bilinear weights have limited subtexel precision, which matters for the normals
    // of sharp lobes) and the exact derivative.
    return sampled + (result - detach(result));
}

/// Value of a material texture (handle bits `handle`) at uv: texel_bilinear(), or `sampled`, its value sampled by
/// the material, for textures without texel parameters (such as UDIM textures).
[Differentiable]
float4 texture_param(float theta, no_diff uint handle, no_diff float2 uv, no_diff float4 sampled)
{
    RawTextureHandle raw = RawTextureHandle(handle);
    uint3 layout = no_diff texel_layout(raw);
    if (layout.x == NO_SLOT)
        return sampled;
    // Material textures correct the uv origin (TextureHandleFlags::uv_origin_correction).
    return texel_bilinear(theta, raw, FALCOR_TEXTURE_COORDINATE_ORIGIN_LOWER_LEFT ? float2(uv.x, 1.0 - uv.y) : uv,
                          sampled);
}

[Differentiable]
float channel(float4 v, no_diff uint c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }
