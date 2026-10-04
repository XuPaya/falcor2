// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Constants, shader parameters and link-time constants.

static const uint MAX_DEPTH = 8;
static const uint NO_SLOT = 0xffffffff;
/// Parameter slots per material. StandardMaterial: base color (0-2), metallic (3), roughness (4),
/// ior (5), emissive factor (6-8), diffuse transmission (9), specular transmission (10), transmission
/// color (11-13), normal map scale (14). OpenPBRMaterial: its attributes in declaration order (44 slots, see
/// openpbr.slang) and the normal map scale (44). Other materials are evaluated without derivatives.
static const uint MATERIAL_SLOTS = 48;
/// Parameter slots per light, in LightID order: emission (intensity or radiance, 0-2) and translation (3-5)
/// of analytic lights; intensity (0-2) and rotation (a world-space rotation vector, 3-5) of environment maps.
static const uint LIGHT_SLOTS = 6;
static const float PHOTON_EMISSIONS = 8192;

public Camera camera;
public uint max_bounces;
public float ray_epsilon;
/// Link-time constant selecting backward or forward mode, so each kernel contains one of them.
public extern static const bool backward;
public uint seed;
public uint standard_type_id;
public uint openpbr_type_id;
/// Link-time constant: whether the scene has OpenPBR materials. Without them, the differentiable
/// OpenPBR evaluation, which is expensive to compile, is compiled out.
public extern static const bool has_openpbr;
/// Link-time constant: whether materials have normal maps. Without them, shading frames are constant, which
/// keeps the BSDF derivatives much cheaper.
public extern static const bool normal_maps;
public uint light_slot_base;
public ParameterBlock<PowerLightSampler> light_sampler;

/// Object-space vertex tangents and parameter tangents (forward mode).
public StructuredBuffer<float3> vertex_tangent;
public StructuredBuffer<float> param_tangent;
/// Object-space vertex gradients (float3 per vertex) and parameter gradients (backward mode).
public RWByteAddressBuffer vertex_grad;
public RWByteAddressBuffer param_grad;
/// Emission property (intensity or radiance) of each light, in LightID order.
public StructuredBuffer<float3> light_value;
/// First texel parameter slot of each texture by TextureHandle.texture_index, or NO_SLOT: four slots (RGBA)
/// per texel, rows from the top.
public StructuredBuffer<uint> texel_slots;
public uint texel_slot_count;
/// Light ID of the (first) environment map light, whose radiance is differentiated, or NO_SLOT.
public uint env_map_id;

/// Per-pixel outputs. Forward mode writes derivative images, backward mode reads dL/dI.
public RWStructuredBuffer<float3> primal;
public RWStructuredBuffer<float3> interior;
public RWStructuredBuffer<float3> pixel_boundary;
public RWByteAddressBuffer primary_boundary;
public RWStructuredBuffer<float3> secondary_boundary;
public StructuredBuffer<float3> dl_di;
