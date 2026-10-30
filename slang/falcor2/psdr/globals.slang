// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Constants, shader parameters and link-time constants.

static const uint NO_SLOT = 0xffffffff;
/// Parameter slots per material. StandardMaterial: base color (0-2), metallic (3), roughness (4),
/// ior (5), emissive factor (6-8), diffuse transmission (9), specular transmission (10), transmission
/// color (11-13), normal map scale (14). OpenPBRMaterial: its attributes in declaration order (44 slots, see
/// openpbr.slang) and the normal map scale (44). SpongeCakeMaterial: albedo (0-2), thickness (3), roughness (4),
/// top albedo (5-7), top thickness (8), top roughness (9), base color (10-12), base weight (13), normal map scale
/// (14). Other materials are evaluated without derivatives.
static const uint MATERIAL_SLOTS = 52;
/// Parameter slots per light, in LightID order: emission (intensity or radiance, 0-2) and translation (3-5)
/// of analytic lights; intensity (0-2) and rotation (a world-space rotation vector, 3-5) of environment maps.
static const uint LIGHT_SLOTS = 6;

public Camera camera;
public uint max_bounces;
/// Russian roulette from this path depth on (the reference path tracer's rr_depth; path vertices so far); 0xffffffff:
/// off.
public uint rr_depth;
public float ray_epsilon;
/// Backward mode: each path sample's interior-term vertex gradient (per world_vertex call) is clamped to
/// max_relative_vertex_derivative x |dL/dI x path value| (per unit length; 0: off). Sharp glossy lobes (mirror-
/// like coats, low roughness) make the material-form vertex derivative heavy-tailed (single samples 10^3-10^4 x
/// the mean); clamping these rare samples trades a small bias for bounded variance.
public float max_relative_vertex_derivative;
/// Photons per secondary-edge photon pass (PSDR.photon_count).
public uint photon_count;
/// Thread-local bound of the current path sample's vertex gradients (set by differentiate_path; 0: none, e.g.
/// in the boundary kernels).
static float vertex_derivative_limit = 0.0;
/// Link-time constant selecting backward or forward mode, so each kernel contains one of them.
public extern static const bool backward;
/// Use inline traversal when available; otherwise the tracing entry points run as ray-generation shaders.
public extern static const bool USE_RAY_QUERY = true;
// Keep absent reverse-mode material implementations out of Vulkan programs.
public extern static const bool HAS_STANDARD_MATERIAL = true;
public extern static const bool HAS_OPENPBR_MATERIAL = true;
public extern static const bool HAS_SPONGECAKE_MATERIAL = true;
/// Heavy BSDF derivative family of this pass, or all families. Surface/emitter decoding remains scene-wide.
public extern static const uint BACKWARD_MATERIAL_FAMILY = 0xffffffff;
/// Camera misses and light-source emission do not belong to a surface vertex; accumulate them in one pass.
public extern static const bool INCLUDE_SHARED_DERIVATIVES = true;
public uint seed;
public uint standard_type_id;
public uint openpbr_type_id;
public uint spongecake_type_id;
/// Whether materials have normal maps. Without them, shading frames are detached (constant).
public bool normal_maps;
public uint light_slot_base;
public extern struct LightSampler : ILightSampler = PowerLightSampler;
public ParameterBlock<LightSampler> light_sampler;

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

/// Path records of the kernels that differentiate whole paths: vertex k of the path in record slot i is
/// path_vertices[k * path_slots + i] (PathVertex, paths.slang), k <= max_bounces, so the buffer has
/// (max_bounces + 1) x path_slots entries and the path depth is a runtime value. The host dispatches the kernels in
/// chunks of at most path_slots threads.
RWStructuredBuffer<PathVertex> path_vertices;
public uint path_slots;
