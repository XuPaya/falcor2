// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// Scene queries of falcor2.psdr: geometry instance layouts, vertices and texture layouts.

/// Vertex offset, vertex count, index offset (in triangles) and triangle count of a geometry instance.
public uint4 instance_info(uint gid)
{
    TriangleGeometryInstanceDesc d
        = reinterpret<TriangleGeometryInstanceDesc>(g_scene.get_geometry_instance_desc(GeometryInstanceID(gid)));
    return uint4(d.vertex_offset, d.vertex_count, uint(g_scene.get_geometry_instance_desc(GeometryInstanceID(gid)).header.material_id), d.index_count / 3);
}

/// Global vertex indices of a triangle.
public uint3 triangle_indices(uint gid, uint prim)
{
    uint[3] ids = g_scene.get_triangle_indices(GeometryInstanceID(gid), prim);
    return uint3(ids[0], ids[1], ids[2]);
}

/// World-space position of a vertex under a geometry instance's transform.
public float3 instance_vertex(uint gid, uint v)
{
    return world_vertex(0, GeometryInstanceID(gid), v);
}

/// Texture index, width and height of a texture, or NO_SLOT for none (or UDIM textures).
uint3 texture_info(RawTextureHandle h)
{
    uint width, height, mip_count;
    if (!h.is_valid() || !h.get_dimensions(0, width, height, mip_count))
        return uint3(NO_SLOT, 0, 0);
    return uint3(h.texture_index, width, height);
}

uint3 texture_info(uint handle, bool udim)
{
    return udim ? uint3(NO_SLOT, 0, 0) : texture_info(RawTextureHandle(handle));
}

/// Texture index, width and height of texture `role` of a material: StandardMaterial base color,
/// metallic-roughness, emissive, normal and transmission textures (roles 0-4); OpenPBRMaterial attributes in
/// declaration order (0-29) and normal map (30).
public uint3 material_texture(uint material, uint role, uint standard_id, uint openpbr_id)
{
    MaterialData data = g_scene.material_system.get_material_data(MaterialID(material));
    if (data.header.type_id == standard_id) {
        StandardMaterial m = reinterpret<StandardMaterial>(data);
        switch (role) {
        case 0: return texture_info(m.base_color_texture.data, m.base_color_texture.is_udim());
        case 1: return texture_info(m.metallic_roughness_texture.data, m.metallic_roughness_texture.is_udim());
        case 2: return texture_info(m.emissive_texture.data, m.emissive_texture.is_udim());
        case 3: return texture_info(m.normal_texture.data, m.normal_texture.is_udim());
        default: return texture_info(m.transmission_texture.data, m.transmission_texture.is_udim());
        }
    }
    if (has_openpbr && data.header.type_id == openpbr_id)
        return openpbr_texture(reinterpret<OpenPBRMaterial>(data), role);
    return uint3(NO_SLOT, 0, 0);
}

/// Texture index, width and height of texture `role` of a SpongeCakeMaterial: albedo, top albedo, base color and
/// normal textures (roles 0-3).
public uint3 spongecake_texture(uint material, uint role, uint spongecake_id)
{
    MaterialData data = g_scene.material_system.get_material_data(MaterialID(material));
    if (data.header.type_id != spongecake_id)
        return uint3(NO_SLOT, 0, 0);
    SpongeCakeMaterial m = reinterpret<SpongeCakeMaterial>(data);
    switch (role) {
    case 0: return texture_info(m.albedo_texture.data, m.albedo_texture.is_udim());
    case 1: return texture_info(m.top_albedo_texture.data, m.top_albedo_texture.is_udim());
    case 2: return texture_info(m.base_color_texture.data, m.base_color_texture.is_udim());
    default: return texture_info(m.normal_texture.data, m.normal_texture.is_udim());
    }
}

/// Texture index, width and height of an environment map light's texture.
public uint3 env_map_texture(uint id) { return texture_info(env_map_light(id).env_map_texture_handle); }
