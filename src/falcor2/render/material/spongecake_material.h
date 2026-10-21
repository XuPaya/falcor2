// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/types.h"

#include <array>
#include <vector>

namespace falcor {

/// SpongeCake material (Wang et al. 2023, "SpongeCake: A Layered Microflake Surface Appearance Model"): a thin,
/// two-sided slab of two volumetric layers (top and body) without interfaces, over an optional Lambertian base. Each
/// layer is a medium of SGGX microflakes (surface-like flakes or fibers) or a Henyey-Greenstein medium (forward
/// scattering, e.g. translucent fibers of sheer fabric). Single scattering is evaluated in closed form, multiple
/// scattering with per-material tables computed in update() (see slang/falcor2/render/bsdfs/spongecake_bsdf.slang).
class SpongeCakeMaterial : public Material {
    FALCOR_SCENE_OBJECT(SpongeCakeMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    SpongeCakeMaterial();
    virtual ~SpongeCakeMaterial() override;

    // SceneObject interface

    virtual void on_load_resources() override;

    // Material interface

    virtual void update(SceneUpdateContext& ctx) override;
    virtual shared::MaterialFlags flags() const override
    {
        return shared::MaterialFlags::two_sided | shared::MaterialFlags::thin_walled;
    }

    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "albedo",
                &SpongeCakeMaterial::m_albedo,
                "Single-scattering albedo of the body layer's flakes (times albedo_texture).",
                reflection::default_value(float3(0.8f)),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "albedo_texture",
                &SpongeCakeMaterial::m_albedo_texture,
                "Body albedo texture (linear; takes precedence over path).",
                reflection::UIFlags::advanced,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_resources)
            )
            .def_rw(
                "albedo_texture_path",
                &SpongeCakeMaterial::m_albedo_texture_path,
                "Body albedo texture path (sRGB).",
                reflection::UIFlags::advanced,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_resources)
            )
            .def_rw(
                "thickness",
                &SpongeCakeMaterial::m_thickness,
                "Optical thickness of the body layer (extinction along the normal for unit projected area).",
                reflection::default_value(4.f),
                reflection::value_range(0.0, 1000.0),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "roughness",
                &SpongeCakeMaterial::m_roughness,
                "SGGX roughness of the body layer's flakes (1: isotropic); for a Henyey-Greenstein body (hg) the "
                "anisotropy is g = 1 - roughness.",
                reflection::default_value(1.f),
                reflection::value_range(0.01, 1.0),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "fiber",
                &SpongeCakeMaterial::m_fiber,
                "Body flakes are fibers along the tangent (else surface-like flakes along the normal).",
                reflection::default_value(false),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "hg",
                &SpongeCakeMaterial::m_hg,
                "Body layer is a Henyey-Greenstein medium with anisotropy g = 1 - roughness (forward scattering; "
                "overrides fiber).",
                reflection::default_value(false),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "top_albedo",
                &SpongeCakeMaterial::m_top_albedo,
                "Single-scattering albedo of the top layer's flakes (times top_albedo_texture).",
                reflection::default_value(float3(1.f)),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "top_albedo_texture",
                &SpongeCakeMaterial::m_top_albedo_texture,
                "Top albedo texture (linear; takes precedence over path).",
                reflection::UIFlags::advanced,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_resources)
            )
            .def_rw(
                "top_albedo_texture_path",
                &SpongeCakeMaterial::m_top_albedo_texture_path,
                "Top albedo texture path (sRGB).",
                reflection::UIFlags::advanced,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_resources)
            )
            .def_rw(
                "top_thickness",
                &SpongeCakeMaterial::m_top_thickness,
                "Optical thickness of the top layer (0: no top layer).",
                reflection::default_value(0.f),
                reflection::value_range(0.0, 1000.0),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "top_roughness",
                &SpongeCakeMaterial::m_top_roughness,
                "SGGX roughness of the top layer's flakes; for a Henyey-Greenstein top layer (top_hg) g = 1 - roughness.",
                reflection::default_value(0.2f),
                reflection::value_range(0.01, 1.0),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "top_fiber",
                &SpongeCakeMaterial::m_top_fiber,
                "Top flakes are fibers along the tangent (else surface-like flakes along the normal).",
                reflection::default_value(false),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "top_hg",
                &SpongeCakeMaterial::m_top_hg,
                "Top layer is a Henyey-Greenstein medium with anisotropy g = 1 - top_roughness (overrides top_fiber).",
                reflection::default_value(false),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "base_color",
                &SpongeCakeMaterial::m_base_color,
                "Albedo of the Lambertian base below the slab (times base_color_texture).",
                reflection::default_value(float3(0.8f)),
                reflection::value_range_unit(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "base_color_texture",
                &SpongeCakeMaterial::m_base_color_texture,
                "Base color texture (linear; takes precedence over path).",
                reflection::UIFlags::advanced,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_resources)
            )
            .def_rw(
                "base_color_texture_path",
                &SpongeCakeMaterial::m_base_color_texture_path,
                "Base color texture path (sRGB).",
                reflection::UIFlags::advanced,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_resources)
            )
            .def_rw(
                "base_weight",
                &SpongeCakeMaterial::m_base_weight,
                "Fraction of the light transmitted by the slab that meets the base (0: two-sided transmissive slab, "
                "1: opaque).",
                reflection::default_value(0.f),
                reflection::value_range_unit(),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "delta_transmission",
                &SpongeCakeMaterial::m_delta_transmission,
                "Pass unscattered light straight through the slab.",
                reflection::default_value(true),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            )
            .def_rw(
                "normal_texture",
                &SpongeCakeMaterial::m_normal_texture,
                "Normal texture (takes precedence over path).",
                reflection::UIFlags::advanced,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_path",
                &SpongeCakeMaterial::m_normal_texture_path,
                "Normal texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_resources)
            )
            .def_rw(
                "normal_texture_scale",
                &SpongeCakeMaterial::m_normal_texture_scale,
                "Normal texture scale.",
                reflection::default_value(1.f),
                reflection::on_change(&SpongeCakeMaterial::mark_dirty_properties)
            );
    }

private:
    void mark_dirty_resources() { mark_dirty(DirtyFlags::resources); }
    void mark_dirty_properties() { mark_dirty(DirtyFlags::properties); }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float3 m_albedo{0.8f};
    ref<sgl::Texture> m_albedo_texture;
    std::filesystem::path m_albedo_texture_path;
    float m_thickness{4.f};
    float m_roughness{1.f};
    bool m_fiber{false};
    bool m_hg{false};
    float3 m_top_albedo{1.f};
    ref<sgl::Texture> m_top_albedo_texture;
    std::filesystem::path m_top_albedo_texture_path;
    float m_top_thickness{0.f};
    float m_top_roughness{0.2f};
    bool m_top_fiber{false};
    bool m_top_hg{false};
    float3 m_base_color{0.8f};
    ref<sgl::Texture> m_base_color_texture;
    std::filesystem::path m_base_color_texture_path;
    float m_base_weight{0.f};
    bool m_delta_transmission{true};
    ref<sgl::Texture> m_normal_texture;
    std::filesystem::path m_normal_texture_path;
    float m_normal_texture_scale{1.f};

    TextureHandle m_albedo_texture_handle;
    TextureHandle m_top_albedo_texture_handle;
    TextureHandle m_base_color_texture_handle;
    TextureHandle m_normal_texture_handle;

    /// Tables of the layers (thicknesses, roughnesses, phase types) they were computed for.
    std::array<float, 6> m_table_key{-1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
    std::vector<float> m_tables;
    ref<sgl::Buffer> m_tables_buffer;
};

} // namespace falcor
