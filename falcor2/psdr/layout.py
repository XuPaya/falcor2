# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parameter layout of PSDR and the constants it shares with the shaders (slang/falcor2/psdr/globals.slang)."""

import numpy as np

import falcor2 as f2

MAX_DEPTH = 8
MATERIAL_SLOTS = 52
LIGHT_SLOTS = 6
PHOTON_EMISSIONS = 8192
NO_SLOT = 0xFFFFFFFF
_OPENPBR_ATTRIBUTES = (
    "base_weight base_color base_diffuse_roughness base_metalness specular_weight specular_color specular_roughness "
    "specular_roughness_anisotropy specular_ior coat_weight coat_color coat_roughness coat_roughness_anisotropy "
    "coat_ior coat_darkening fuzz_weight fuzz_color fuzz_roughness transmission_weight transmission_color "
    "transmission_depth transmission_scatter transmission_scatter_anisotropy transmission_dispersion_scale "
    "transmission_dispersion_abbe_number thin_film_weight thin_film_thickness thin_film_ior emission_luminance "
    "emission_color"
).split()
_OPENPBR_COLORS = {"base_color", "specular_color", "coat_color", "fuzz_color", "transmission_color",
                   "transmission_scatter", "emission_color"}
MATERIAL_PARAMS = {
    "StandardMaterial": {
        "base_color_factor": 0,
        "metallic_factor": 3,
        "roughness_factor": 4,
        "ior": 5,
        "emissive_factor": 6,
        "diffuse_transmission_factor": 9,
        "specular_transmission_factor": 10,
        "transmission_factor": 11,
        "normal_texture_scale": 14,
    },
    # Attributes in declaration order, three slots for colors (see slang/falcor2/psdr/openpbr.slang).
    "OpenPBRMaterial": dict(zip(_OPENPBR_ATTRIBUTES, np.cumsum(
        [0] + [3 if name in _OPENPBR_COLORS else 1 for name in _OPENPBR_ATTRIBUTES[:-1]]).tolist()))
    | {"normal_texture_scale": 44, "subsurface_weight": 45, "subsurface_color": 46, "subsurface_scatter_anisotropy": 49},
}
# Textures with texel parameters (see material_texture() in slang/falcor2/psdr/queries.slang).
MATERIAL_TEXTURES = {
    "StandardMaterial": ("base_color", "metallic_roughness", "emissive", "normal", "transmission"),
    "OpenPBRMaterial": tuple(_OPENPBR_ATTRIBUTES) + ("normal", "subsurface_weight", "subsurface_color",
                                                     "subsurface_scatter_anisotropy"),
}
# Analytic lights: emission and world-space translation. Environment map: intensity and rotation (a rotation
# vector in world space, applied before the map's orientation).
LIGHT_PARAMS = {"emission": 0, "translation": 3}
ENV_MAP_PARAMS = {"intensity": 0, "rotation": 3}
TERMS = ("interior", "pixel", "primary", "secondary")
# Environment lights (LightFlags::environment) are numbered after the analytic lights.
ENVIRONMENT_LIGHTS = (f2.ConstantLight, f2.EnvMapLight, f2.DistantLight)


def light_emission(light: f2.Light):
    """The emission parameter of a light: the intensity of point lights and environment maps, else the radiance."""
    value = light.intensity if isinstance(light, (f2.PointLight, f2.EnvMapLight)) else light.radiance
    return [value[0], value[1], value[2]]
