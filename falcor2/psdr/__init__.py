# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-space differentiable rendering (PSDR) of falcor2 scenes.

Forward mode renders image derivatives along tangents of the vertex positions and scene
parameters. Backward mode accumulates dL/dparameter from a per-pixel adjoint dL/dI. Four
terms are supported: interior, pixel boundary, primary edges and photon-mapping secondary
edges (see the Slang module in slang/falcor2/psdr/).

Vertices are indexed like the scene vertex buffer and are object-space. Parameters are the factors of
StandardMaterial and OpenPBRMaterial (``MATERIAL_PARAMS``), the emission (intensity or radiance) and
world-space translation of analytic lights and the intensity and rotation of an environment map
(``LIGHT_PARAMS``, ``ENV_MAP_PARAMS``), and the texels of the textures of these materials (including
normal maps) and of the environment map (``texture_slot()``). After editing materials, lights, textures or
vertex positions, call ``scene.update()`` and then ``PSDR.update()``. Adding or removing objects, materials,
lights or textures requires a new PSDR object.

Modules: renderer (the PSDR renderer), layout (the parameter layout), edges (the edge lists of the boundary
terms) and openpbr (the differentiable OpenPBR headers).
"""

from falcor2.psdr.layout import (ENV_MAP_PARAMS, LIGHT_PARAMS, LIGHT_SLOTS, MATERIAL_PARAMS, MATERIAL_SLOTS,
                                 MATERIAL_TEXTURES, NO_SLOT, PHOTON_EMISSIONS, TERMS)
from falcor2.psdr.openpbr import generate_openpbr_headers
from falcor2.psdr.renderer import PSDR

__all__ = ["ENV_MAP_PARAMS", "LIGHT_PARAMS", "LIGHT_SLOTS", "MATERIAL_PARAMS", "MATERIAL_SLOTS", "MATERIAL_TEXTURES",
           "NO_SLOT", "PHOTON_EMISSIONS", "PSDR", "TERMS", "generate_openpbr_headers"]
