// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Path-space differentiable rendering (PSDR) on the falcor2 scene.
//
// Estimators follow PSDR-Enzyme: the interior term uses the material-form reparameterization
// (algorithm1.cpp), plus pixel-boundary (boundary_pixel.cpp), primary-edge (boundary_primary.cpp)
// and photon-mapping secondary-edge (boundary_direct_pm.cpp) terms. Sampling, pdfs, MIS weights and
// visibility are detached. Every derivative enters through world_vertex() and param() (texels are
// parameters too): forward mode reads tangents and returns image derivatives, backward mode atomically
// accumulates gradients.
//
// The module's files: globals.slang (shader parameters), parameters.slang and geometry.slang (differentiable
// scene access), materials.slang and openpbr.slang (BSDFs), lights.slang, paths.slang (path records and the
// terms of their vertices), a file per term (interior.slang, also the primal image, pixel_boundary.slang,
// primary.slang and secondary.slang, with boundary.slang shared by the boundary terms), light_paths.slang (light
// paths from the analytic lights, part of the interior term) and queries.slang (scene queries of falcor2.psdr).

module psdr;

import falcor2.render;
import falcor2.utils;

__include "globals.slang";
__include "parameters.slang";
__include "geometry.slang";
__include "materials.slang";
__include "openpbr.slang";
__include "lights.slang";
__include "paths.slang";
__include "interior.slang";
__include "pixel_boundary.slang";
__include "boundary.slang";
__include "primary.slang";
__include "secondary.slang";
__include "light_paths.slang";
__include "queries.slang";
