# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed derivative families must preserve fallback vertices and shared environment gradients."""

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.psdr import PSDR

from .test_adjoint_pathtracer import _entity, add_camera
from .test_psdr_parity import DEVICE_TYPES
from .test_psdr_shading_frames import bumpy_patch


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_material_dispatch_preserves_light_gradients_after_reassignment(device_type, tmp_path):
    device = helpers.get_device(
        device_type,
        floating_point_mode=spy.SlangFloatingPointMode.precise,
        shader_debug_info=spy.SlangDebugInfoLevel.none,
    )
    if not device.has_feature(spy.Feature.ray_query):
        pytest.skip("Inline ray queries are required")
    scene = f2.Scene.create(device)
    standard = scene.create_material("StandardMaterial", f2.Properties({
        "base_color_factor": spy.float3(0.25), "roughness_factor": 0.5,
    }))
    fallback = scene.create_material("StandardSpecGlossMaterial", f2.Properties({
        "diffuse_factor": spy.float3(0.75), "specular_factor": spy.float3(0.04),
        "glossiness_factor": 0.25,
    }))
    left = bumpy_patch(scene, -1.6, standard, n=4)
    right = bumpy_patch(scene, 0.0, fallback, n=4)
    point = _entity(scene, (-0.5, 2, 0.7)).create_component(f2.PointLight)
    point.intensity = spy.float3(3.0)
    sky = tmp_path / "sky.exr"
    spy.Bitmap(np.ones((4, 8, 4), np.float32)).write(str(sky))
    env = scene.create_entity().create_component(f2.EnvMapLight)
    env.env_map_path = str(sky)
    env.intensity = spy.float3(0.2)
    camera = add_camera(scene, (0, 3, 0.4), (0, 0, 0), (0, 0, 1), 70, 24, 12)
    scene.update()
    psdr = PSDR(device, scene, camera, max_bounces=1, light_sampler=f2.UniformLightSampler(),
                photon_count=256, vertex_grad_clamp=0)
    adjoint = np.broadcast_to(np.linspace(0.2, 1, 24, dtype=np.float32)[None, :, None], (12, 24, 3)).copy() / 288

    for reassigned in (False, True):
        if reassigned:
            left.materials, right.materials = [fallback], [standard]
            scene.update()
            psdr.update()
        _, gradient = psdr.backward(adjoint, spp=4, seed=7, terms=("interior",), antithetic=False)
        for light, base, parameter in ((point, 3.0, "emission"), (env, 0.2, "intensity")):
            slot = psdr.light_slot(light, parameter)
            reverse = float(gradient[slot:slot + 3].sum())
            losses = []
            try:
                for sign in (-1, 1):
                    light.intensity = spy.float3(base + sign * 0.01)
                    scene.update()
                    psdr.update(geometry=False)
                    image = psdr.forward(spp=4, seed=7, terms=(), antithetic=False)["primal"]
                    losses.append(float(np.sum(image * adjoint)))
            finally:
                light.intensity = spy.float3(base)
                scene.update()
                psdr.update(geometry=False)
            numerical = (losses[1] - losses[0]) / 0.02
            assert numerical > 1e-4, "The light must affect the rendered image"
            np.testing.assert_allclose(reverse, numerical, rtol=2e-3, atol=1e-6,
                                       err_msg=f"{parameter}, reassigned={reassigned}")
