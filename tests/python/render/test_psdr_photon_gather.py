# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the actual PSDR photon traversal against an independent geometric reference."""

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor.scene_shader import SceneShaderHelper
from falcor2.psdr import PSDR

from .test_psdr_trace_ray import DEVICE_TYPES, small_scene


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_trace_ray_gather_nearest_unique_photons(device_type):
    device = helpers.get_device(device_type, floating_point_mode=spy.SlangFloatingPointMode.precise)
    if not device.has_feature(spy.Feature.ray_tracing):
        pytest.skip("A ray-tracing pipeline is required")
    scene, camera, _, _ = small_scene(device, size=1)
    radius, epsilon = 0.15, 1e-4
    positions = np.column_stack((np.arange(1, 25) * 0.004, np.zeros(24), np.linspace(0.05, 0.95, 24)))
    # Endpoint-overlap spheres must be visited even though their centers lie outside the segment.
    positions = np.vstack((positions, [[0.001, 0, -0.05], [0.002, 0, 1.05],
                                       [0.0001, 0, 0.5], [0.2, 0, 0.5], [0.0002, 0, -0.3]]))
    positions = positions.astype(np.float32)
    count = len(positions)
    photons = np.zeros((count, 12), np.float32)
    photons[:, :3] = positions
    # Same-side face normals reject candidate 26, despite its very small perpendicular distance.
    photons[26, 4:7] = photons[26, 8:11] = [0, 0, 1]
    photons.view(np.uint32)[26, 11] = 1
    psdr = PSDR(device, scene, camera, max_bounces=0, photon_radius=radius, ray_epsilon=epsilon,
                photon_count=count, traversal="trace_ray")
    psdr._buffers["photons"].copy_from_numpy(photons)
    bounds = np.stack((positions - radius, positions + radius), axis=1).astype(np.float32)
    psdr._buffers["photon_bounds"].copy_from_numpy(bounds)
    psdr._build_photon_bvh()

    # Link the defining module so pipeline lookup can find its stage entry points.
    probe = spy.Module(device.load_module(str(Path(__file__).with_suffix(".slang"))),
                       link=[device.load_module("falcor2.psdr.psdr")])
    helper = SceneShaderHelper(device)
    module = helper.get_module(scene, probe)
    indices = device.create_buffer(data=np.zeros(17, np.uint32), struct_size=4,
                                   usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource)
    distances = device.create_buffer(data=np.zeros(17, np.float32), struct_size=4,
                                     usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource)

    def bind(cursor):
        cursor["photons"] = psdr._buffers["photons"]
        cursor["photon_tlas"] = psdr._photon_accel[2]
        cursor["photon_radius"] = radius
        cursor["ray_epsilon"] = epsilon
        cursor["result_indices"] = indices
        cursor["result_distances"] = distances

    function = module.probe_photon_gather
    function = function.constants({"PHOTON_HIT_GROUP_INDEX": 0, "PHOTON_MISS_INDEX": 0})
    function = function.prelude("export struct SceneTraversal : IPsdrTraversal = PipelineTraversal;\n")
    function = function.write(bind)
    function = function.ray_tracing(
        hit_groups=[{"hit_group_name": "photons", "intersection_entry_point": "psdr_photon_intersection",
                     "any_hit_entry_point": "psdr_photon_any_hit"}],
        miss_entry_points=["psdr_photon_miss"], max_recursion=1, max_ray_payload_size=136, max_attribute_size=4,
    )
    function(origin=spy.float3(0), dir=spy.float3(0, 0, 1), distance=1.0)

    found_indices = indices.to_numpy().view(np.uint32)
    found_distances = distances.to_numpy().view(np.float32)
    actual = found_indices[1:1 + found_indices[0]]
    radial = np.linalg.norm(positions[:, :2], axis=1)
    eligible = (radial <= radius) & (positions[:, 2] + radius >= epsilon) & (positions[:, 2] - radius <= 1 - epsilon)
    eligible[26] = False
    candidates = np.flatnonzero(eligible)
    expected = candidates[np.argsort(radial[candidates])[:16]]
    assert len(actual) == 16 and len(np.unique(actual)) == 16
    np.testing.assert_array_equal(np.sort(actual), np.sort(expected))
    np.testing.assert_allclose(found_distances[1:17], radial[actual], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(found_distances[0], radial[expected].max(), rtol=1e-6)

    # A smaller radius leaves fewer than 16 neighbors; the buffers' broader AABBs remain conservative.
    radius = 0.01
    function(origin=spy.float3(0), dir=spy.float3(0, 0, 1), distance=1.0)
    found_indices = indices.to_numpy().view(np.uint32)
    actual = found_indices[1:1 + found_indices[0]]
    np.testing.assert_array_equal(np.sort(actual), [0, 1])
    # The endpoint centers are now farther than this smaller radius from the finite segment.
    np.testing.assert_allclose(distances.to_numpy().view(np.float32)[0], radial[1], rtol=1e-6)

    function(origin=spy.float3(10, 0, 0), dir=spy.float3(0, 0, 1), distance=1.0)
    assert indices.to_numpy().view(np.uint32)[0] == 0
    assert distances.to_numpy().view(np.float32)[0] == 0
