# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame gradients: derivatives of the image with respect to the stored vertex normals and tangents that the shading
frames interpolate (PSDR.backward(frame_gradients=True), set_tangents(normals=..., frame_tangents=...))."""

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.psdr import PSDR

from .test_adjoint_pathtracer import _entity, add_camera, add_mesh
from .test_psdr_parity import DEVICE_TYPES

FAMILIES = {
    "StandardMaterial": {"base_color_factor": spy.float3(0.5), "roughness_factor": 0.35},
    "OpenPBRMaterial": {"base_color": spy.float3(0.5), "specular_roughness": 0.35,
                        "specular_roughness_anisotropy": 0.6},
    "SpongeCakeMaterial": {"albedo": spy.float3(0.5), "roughness": 0.4, "thickness": 1.0, "base_weight": 0.0},
}


def bumpy_patch(scene, x0, material, n=9):
    """Smooth-shaded height field over [x0, x0 + 1.6] x [-0.8, 0.8] (normals +y)."""
    u, w = np.meshgrid(np.linspace(0, 1.6, n), np.linspace(-0.8, 0.8, n), indexing="ij")
    y = 0.12 * np.cos(3 * u) * np.cos(4 * w)
    positions = np.stack([x0 + u, y, w], -1).reshape(-1, 3)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1], i[1:, :-1], i[:-1, 1:], i[1:, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    return add_mesh(scene, positions, faces, material)


@pytest.fixture(scope="module", params=DEVICE_TYPES)
def patches(request):
    # Use production-style shader code: debug metadata makes DXC's generated AD kernels much larger.
    # Keep precise arithmetic and the device's validation layers enabled.
    device = helpers.get_device(request.param, floating_point_mode=spy.SlangFloatingPointMode.precise,
                                shader_debug_info=spy.SlangDebugInfoLevel.none)
    if not device.has_feature(spy.Feature.ray_query):
        pytest.skip("Inline ray queries are required")
    scene = f2.Scene.create(device)
    instances = {}
    for k, (family, props) in enumerate(FAMILIES.items()):
        material = scene.create_material(family, f2.Properties(props))
        instances[family] = bumpy_patch(scene, -2.4 + 1.6 * k, material)
    light = _entity(scene, (-0.5, 2, 0.7)).create_component(f2.PointLight)
    light.intensity = spy.float3(3)
    camera = add_camera(scene, (0, 3, 0.4), (0, 0, 0), (0, 0, 1), 70, 24, 12)
    scene.update()
    psdr = PSDR(device, scene, camera, max_bounces=1, light_sampler=f2.UniformLightSampler(),
                photon_count=256, vertex_grad_clamp=0)
    adjoint = np.broadcast_to(np.linspace(0.2, 1, psdr.width, dtype=np.float32)[None, :, None],
                              (psdr.height, psdr.width, 3)).copy() / (psdr.width * psdr.height)
    return scene, instances, psdr, adjoint


def directions(geometry):
    """Smooth perturbations of the stored normals and tangents, orthogonal to them (the stored tangents are unit)."""
    normals, tangents = np.asarray(geometry.normals(0), np.float64), np.asarray(geometry.tangents(0), np.float64)
    p = np.asarray(geometry.positions(0), np.float64)
    field = np.stack([np.sin(2 * p[:, 0]), 0.5 * np.cos(p[:, 2]), np.cos(3 * p[:, 2])], -1)
    dn = np.cross(normals, field)
    dt = np.cross(tangents, normals) * np.sin(3 * p[:, 0])[:, None]
    return normals, tangents, dn, dt


def frame_difference(scene, psdr, geometry, normals, tangents, dn, dt, adjoint, epsilon, spp, seed):
    values = []
    try:
        for sign in (-1, 1):
            geometry.set_normals(0, (normals + sign * epsilon * dn).astype(np.float32))
            geometry.set_tangents(0, (tangents + sign * epsilon * dt).astype(np.float32))
            scene.update()
            psdr.update()
            image = psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"]
            values.append(float(np.sum(image * adjoint)))
    finally:
        geometry.set_normals(0, normals.astype(np.float32))
        geometry.set_tangents(0, tangents.astype(np.float32))
        scene.update()
        psdr.update()
    return (values[1] - values[0]) / (2 * epsilon)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_frame_gradients_match_finite_differences_and_forward_mode(patches, family):
    scene, instances, psdr, adjoint = patches
    instance = instances[family]
    rows = psdr.vertex_range(instance)
    normals, tangents, dn, dt = directions(instance.geometry)
    spp, seed = 64, 7
    vertex, _ = psdr.backward(adjoint, spp=spp, seed=seed, terms=("interior",), antithetic=False,
                              frame_gradients=True)
    grad_n, grad_t = psdr.frame_gradients()
    assert np.isfinite(grad_n).all() and np.isfinite(grad_t).all()
    reverse = float(np.sum(grad_n[rows] * dn) + np.sum(grad_t[rows] * dt))

    full_n, full_t = np.zeros((psdr.vertex_count, 3)), np.zeros((psdr.vertex_count, 3))
    full_n[rows], full_t[rows] = dn, dt
    try:
        psdr.set_tangents(normals=full_n, frame_tangents=full_t)
        image = psdr.forward(spp=spp, seed=seed, terms=("interior",), antithetic=False)["interior"]
    finally:
        psdr.set_tangents()
    forward = float(np.sum(image * adjoint))
    np.testing.assert_allclose(reverse, forward, rtol=3e-3, atol=1e-7, err_msg=f"{family}: reverse vs forward")

    # Without frame_gradients the stored frames are constant: no frame gradients, the same vertex gradients.
    vertex_constant, _ = psdr.backward(adjoint, spp=spp, seed=seed, terms=("interior",), antithetic=False)
    assert psdr.frame_gradients() is None
    np.testing.assert_allclose(vertex_constant, vertex, rtol=2e-3, atol=1e-7)

    # (Writing the normals moves the mesh in the global vertex buffer: vertex ranges change from here on.)
    numerical = frame_difference(scene, psdr, instance.geometry, normals, tangents, dn, dt, adjoint, 0.02, spp, seed)
    print(f"{family}: frame gradient={reverse:.6g}, finite difference={numerical:.6g}")
    assert abs(numerical) > 1e-5, f"{family}: the perturbation must change the image"
    np.testing.assert_allclose(reverse, numerical, rtol=0.1, err_msg=f"{family}: reverse vs finite difference")


def test_frame_gradients_accumulate(patches):
    _, _, psdr, adjoint = patches
    psdr.backward(adjoint, spp=4, seed=3, terms=("interior",), antithetic=False, frame_gradients=True)
    expected = [g.copy() for g in psdr.frame_gradients()]
    for _ in range(2):
        psdr.backward(adjoint, spp=4, seed=3, terms=("interior",), antithetic=False, frame_gradients=True,
                      accumulate=True)
    psdr.gradients()
    for actual, single in zip(psdr.frame_gradients(), expected):
        np.testing.assert_allclose(actual, 2 * single, rtol=2e-3, atol=1e-7)
