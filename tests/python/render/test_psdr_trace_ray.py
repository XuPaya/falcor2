# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PSDR's TraceRay fallback: primal parity, opacity and derivative behavior.

Run on Vulkan with SLANGPY_DEVICE=vulkan python -m pytest
tests/python/render/test_psdr_trace_ray.py -q -s. The fallback is forced so these
checks also exercise it on devices that support inline ray queries.
"""

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.psdr import PSDR, TERMS

from .test_adjoint_pathtracer import _entity, add_box, add_camera, add_mesh, add_quad_y
from .test_psdr_parity import forward_image, path_tracer


DEVICE_TYPES = [t for t in helpers.DEFAULT_DEVICE_TYPES if t != spy.DeviceType.cuda]


def standard(scene, color=(0.6, 0.6, 0.6), **props):
    return scene.create_material("StandardMaterial", f2.Properties({
        "base_color_factor": spy.float3(*color), "roughness_factor": 0.8, **props,
    }))


def small_scene(device, size=24):
    scene = f2.Scene.create(device)
    add_quad_y(scene, -1.5, 1.5, -1.5, 1.5, 0, standard(scene))
    box = add_box(scene, (-0.3, 0.15, -0.3), (0.3, 0.75, 0.3), standard(scene, (0.7, 0.2, 0.1)))
    light = _entity(scene, (-0.6, 1.8, -0.7)).create_component(f2.PointLight)
    light.intensity = spy.float3(3.0)
    camera = add_camera(scene, (1.7, 1.5, -2.5), (0, 0.25, 0), (0, 1, 0), 45, size, size)
    scene.update()
    return scene, camera, box, light


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("family, parameter", [("OpenPBRMaterial", "base_color"), ("SpongeCakeMaterial", "albedo")])
def test_trace_ray_material_backward_matches_finite_difference(device_type, family, parameter):
    """Exercise full reverse AD for both larger BSDF implementations on Vulkan."""
    device = helpers.get_device(device_type, floating_point_mode=spy.SlangFloatingPointMode.precise)
    if not device.has_feature(spy.Feature.ray_tracing):
        pytest.skip("A ray-tracing pipeline is required")
    scene = f2.Scene.create(device)
    properties = {parameter: spy.float3(0.5)}
    if family == "OpenPBRMaterial":
        properties["specular_roughness"] = 0.8
    else:
        properties.update(roughness=0.8, thickness=1.0, base_weight=0.0)
    material = scene.create_material(family, f2.Properties(properties))
    add_quad_y(scene, -4, 4, -4, 4, 0, material)
    light = _entity(scene, (-0.4, 2, 0.5)).create_component(f2.PointLight)
    light.intensity = spy.float3(3.0)
    camera = add_camera(scene, (0, 1, -2), (0, 0, 0), (0, 1, 0), 20, 8, 8)
    scene.update()
    psdr = PSDR(device, scene, camera, max_bounces=1, traversal="trace_ray")
    spp, seed = 4, 13
    image = psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"]
    assert np.isfinite(image).all() and image.max() > 0
    adjoint = np.full_like(image, 1.0 / image.size)
    vertices, gradients = psdr.backward(adjoint, spp=spp, seed=seed, terms=("interior",), antithetic=False)
    assert np.isfinite(vertices).all() and np.isfinite(gradients).all()
    slot = psdr.material_slot(material, parameter)
    analytic = gradients[slot:slot + 3].sum()
    # Binary-exact values survive half-precision material storage without a mismatched FD step.
    epsilon = 0.03125
    means = []
    try:
        for value in (0.5 - epsilon, 0.5 + epsilon):
            material[parameter] = spy.float3(value)
            scene.update()
            psdr.update(geometry=False)
            means.append(psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"].mean())
    finally:
        material[parameter] = spy.float3(0.5)
        scene.update()
        psdr.update(geometry=False)
    finite_difference = (means[1] - means[0]) / (2 * epsilon)
    print(f"{family} {parameter}: analytic={analytic:.8g}, finite_difference={finite_difference:.8g}")
    assert analytic > 0 and finite_difference > 0
    np.testing.assert_allclose(analytic, finite_difference, rtol=0.05, atol=1e-5)


@pytest.fixture(scope="module", params=DEVICE_TYPES)
def context(request):
    device = helpers.get_device(request.param, floating_point_mode=spy.SlangFloatingPointMode.precise)
    if not device.has_feature(spy.Feature.ray_tracing):
        pytest.skip("A ray-tracing pipeline is required")
    scene, camera, box, light = small_scene(device)
    psdr = PSDR(device, scene, camera, max_bounces=2, photon_radius=0.05,
                photon_count=4096, check_visibility=True,
                light_sampler=f2.UniformLightSampler(), traversal="trace_ray")
    return device, scene, camera, box, light, psdr


def test_trace_ray_primal_matches_reference(context):
    device, scene, camera, _, _, psdr = context
    spp, seed = 4, 7
    image = psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"]
    assert np.isfinite(image).all() and image.max() > 0
    reference = path_tracer(device, camera, 3, f2.UniformLightSampler())
    expected = forward_image(device, scene, camera, reference, spp, seed)
    np.testing.assert_allclose(image, expected, rtol=2e-5, atol=2e-6)


def test_trace_ray_light_derivative_matches_finite_difference(context):
    _, scene, _, _, light, psdr = context
    spp, seed, epsilon = 8, 11, 0.03
    adjoint = np.full((psdr.height, psdr.width, 3), 1 / (psdr.height * psdr.width * 3), np.float32)
    _, gradients = psdr.backward(adjoint, spp=spp, seed=seed, terms=("interior",), antithetic=False)
    slot = psdr.light_slot(light, "emission")
    analytic = gradients[slot:slot + 3].sum()
    try:
        means = []
        for intensity in (3.0 - epsilon, 3.0 + epsilon):
            light.intensity = spy.float3(intensity)
            scene.update()
            psdr.update(geometry=False)
            means.append(psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"].mean())
    finally:
        light.intensity = spy.float3(3.0)
        scene.update()
        psdr.update(geometry=False)
    finite_difference = (means[1] - means[0]) / (2 * epsilon)
    assert analytic > 0 and finite_difference > 0
    np.testing.assert_allclose(analytic, finite_difference, rtol=0.02, atol=2e-5)


@pytest.mark.parametrize("term", TERMS)
def test_trace_ray_backward_terms(context, term):
    _, _, _, box, _, psdr = context
    # A spatially varying adjoint avoids cancellation of silhouette contributions.
    x = np.linspace(0.2, 1.0, psdr.width, dtype=np.float32)[None, :, None]
    adjoint = np.broadcast_to(x, (psdr.height, psdr.width, 3)).copy()
    vertex, params = psdr.backward(adjoint, spp=4, seed=17, terms=(term,), antithetic=False)
    assert np.isfinite(vertex).all() and np.isfinite(params).all()
    assert np.linalg.norm(vertex[psdr.vertex_range(box)]) > 0, term


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_trace_ray_opacity_rejects_and_accepts_hits(device_type):
    device = helpers.get_device(device_type, floating_point_mode=spy.SlangFloatingPointMode.precise)
    if not device.has_feature(spy.Feature.ray_tracing):
        pytest.skip("A ray-tracing pipeline is required")
    scene = f2.Scene.create(device)
    emitter = standard(scene, emissive_factor=spy.float3(2, 1, 0.5))
    mask = standard(scene, (0, 0, 0), alpha_mode=f2.AlphaMode.mask, alpha_factor=0.25, alpha_cutoff=0.5)
    faces = [[0, 2, 1], [1, 2, 3]]
    for z, material in ((0, emitter), (-1, mask)):
        add_mesh(scene, [[-2, -2, z], [2, -2, z], [-2, 2, z], [2, 2, z]], faces, material)
    camera = add_camera(scene, (0, 0, -3), (0, 0, 0), (0, 1, 0), 30, 8, 8)
    scene.update()
    psdr = PSDR(device, scene, camera, max_bounces=0, traversal="trace_ray")
    image = psdr.forward(spp=1, seed=3, terms=())["primal"]
    np.testing.assert_allclose(image, np.broadcast_to([2, 1, 0.5], image.shape), rtol=1e-5)
    mask.alpha_factor = 0.75
    scene.update()
    psdr.update(geometry=False)
    blocked = psdr.forward(spp=1, seed=3, terms=())["primal"]
    np.testing.assert_array_equal(blocked, 0)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_trace_ray_opacity_in_shadow_rays(device_type):
    device = helpers.get_device(device_type, floating_point_mode=spy.SlangFloatingPointMode.precise)
    if not device.has_feature(spy.Feature.ray_tracing):
        pytest.skip("A ray-tracing pipeline is required")
    scene = f2.Scene.create(device)
    add_quad_y(scene, -4, 4, -4, 4, 0, standard(scene))
    mask = standard(scene, alpha_mode=f2.AlphaMode.mask, alpha_factor=0.25, alpha_cutoff=0.5)
    add_quad_y(scene, -4, 4, -4, 4, 1, mask)
    light = _entity(scene, (0, 2, 0)).create_component(f2.PointLight)
    light.intensity = spy.float3(3.0)
    # The camera is below the mask: only shadow rays towards the light cross it.
    camera = add_camera(scene, (0, 0.5, -1), (0, 0, 0), (0, 1, 0), 20, 8, 8)
    scene.update()
    psdr = PSDR(device, scene, camera, max_bounces=1, traversal="trace_ray")
    lit = psdr.forward(spp=1, seed=3, terms=())["primal"]
    assert np.isfinite(lit).all() and lit.min() > 0
    mask.alpha_factor = 0.75
    scene.update()
    psdr.update(geometry=False)
    blocked = psdr.forward(spp=1, seed=3, terms=())["primal"]
    np.testing.assert_array_equal(blocked, 0)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_traversals_match(device_type):
    """Inline ray queries and the TraceRay pipeline both compile where the device supports them (TraceRay must stay
    out of the inline programs' compute kernels: HLSL rejects it there, E36107) and agree on every term."""
    device = helpers.get_device(device_type, floating_point_mode=spy.SlangFloatingPointMode.precise)
    if not (device.has_feature(spy.Feature.ray_query) and device.has_feature(spy.Feature.ray_tracing)):
        pytest.skip("Inline ray queries and a ray-tracing pipeline are required")
    scene, camera, box, _ = small_scene(device)
    x = np.linspace(0.2, 1.0, camera.width, dtype=np.float32)[None, :, None]
    adjoint = np.broadcast_to(x, (camera.height, camera.width, 3)).copy()
    results = {}
    for traversal in ("ray_query", "trace_ray"):
        psdr = PSDR(device, scene, camera, max_bounces=2, photon_radius=0.05, photon_count=4096,
                    check_visibility=True, light_sampler=f2.UniformLightSampler(), traversal=traversal)
        image = psdr.forward(spp=4, seed=7, terms=(), antithetic=False)["primal"]
        grads = {term: psdr.backward(adjoint, spp=4, seed=17, terms=(term,), antithetic=False) for term in TERMS}
        results[traversal] = image, grads, psdr.vertex_range(box)
    (image, grads, box_range), (other, other_grads, _) = results["ray_query"], results["trace_ray"]
    assert np.isfinite(image).all() and image.max() > 0
    np.testing.assert_allclose(image, other, rtol=2e-5, atol=2e-6)
    for term in TERMS:
        for a, b in zip(grads[term], other_grads[term]):
            assert np.isfinite(a).all() and np.isfinite(b).all(), term
            # Float atomics accumulate in a different order.
            assert np.linalg.norm(a - b) <= 1e-3 * max(np.linalg.norm(a), 1e-12), term
        assert np.linalg.norm(grads[term][0][box_range]) > 0, term
