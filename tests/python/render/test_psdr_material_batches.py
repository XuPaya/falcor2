# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External gradient checks for scenes containing all three differentiable material families."""

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.psdr import PSDR
from falcor2.psdr.renderer import MATERIAL_BACKWARD_KERNELS, MATERIAL_FAMILIES, MATERIAL_OTHER

from .test_adjoint_pathtracer import _entity, add_box, add_camera, add_quad_y
from .test_psdr_trace_ray import DEVICE_TYPES


def device_for(device_type):
    device = helpers.get_device(device_type, floating_point_mode=spy.SlangFloatingPointMode.precise,
                                shader_debug_info=spy.SlangDebugInfoLevel.none)
    if not device.has_feature(spy.Feature.ray_tracing):
        pytest.skip("A ray-tracing pipeline is required")
    return device


def materials_for(scene, families=tuple(MATERIAL_FAMILIES)):
    definitions = [
        ("StandardMaterial", "base_color_factor", {"roughness_factor": 0.8}),
        ("OpenPBRMaterial", "base_color", {"specular_roughness": 0.8}),
        ("SpongeCakeMaterial", "albedo", {"roughness": 0.8, "thickness": 1.0, "base_weight": 0.0}),
    ]
    return [(scene.create_material(family, f2.Properties({parameter: spy.float3(0.5), **properties})), parameter)
            for family, parameter, properties in definitions if family in families]


def panel_scene(device):
    scene = f2.Scene.create(device)
    materials = materials_for(scene)
    panels = [add_quad_y(scene, x0, x1, -3, 3, 0, material)
              for (x0, x1), (material, _) in zip(((-3, -0.6), (-0.6, 0.6), (0.6, 3)), materials)]
    light = _entity(scene, (-0.5, 2, 0.7)).create_component(f2.PointLight)
    light.intensity = spy.float3(3)
    camera = add_camera(scene, (0, 3, 0), (0, 0, 0), (0, 0, 1), 45, 12, 8)
    scene.update()
    return scene, camera, materials, panels, light


def renderer_for(device, scene, camera, max_bounces=1, **options):
    return PSDR(device, scene, camera, max_bounces=max_bounces, light_sampler=f2.UniformLightSampler(),
                traversal="trace_ray", photon_count=256, **options)


def backward(psdr, adjoint=None, seed=19, spp=4):
    if adjoint is None:
        adjoint = np.full((psdr.height, psdr.width, 3), 1 / (psdr.height * psdr.width * 3), np.float32)
    result = psdr.backward(adjoint, spp=spp, seed=seed, terms=("interior",), antithetic=False)
    assert all(np.isfinite(values).all() for values in result)
    return result


def difference(scene, psdr, setter, center, adjoint=None, epsilon=0.03125, seed=19, spp=4):
    values = []
    try:
        for value in (center - epsilon, center + epsilon):
            setter(value)
            scene.update()
            psdr.update(geometry=False)
            image = psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"]
            values.append(image.mean() if adjoint is None else np.sum(image * adjoint))
    finally:
        setter(center)
        scene.update()
        psdr.update(geometry=False)
    return (values[1] - values[0]) / (2 * epsilon)


def check_derivative(analytic, numerical, label, tolerance=0.05):
    print(f"{label}: analytic={analytic:.8g}, finite_difference={numerical:.8g}")
    assert np.isfinite(analytic) and np.isfinite(numerical) and analytic > 0 and numerical > 0, label
    np.testing.assert_allclose(analytic, numerical, rtol=tolerance, atol=2e-5, err_msg=label)


@pytest.fixture(scope="module", params=DEVICE_TYPES)
def panels(request):
    device = device_for(request.param)
    scene, camera, materials, instances, light = panel_scene(device)
    return scene, materials, instances, light, renderer_for(device, scene, camera)


def test_mixed_material_and_analytic_light_derivatives(panels):
    scene, materials, instances, light, psdr = panels
    vertices, params = backward(psdr)
    for material, parameter in materials:
        slot = psdr.material_slot(material, parameter)
        numerical = difference(scene, psdr, lambda value: material.__setitem__(parameter, spy.float3(value)), 0.5)
        check_derivative(params[slot:slot + 3].sum(), numerical, material.slang_type_name)
    for instance in instances:
        assert np.linalg.norm(vertices[psdr.vertex_range(instance)]) > 0
    slot = psdr.light_slot(light, "emission")
    numerical = difference(scene, psdr, lambda value: setattr(light, "intensity", spy.float3(value)), 3.0)
    check_derivative(params[slot:slot + 3].sum(), numerical, "point-light intensity", tolerance=0.01)


def test_mixed_split_unsplit_and_accumulation(panels):
    _, _, _, _, psdr = panels
    first = np.broadcast_to(np.linspace(0.1, 1, psdr.width, dtype=np.float32)[None, :, None],
                            (psdr.height, psdr.width, 3)).copy() / (psdr.width * psdr.height)
    second = first[:, ::-1, :].copy() * np.array([0.4, 0.7, 1.0], np.float32)
    psdr.split_interior = True
    expected_first = backward(psdr, first)
    expected_second = backward(psdr, second, seed=23)
    for adjoint, seed in ((first, 19), (second, 23)):
        assert psdr.backward(adjoint, spp=4, seed=seed, terms=("interior",), antithetic=False,
                             accumulate=True) is None
    for actual, a, b in zip(psdr.gradients(), expected_first, expected_second):
        np.testing.assert_allclose(actual, a + b, rtol=2e-3, atol=2e-6)
    # Reading the sum must start a fresh accumulation sequence.
    psdr.backward(first, spp=4, seed=19, terms=("interior",), antithetic=False, accumulate=True)
    for actual, expected in zip(psdr.gradients(), expected_first):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-6)
    try:
        psdr.split_interior = False
        for actual, expected in zip(backward(psdr, first), expected_first):
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-6)
    finally:
        psdr.split_interior = True


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("with_surfaces", [False, True])
def test_primary_environment_gradient_is_counted_once(device_type, with_surfaces, tmp_path):
    device = device_for(device_type)
    scene = panel_scene(device)[0] if with_surfaces else f2.Scene.create(device)
    color = np.array([0.25, 0.5, 0.75], np.float32)
    texels = np.ones((4, 8, 4), np.float32)
    texels[..., :3] = color
    path = tmp_path / "constant.exr"
    spy.Bitmap(texels).write(str(path))
    environment = scene.create_entity().create_component(f2.EnvMapLight)
    environment.env_map_path = str(path)
    environment.intensity = spy.float3(2)
    # Every camera ray misses, even when the three material families remain in the scene.
    camera = add_camera(scene, (0, 3, 0), (0, 4, 0), (0, 0, 1), 30, 4, 4)
    scene.update()
    psdr = renderer_for(device, scene, camera)
    image = psdr.forward(spp=1, seed=19, terms=())["primal"]
    np.testing.assert_allclose(image, np.broadcast_to(2 * color, image.shape), rtol=1e-6)
    slot = psdr.light_slot(environment, "intensity")
    for split in (True, False):
        psdr.split_interior = split
        vertices, params = backward(psdr, spp=1)
        np.testing.assert_array_equal(vertices, 0)
        np.testing.assert_allclose(params[slot:slot + 3], color / 3, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(params[psdr.texture_slot(environment)].sum(), 2.0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_cross_family_emissive_triangle_derivative(device_type):
    device = device_for(device_type)
    scene, camera, _, _, light = panel_scene(device)
    light.active = False
    emitter = scene.create_material("OpenPBRMaterial", f2.Properties({
        "base_color": spy.float3(0.5), "emission_luminance": 4.0, "emission_color": spy.float3(1),
    }))
    # The patch lies outside the camera view and shines onto the receiver panels.
    add_quad_y(scene, -2.5, -1.5, -0.5, 0.5, 1.5, emitter, up=False)
    scene.update()
    psdr = renderer_for(device, scene, camera)
    adjoint = np.zeros((psdr.height, psdr.width, 3), np.float32)
    adjoint[:, :2, :] = adjoint[:, -2:, :] = 1
    adjoint /= adjoint.sum()
    # Only the outer Standard/SpongeCake panels contribute; the OpenPBR panel is masked out.
    _, params = backward(psdr, adjoint, spp=8)
    numerical = difference(scene, psdr, lambda value: emitter.__setitem__("emission_luminance", value),
                           4.0, adjoint, epsilon=0.125, spp=8)
    check_derivative(params[psdr.material_slot(emitter, "emission_luminance")], numerical,
                     "OpenPBR emitter seen through other BSDF families", tolerance=0.01)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_mixed_geometry_forward_reverse_identity(device_type):
    device = device_for(device_type)
    scene = f2.Scene.create(device)
    materials = materials_for(scene)
    instances = [add_quad_y(scene, -2, 2, -2, 2, 0, materials[0][0]),
                 add_box(scene, (-0.75, 0.05, -0.3), (-0.1, 0.7, 0.35), materials[1][0]),
                 add_box(scene, (0.1, 0.05, -0.3), (0.75, 0.7, 0.35), materials[2][0])]
    # This material has no PSDR BSDF derivatives, but its geometric derivatives must still be visited.
    other = scene.create_material(f2.StandardSpecGlossMaterial, f2.Properties({
        "diffuse_factor": spy.float3(0.5), "specular_factor": spy.float3(0), "glossiness_factor": 0.2,
    }))
    materials.append((other, None))
    instances.append(add_box(scene, (-0.25, 0.05, -1.0), (0.25, 0.35, -0.6), other))
    light = _entity(scene, (-0.5, 2, -0.7)).create_component(f2.PointLight)
    light.intensity = spy.float3(3)
    camera = add_camera(scene, (0, 1.8, -3), (0, 0.3, 0), (0, 1, 0), 45, 12, 12)
    scene.update()
    # The optional backward-only clamp intentionally breaks the forward/reverse identity.
    psdr = renderer_for(device, scene, camera, max_bounces=2, vertex_grad_clamp=0)
    deep = psdr.forward(spp=8, seed=31, terms=(), antithetic=False)["primal"]
    shallow = renderer_for(device, scene, camera, max_bounces=1).forward(
        spp=8, seed=31, terms=(), antithetic=False)["primal"]
    assert deep.sum() > shallow.sum() + 1e-5, "The geometry fixture must contain indirect surface paths"
    adjoint = np.broadcast_to(np.linspace(0.2, 1, psdr.width, dtype=np.float32)[None, :, None],
                             (psdr.height, psdr.width, 3)).copy() / (psdr.width * psdr.height)
    vertices, _ = backward(psdr, adjoint, seed=31, spp=8)
    for instance, (material, _) in zip(instances, materials):
        selection = psdr.vertex_range(instance)
        assert np.linalg.norm(vertices[selection]) > 0, material.slang_type_name
        tangent = np.zeros_like(vertices)
        tangent[selection] = [0.2, 0.3, -0.1]
        psdr.set_tangents(vertex=tangent)
        derivative = psdr.forward(spp=8, seed=31, terms=("interior",), antithetic=False)["interior"]
        forward = float(np.sum(derivative * adjoint))
        reverse = float(np.sum(vertices * tangent))
        assert np.isfinite(forward) and np.isfinite(reverse)
        np.testing.assert_allclose(reverse, forward, rtol=3e-3, atol=2e-5, err_msg=material.slang_type_name)


def test_mixed_light_tracing_derivatives_and_frame_merging(panels):
    scene, materials, _, light, psdr = panels
    try:
        psdr.set_light_tracing(paths_per_pixel=2)
        psdr.merge_light_frames = True
        expected = backward(psdr)
        for material, parameter in materials:
            slot = psdr.material_slot(material, parameter)
            numerical = difference(scene, psdr, lambda value: material.__setitem__(parameter, spy.float3(value)), 0.5)
            check_derivative(expected[1][slot:slot + 3].sum(), numerical, "light tracing " + material.slang_type_name)
        slot = psdr.light_slot(light, "emission")
        numerical = difference(scene, psdr, lambda value: setattr(light, "intensity", spy.float3(value)), 3.0)
        check_derivative(expected[1][slot:slot + 3].sum(), numerical, "light tracing shared emission", tolerance=0.01)
        psdr.merge_light_frames = False
        for actual, reference in zip(backward(psdr), expected):
            np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-6)
    finally:
        psdr.merge_light_frames = True
        psdr.set_light_tracing(paths_per_pixel=0)


def family_scene(device, families, other=False):
    """Panels of the given material families (and a StandardSpecGlossMaterial panel with other=True)."""
    scene = f2.Scene.create(device)
    materials = [material for material, _ in materials_for(scene, families)]
    if other:
        materials.append(scene.create_material(f2.StandardSpecGlossMaterial, f2.Properties({
            "diffuse_factor": spy.float3(0.5), "specular_factor": spy.float3(0), "glossiness_factor": 0.2})))
    for i, material in enumerate(materials):
        x0 = -3 + 6 * i / len(materials)
        add_quad_y(scene, x0, x0 + 6 / len(materials), -3, 3, 0, material)
    camera = add_camera(scene, (0, 3, 0), (0, 0, 0), (0, 0, 1), 45, 12, 8)
    scene.update()
    return scene, camera


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_backward_programs_split_by_material_family(device_type):
    """Every heavy backward kernel runs one program per material family whenever a program could hold more than one
    family's reverse-mode BSDFs: all three took DXC (D3D12) hours to compile, and NVIDIA's Vulkan driver (Windows)
    rejects a program with OpenPBR and SpongeCake ("failed to validate local memory"). With trace_ray a single
    family keeps one program (HAS_*_MATERIAL); ray_query programs are shared across scenes, so they always split."""
    device = device_for(device_type)
    standard, openpbr, spongecake = MATERIAL_FAMILIES.values()
    cases = [("trace_ray", tuple(MATERIAL_FAMILIES), False, ((standard, True), (openpbr, False), (spongecake, False))),
             ("trace_ray", ("OpenPBRMaterial", "SpongeCakeMaterial"), False, ((openpbr, True), (spongecake, False))),
             ("trace_ray", ("OpenPBRMaterial", "SpongeCakeMaterial"), True,
              ((openpbr, True), (spongecake, False), (MATERIAL_OTHER, False))),
             ("trace_ray", ("SpongeCakeMaterial",), True, ((None, True),)),
             ("trace_ray", ("OpenPBRMaterial",), False, ((None, True),))]
    if device.has_feature(spy.Feature.ray_query):
        cases += [("ray_query", ("OpenPBRMaterial",), False, ((openpbr, True),)),
                  ("ray_query", ("StandardMaterial",), True, ((standard, True), (MATERIAL_OTHER, False))),
                  ("ray_query", (), True, ((MATERIAL_OTHER, True),)),
                  ("ray_query", tuple(MATERIAL_FAMILIES), False,
                   ((standard, True), (openpbr, False), (spongecake, False)))]
    for traversal, families, other, expected in cases:
        scene, camera = family_scene(device, families, other)
        psdr = PSDR(device, scene, camera, max_bounces=1, light_sampler=f2.UniformLightSampler(), traversal=traversal)
        for name in sorted(MATERIAL_BACKWARD_KERNELS):
            label = f"{traversal} {families} other={other} {name}"
            assert psdr._kernel_specializations(name, True) == expected, label
            assert psdr._kernel_specializations(name, False) == ((None, True),), label


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_two_family_backward_derivatives(device_type):
    """OpenPBR and SpongeCake without StandardMaterial: before the per-family split for two families, trace_ray
    compiled both reverse-mode BSDFs into one backward program, which NVIDIA's Vulkan driver (Windows) rejected."""
    device = device_for(device_type)
    scene = f2.Scene.create(device)
    materials = materials_for(scene, ("OpenPBRMaterial", "SpongeCakeMaterial"))
    for (x0, x1), (material, _) in zip(((-3, 0), (0, 3)), materials):
        add_quad_y(scene, x0, x1, -3, 3, 0, material)
    light = _entity(scene, (-0.5, 2, 0.7)).create_component(f2.PointLight)
    light.intensity = spy.float3(3)
    camera = add_camera(scene, (0, 3, 0), (0, 0, 0), (0, 0, 1), 45, 12, 8)
    scene.update()
    psdr = renderer_for(device, scene, camera)
    _, params = backward(psdr)
    for material, parameter in materials:
        slot = psdr.material_slot(material, parameter)
        numerical = difference(scene, psdr, lambda value: material.__setitem__(parameter, spy.float3(value)), 0.5)
        check_derivative(params[slot:slot + 3].sum(), numerical, material.slang_type_name)
