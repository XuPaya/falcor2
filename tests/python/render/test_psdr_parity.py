# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PSDR's primal image is the forward renderer's image: same seed and spp, ReferencePathTracerNode (next-event
estimation + MIS) or AdjointPathTracerNode (with PSDR light tracing), accumulated by AccumulatorNode, against
PSDR.forward(spp, seed)["primal"] with max_bounces = max_depth - 1.

The scenes exercise what the reference path tracer does besides plain surface scattering: emissive triangles (two-sided
emitters), rect / sphere / sun-disk lights and an environment map with both light samplers, glossy OpenPBR and
StandardMaterial surfaces, opacity cut-outs (alpha test), closed glass and a scattering medium
(interior tracking, volume vertices), Russian roulette, and the adjoint tracer's light paths through a SpongeCake sheer.

Parity is per pixel: the same paths with the same random numbers. With precise floating point (no contraction into
fused multiply-adds) the camera paths agree within float32 rounding error. With the default (fast) compile the same
expressions are contracted differently in the forward renderer's ray-tracing pipeline and PSDR's compute kernels: rounding-level
differences, which rarely flip a sampling decision later in a path (that sample then differs). With light tracing the
splats' float atomics add in a different order (rounding level in splatted pixels).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.psdr import PSDR
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import AccumulatorNode, ReferencePathTracerNode
from falcor2.rendernodes.adjoint_pathtracer_node import AdjointPathTracerNode

from .test_adjoint_pathtracer import (_entity, add_box, add_camera, add_mesh, add_quad_y, add_sun, diffuse, openpbr,
                                      spongecake, window_scene)

# OpenPBR scenes do not compile for CUDA (nvrtc 11.6), also in the reference path tracer.
DEVICE_TYPES = [t for t in helpers.DEFAULT_DEVICE_TYPES if t != spy.DeviceType.cuda]
FP_MODES = {"default": spy.SlangFloatingPointMode.default, "precise": spy.SlangFloatingPointMode.precise}
# Precise-mode parity: separate shader pipelines can round float32 results differently. Windows Vulkan (ray_query)
# needs rtol 8.7e-7 (max_depth 6) and 2.4e-6 (max_depth 16 and 32); a flipped sampling decision differs by far more.
PRECISE_TOLERANCE = {"rtol": 1e-5, "atol": 1e-8}


def standard(scene: f2.Scene, color, roughness: float, metallic: float = 0.0) -> f2.Material:
    return scene.create_material("StandardMaterial", f2.Properties({
        "base_color_factor": spy.float3(*color), "roughness_factor": roughness, "metallic_factor": metallic}))


def parity_scene(device: spy.Device, variant: str, size: int = 64):
    """A closed room (two-sided diffuse box) with an open window: a sun disk and an environment map outside, a rect
    light, a sphere light, an emissive (two-sided) bulb mesh, glossy OpenPBR and StandardMaterial boxes.
    variant "opacity": two alpha-tested sheets, one kept and one cut away; "media": a closed glass box and a closed
    scattering (subsurface) box."""
    scene = f2.Scene.create(device)
    wall = diffuse(scene, 0.6)
    # Room with a window opening in the +x wall.
    add_quad_y(scene, -2.0, 2.0, -2.0, 2.0, 0.0, wall)
    add_quad_y(scene, -2.0, 2.0, -2.0, 2.0, 2.5, wall, up=False)
    for z in (-2.0, 2.0):
        add_mesh(scene, [[-2, 0, z], [-2, 2.5, z], [2, 0, z], [2, 2.5, z]], [[0, 1, 2], [2, 1, 3]], wall)
    add_mesh(scene, [[-2, 0, -2], [-2, 2.5, -2], [-2, 0, 2], [-2, 2.5, 2]], [[0, 1, 2], [2, 1, 3]], wall)
    # +x wall with a 1 x 1 m window centred at (y 1.4, z 0)
    x = 2.0
    for y0, y1, z0, z1 in ((0.0, 0.9, -2.0, 2.0), (1.9, 2.5, -2.0, 2.0), (0.9, 1.9, -2.0, -0.5), (0.9, 1.9, 0.5, 2.0)):
        add_mesh(scene, [[x, y0, z0], [x, y1, z0], [x, y0, z1], [x, y1, z1]], [[0, 1, 2], [2, 1, 3]], wall)
    add_box(scene, (-0.5, 0.0, -0.4), (0.3, 0.7, 0.4), openpbr(scene, base_color=(0.7, 0.4, 0.2), specular_roughness=0.3,
                                                               coat_weight=0.5, coat_roughness=0.1))
    add_box(scene, (0.6, 0.0, 0.5), (1.1, 1.2, 1.0), standard(scene, (0.3, 0.5, 0.7), 0.4, 0.3))
    # Emissive two-sided bulb (OpenPBR emission) hanging from the ceiling.
    bulb = openpbr(scene, base_color=(0.9, 0.9, 0.9), emission_luminance=30.0, emission_color=(1.0, 0.8, 0.6))
    add_box(scene, (-1.2, 2.0, 0.9), (-1.0, 2.2, 1.1), bulb)
    rect = _entity(scene, (0.0, 2.45, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)).create_component(f2.RectLight)
    rect.radiance = spy.float3(6.0)
    rect.width, rect.height = 0.8, 0.6
    sphere = _entity(scene, (-1.3, 1.6, -1.2)).create_component(f2.SphereLight)
    sphere.radiance = spy.float3(20.0, 15.0, 10.0)
    sphere.radius = 0.1
    add_sun(scene, direction=(-0.8, -0.5, 0.2), irradiance=20.0)
    texels = np.ones((16, 32, 4), np.float32)
    texels[:8, :, :3] *= np.array([0.6, 0.8, 1.2], np.float32)
    texels[8:, :, :3] *= 0.2
    path = Path(tempfile.mkdtemp(prefix="psdr_parity_")) / "sky.exr"
    spy.Bitmap(texels).write(str(path))
    env = scene.create_entity().create_component(f2.EnvMapLight)
    env.env_map_path = str(path)
    if variant == "opacity":
        kept = openpbr(scene, base_color=(0.9, 0.9, 0.85), specular_roughness=0.5, opacity_factor=0.5,
                       opacity_threshold=0.4)
        add_mesh(scene, [[1.9, 0.9, -0.6], [1.9, 2.0, -0.6], [1.9, 0.9, 0.6], [1.9, 2.0, 0.6]], [[0, 1, 2], [2, 1, 3]], kept)
        cut = openpbr(scene, base_color=(0.4, 0.4, 0.4), opacity_factor=0.5, opacity_threshold=0.6)
        add_mesh(scene, [[0.5, 0.0, -1.9], [0.5, 1.5, -1.9], [1.5, 0.0, -1.9], [1.5, 1.5, -1.9]], [[0, 2, 1], [2, 3, 1]],
                 cut)
    elif variant == "media":
        add_box(scene, (-0.3, 0.7, -0.3), (0.1, 1.1, 0.1), openpbr(scene, transmission_weight=1.0, specular_roughness=0.0,
                                                                   specular_ior=1.5))
        add_box(scene, (0.7, 1.2, 0.6), (1.0, 1.5, 0.9), openpbr(scene, base_color=(0.9, 0.7, 0.5), subsurface_weight=1.0,
                                                                 subsurface_color=(0.9, 0.7, 0.5), subsurface_radius=0.05))
    camera = add_camera(scene, (1.5, 1.4, -1.7), (-0.4, 0.8, 0.6), (0.0, 1.0, 0.0), 75.0, size, size)
    scene.update()
    return scene, camera


def forward_image(device, scene, camera, node, spp: int, seed: int) -> np.ndarray:
    accumulator = AccumulatorNode.create(device)
    out = None
    for i in range(spp):
        color, _ = node._exec(scene, camera, iteration=seed * 65536 + i)
        out = accumulator._exec(color)
    return np.asarray(out.to_numpy(), np.float64)[..., :3]


def path_tracer(device, camera, max_depth: int, sampler) -> ReferencePathTracerNode:
    node = ReferencePathTracerNode(device)
    node.enable_nee = True
    node.enable_mis = True
    node.max_depth = max_depth
    node.light_sampler = sampler
    node.output_spec = ContainerSpec.texture2d(spy.Format.rgba32_float, (camera.height, camera.width))
    return node


def compare(forward: np.ndarray, primal: np.ndarray, other_seed: np.ndarray, label: str):
    """Per-pixel parity: (fraction of pixels differing by more than 1e-4 relative, mean |diff| / mean |noise|)."""
    d = np.abs(forward - primal)
    rel = d / np.maximum(np.maximum(np.abs(forward), np.abs(primal)), 1e-6)
    bad = float(np.mean(np.any(rel > 1e-4, -1)))
    noise = float(np.abs(forward - other_seed).mean())
    ratio = float(d.mean()) / max(noise, 1e-30)
    print(f"{label}: pixels > 1e-4 rel {bad:.4f}, identical {np.mean(np.all(d == 0, -1)):.4f}, max |d| {d.max():.3g}, "
          f"mean |d| {d.mean():.3g} = {ratio:.2e} x noise, means {forward.mean():.5g} / {primal.mean():.5g}")
    return bad, ratio


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("fp", ["precise", "default"])
@pytest.mark.parametrize("variant", ["plain", "opacity", "media"])
@pytest.mark.parametrize("sampler", ["power", "uniform"])
def test_psdr_primal_matches_path_tracer(device_type: spy.DeviceType, fp: str, variant: str, sampler: str):
    device = helpers.get_device(device_type, floating_point_mode=FP_MODES[fp])
    spp, seed, max_depth = 4, 7, 6
    scene, camera = parity_scene(device, variant)
    make = {"power": f2.PowerLightSampler, "uniform": f2.UniformLightSampler}[sampler]
    node = path_tracer(device, camera, max_depth, make())
    forward = forward_image(device, scene, camera, node, spp, seed)
    other = forward_image(device, scene, camera, node, spp, seed + 1)
    psdr = PSDR(device, scene, camera, max_bounces=max_depth - 1, light_sampler=make())
    primal = psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"]
    bad, ratio = compare(forward, primal, other, f"{fp}/{variant}/{sampler}")
    if fp == "precise":
        np.testing.assert_allclose(forward, primal, **PRECISE_TOLERANCE, equal_nan=False)
    assert bad < 0.01
    assert ratio < 1e-2


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("max_depth", [16, 32])
def test_psdr_primal_matches_path_tracer_deep(device_type: spy.DeviceType, max_depth: int):
    """Deep paths (indoor depths; PSDR compiles its path records for the depth): no Russian roulette, so every path
    reaches max_depth in the closed room."""
    device = helpers.get_device(device_type, floating_point_mode=FP_MODES["precise"])
    spp, seed = 4, 7
    scene, camera = parity_scene(device, "plain")
    node = path_tracer(device, camera, max_depth, f2.PowerLightSampler())
    node.rr_depth = 1000
    forward = forward_image(device, scene, camera, node, spp, seed)
    other = forward_image(device, scene, camera, node, spp, seed + 1)
    psdr = PSDR(device, scene, camera, max_bounces=max_depth - 1, light_sampler=f2.PowerLightSampler(), rr_depth=None)
    primal = psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"]
    compare(forward, primal, other, f"precise/deep {max_depth}")
    np.testing.assert_allclose(forward, primal, **PRECISE_TOLERANCE, equal_nan=False)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("fp", ["precise", "default"])
def test_psdr_primal_matches_adjoint_tracer(device_type: spy.DeviceType, fp: str):
    """Light tracing: the sun through a SpongeCake sheer (delta + scattered transmission) and a room's lights."""
    device = helpers.get_device(device_type, floating_point_mode=FP_MODES[fp])
    spp, seed, max_depth = 4, 5, 5
    sheet = {"albedo": (0.95, 0.95, 0.95), "thickness": 0.6, "roughness": 0.15, "hg": True, "delta_transmission": True}
    for scene, camera in (window_scene(device, "sheet", size=64, sheet=sheet), parity_scene(device, "plain")):
        node = AdjointPathTracerNode(device)
        node.max_depth = max_depth
        node.light_sampler = f2.PowerLightSampler()
        forward = forward_image(device, scene, camera, node, spp, seed)
        other = forward_image(device, scene, camera, node, spp, seed + 1)
        psdr = PSDR(device, scene, camera, max_bounces=max_depth - 1, light_sampler=f2.PowerLightSampler())
        psdr.set_light_tracing(1.0)
        primal = psdr.forward(spp=spp, seed=seed, terms=(), antithetic=False)["primal"]
        bad, ratio = compare(forward, primal, other, f"{fp}/adjoint")
        # The light splats' float atomics add in a different order: rounding-level differences in splatted pixels.
        if fp == "precise":
            assert bad == 0.0
        assert ratio < 1e-2
