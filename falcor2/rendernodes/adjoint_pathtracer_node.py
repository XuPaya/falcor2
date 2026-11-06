# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adjoint (light-tracing) path tracer: the reference path tracer plus a light tracer, combined by multiple importance
sampling as bidirectional path tracing does (Veach 1997, ch. 10).

The light tracer emits from every scene light - an emitter chosen by the path tracer's light sampler (its emission
selection), then a position and a direction (falcor2 render/light_emission.slang): analytic lights, emissive
triangles and environment lights - traces the light path by adjoint BSDF sampling and connects every surface vertex to
the pinhole camera (atomic splats into a float buffer). This is the t = 1 strategy of bidirectional path tracing; it
reaches what the path tracer cannot: light through delta and peaked transmission (glass panes, sheer fabric) whose
next-event shadow rays are blocked.

Estimator: for every path the strategies that can generate it - the camera path hitting the emitter (s = 0), next-event
estimation (s = 1) and light tracing (t = 1) - are weighted by the balance heuristic. Each tracer evaluates the other
strategies' densities of its own samples (partial density ratios accumulated along the subpaths, the dVC quantities of
Georgiev 2012 restricted to these strategies); a strategy that cannot generate a path has density zero.

Emission directions of finite lights outside the emission target sphere (default: the scene's bounding sphere) are
sampled uniformly over the sphere's projected disk, as pbrt samples distant lights over the scene's disk; light leaving
outside the target has light-tracing density zero (only the path tracer covers it). Limits: pinhole camera (depth of
field: light tracing off for those paths), scattering media end light paths, no nested-interior priorities.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Optional

import slangpy as spy
from slangpy import CommandEncoder, Device

import falcor2 as f2
from falcor2.editor.scene_shader import (SceneShaderHelper, bind_light_sampler, light_sampler_key,
                                         light_sampler_prelude, scene_ray_tracing)
from falcor2.rendergraph import Container, ContainerSpec, RenderNode
from falcor2.rendernodes.reference_pathtracer_node import ReferencePathTracerNode

ADJOINT_MODULE_PATH = "falcor2/rendernodes/adjoint_pathtracer.slang"
ENVIRONMENT_LIGHTS = (f2.ConstantLight, f2.EnvMapLight, f2.DistantLight)


def scene_bounding_sphere(scene: f2.Scene) -> tuple[tuple[float, float, float], float]:
    """Bounding sphere (center, radius) of the scene's geometry instances, as the light tracer's emission uses it (PSDR's
    light term uses the same function, so that both emit the same light paths)."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for component in scene.components:
        if not isinstance(component, f2.GeometryInstance):
            continue
        aabb = component.entity.world_aabb
        if hasattr(aabb, "valid") and not aabb.valid:
            continue
        a, b = aabb.min, aabb.max
        for i in range(3):
            if math.isfinite(a[i]) and math.isfinite(b[i]) and a[i] <= b[i]:
                lo[i] = min(lo[i], a[i])
                hi[i] = max(hi[i], b[i])
    if not all(math.isfinite(v) for v in lo + hi):
        lo, hi = [-1.0] * 3, [1.0] * 3
    center = [(lo[i] + hi[i]) * 0.5 for i in range(3)]
    radius = 0.5 * math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    return (center[0], center[1], center[2]), max(radius * 1.001, 1e-3)


class AdjointMode(IntEnum):
    """What the node renders."""

    #: Path tracer + light tracer, combined by multiple importance sampling: the full image.
    combined = 0
    #: Light tracer alone, weight one (validation against the path tracer; no light path reaches the camera
    #: directly from an emitter, and delta vertices next to the camera are not connected).
    light_only = 1


class AdjointPathTracerNode(RenderNode):
    def __init__(self, device: Device):
        super().__init__()
        self._device = device
        self.path_tracer = ReferencePathTracerNode(device)
        self.path_tracer.enable_nee = True
        self.path_tracer.enable_light_tracer_mis = True
        self._scene_shader = SceneShaderHelper(device)
        self._source_module = spy.Module.load_from_file(device, ADJOINT_MODULE_PATH)
        self._module = None
        self._funcs: dict[str, Any] = {}
        self._funcs_key = None
        self._accumulation: spy.Buffer | None = None
        self._output = None
        self._bounds_key = None
        self._bounds = (spy.float3(0.0), 1.0)
        self._mode = AdjointMode.combined
        self._light_paths_per_pixel = 1.0
        self._emission_target: tuple[tuple[float, float, float], float] | None = None
        self._enable_russian_roulette = True
        #: Light paths traced in the last frame.
        self.light_path_count = 0

    @classmethod
    def create(cls, device: Device) -> "AdjointPathTracerNode":
        return cls(device)

    # Settings ----------------------------------------------------------------------------------------

    @property
    def mode(self) -> AdjointMode:
        """combined (path tracer + light tracer) or light_only (validation)."""
        return self._mode

    @mode.setter
    def mode(self, value: AdjointMode) -> None:
        self._mode = AdjointMode(value)
        self.path_tracer.enable_light_tracer_mis = self._mode == AdjointMode.combined

    @property
    def light_sampler(self) -> f2.LightSampler:
        """The light sampler of both tracers (next-event estimation and emission selection)."""
        return self.path_tracer.light_sampler

    @light_sampler.setter
    def light_sampler(self, value: f2.LightSampler) -> None:
        self.path_tracer.light_sampler = value

    @property
    def max_depth(self) -> int:
        """Maximum path depth (path tracer semantics: paths with up to max_depth - 1 surface vertices)."""
        return self.path_tracer.max_depth

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        self.path_tracer.max_depth = value

    @property
    def rr_depth(self) -> int:
        """Depth at which Russian roulette starts (both tracers)."""
        return self.path_tracer.rr_depth

    @rr_depth.setter
    def rr_depth(self, value: int) -> None:
        self.path_tracer.rr_depth = value

    @property
    def enable_russian_roulette(self) -> bool:
        return self._enable_russian_roulette

    @enable_russian_roulette.setter
    def enable_russian_roulette(self, value: bool) -> None:
        self._enable_russian_roulette = bool(value)
        self.path_tracer.enable_russian_roulette = bool(value)


    @property
    def light_paths_per_pixel(self) -> float:
        """Light paths per frame, relative to the pixel count."""
        return self._light_paths_per_pixel

    @light_paths_per_pixel.setter
    def light_paths_per_pixel(self, value: float) -> None:
        self._light_paths_per_pixel = max(0.0, float(value))


    @property
    def emission_target(self) -> tuple[tuple[float, float, float], float] | None:
        """(center, radius) sphere finite lights outside it aim their particles at; None: the scene's bounding sphere.
        Light leaving outside it is covered by the path tracer only (a sphere around a window concentrates the sun's
        particles there)."""
        return self._emission_target

    @emission_target.setter
    def emission_target(self, value) -> None:
        if value is None:
            self._emission_target = None
        else:
            center, radius = value
            self._emission_target = (tuple(float(c) for c in center), float(radius))

    # Helpers -----------------------------------------------------------------------------------------

    def _scene_bounds(self, scene: f2.Scene) -> tuple[spy.float3, float]:
        """Bounding sphere of the scene's geometry instances (cached per scene update)."""
        key = (id(scene), scene.update_generation)
        if key == self._bounds_key:
            return self._bounds
        center, radius = scene_bounding_sphere(scene)
        self._bounds = (spy.float3(*center), radius)
        self._bounds_key = key
        return self._bounds

    @staticmethod
    def _analytic_light_count(scene: f2.Scene) -> int:
        return sum(
            1
            for c in scene.components
            if isinstance(c, f2.Light) and c.active and not isinstance(c, ENVIRONMENT_LIGHTS)
        )

    def _get_funcs(self, scene: f2.Scene) -> dict[str, Any]:
        module = self._scene_shader.get_module(scene, self._source_module)
        pt = self.path_tracer
        use_ray_query = self._device.has_feature(spy.Feature.ray_query)
        constants = {
            "ENABLE_INTERIOR_TRACKING": pt.enable_interior_tracking,
            "ENABLE_HOMOGENEOUS_MEDIA": pt.enable_homogeneous_media,
            "ENABLE_RUSSIAN_ROULETTE": self._enable_russian_roulette,
        }
        sampler = pt.light_sampler
        key = (module, use_ray_query, tuple(sorted(constants.items())), light_sampler_key(sampler))
        if key != self._funcs_key:
            conformances = scene.requirements.type_conformances
            # The traversal is a link-time type, so only trace_ray programs contain TraceRay.
            traversal = "InlineAdjointTraversal" if use_ray_query else "PipelineAdjointTraversal"
            prelude = (light_sampler_prelude(sampler)
                       + f"export struct AdjointTraversal : IAdjointTraversal = {traversal};\n")
            self._funcs = {
                name: getattr(module, name)
                .constants(constants)
                .prelude(prelude)
                .type_conformances(conformances)
                .write(self._scene_shader.bind_scene)
                for name in ("trace_light_paths", "resolve", "clear_accumulation")
            }
            if not use_ray_query:
                self._funcs["trace_light_paths"] = scene_ray_tracing(
                    self._funcs["trace_light_paths"], scene, "adjoint"
                )
            self._funcs_key = key
        return self._funcs

    def _buffers(self, pixel_count: int) -> None:
        usage = spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource
        if self._accumulation is None or self._accumulation.size < pixel_count * 12:
            self._accumulation = self._device.create_buffer(size=pixel_count * 12, usage=usage)

    def _get_output(self, camera: f2.Camera) -> Any:
        spec = ContainerSpec.texture2d(spy.Format.rgba32_float, (camera.height, camera.width))
        self._output = Container.create_temp(self._device, spec, current=self._output)
        return self._output

    # Rendering ---------------------------------------------------------------------------------------

    def _exec(
        self,
        scene: f2.Scene,
        camera: f2.Camera,
        iteration: int = 0,
        subpixel_offset: Any = spy.float2(0.0, 0.0),
        subpixel_random_jitter: float = 1.0,
        cmd: Optional[CommandEncoder] = None,
    ) -> tuple[Any, dict[str, Any | None]]:
        """Render one frame (one path-tracer sample per pixel + one light-tracer pass)."""
        # Keep the path tracer, accumulation clear, light tracer and resolve in one
        # submission. A caller-provided encoder remains owned by the caller.
        owns_cmd = cmd is None
        if owns_cmd:
            cmd = self._device.create_command_encoder()
        width, height = int(camera.width), int(camera.height)
        output = self._get_output(camera)
        guides: dict[str, Any | None] = {}
        pt_color = None
        # MIS settings shared with the path tracer (it weighs its samples against the light tracer's).
        mis_paths = float(int(round(self._light_paths_per_pixel * width * height)))
        bounds_center, bounds_radius = self._scene_bounds(scene)
        if self._emission_target is None:
            mis_target = ((bounds_center.x, bounds_center.y, bounds_center.z), bounds_radius)
        else:
            mis_target = self._emission_target
        self.path_tracer.set_light_tracer(mis_paths, mis_target[0], mis_target[1],
                                         (bounds_center.x, bounds_center.y, bounds_center.z), bounds_radius)
        if self._mode == AdjointMode.combined:
            self.path_tracer.output_spec = ContainerSpec.texture2d(spy.Format.rgba32_float, (height, width))
            pt_color, guides = self.path_tracer._exec(
                scene,
                camera,
                iteration=iteration,
                subpixel_offset=subpixel_offset,
                subpixel_random_jitter=subpixel_random_jitter,
                cmd=cmd,
            )

        pixel_count = width * height
        path_count = int(round(self._light_paths_per_pixel * pixel_count))
        self.light_path_count = path_count
        self._buffers(pixel_count)
        render_camera = camera.calc_uniforms(width, height)
        scene_center, scene_radius = self._scene_bounds(scene)
        if self._emission_target is None:
            target_center, target_radius = scene_center, scene_radius
        else:
            target_center, target_radius = spy.float3(*self._emission_target[0]), self._emission_target[1]
        emission = {
            "target_center": target_center,
            "target_radius": float(target_radius),
            "scene_center": scene_center,
            "scene_radius": float(scene_radius),
        }
        settings = {
            "max_depth": self.path_tracer.max_depth,
            "rr_depth": self.path_tracer.rr_depth,
            "mode": 0 if self._mode == AdjointMode.combined else 1,
            "light_path_count": path_count,
            "contribution_scale": 1.0 / path_count if path_count > 0 else 0.0,
            # The path tracer renders one sample per pixel and frame.
            "mis_path_count": float(path_count),
            "emission": emission,
            "iteration": int(iteration) & 0xFFFFFFFF,
            "camera_ray_t_max": 1000.0,
        }
        funcs = self._get_funcs(scene)
        sampler = self.path_tracer.light_sampler
        if self._mode != AdjointMode.combined:
            sampler.update(scene, cmd)      # the path tracer updates it otherwise

        def bind(cursor: Any) -> None:
            bind_light_sampler(cursor, sampler)
            block = cursor["g_light_tracer"]
            for name, value in settings.items():
                block["settings"][name] = value
            block["camera"] = render_camera
            block["accumulation"] = self._accumulation

        float_count = pixel_count * 3
        funcs["clear_accumulation"].write(bind)(
            tid=spy.grid((float_count,)), count=float_count, _append_to=cmd
        )
        if path_count > 0:
            funcs["trace_light_paths"].write(bind)(tid=spy.grid((path_count,)), _append_to=cmd)
        funcs["resolve"].write(bind)(
            ray_sampler=render_camera,
            path_tracer_color=pt_color if pt_color is not None else spy.float4(0.0),
            add_path_tracer=pt_color is not None,
            output=Container.to_render_layout(output),
            _append_to=cmd,
        )
        if owns_cmd:
            self._device.submit_command_buffer(cmd.finish())
        return output, guides
