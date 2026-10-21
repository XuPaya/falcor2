# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PSDR renderer: the kernels of the Slang module falcor2.psdr.psdr (slang/falcor2/psdr/), their buffers and
the parameter layout of a scene."""

from typing import Optional

import numpy as np
import slangpy as spy

import falcor2 as f2
from falcor2.editor.scene_shader import SceneShaderHelper
from falcor2.psdr.edges import classify_edges, edge_topology, normalize_cdf
from falcor2.psdr.layout import (ENV_MAP_PARAMS, ENVIRONMENT_LIGHTS, LIGHT_PARAMS, LIGHT_SLOTS, MATERIAL_PARAMS,
                                 MATERIAL_SLOTS, MATERIAL_TEXTURES, MAX_DEPTH, NO_SLOT, PHOTON_EMISSIONS, TERMS,
                                 light_emission)
from falcor2.psdr.openpbr import generate_openpbr_headers

# Default vertex_grad_clamp (see PSDR.__init__); 0 disables the clamp.
VERTEX_GRAD_CLAMP = 0.0
# Shader parameters of the light term (light_paths.slang), bound only with light tracing on.
LIGHT_TRACING_KEYS = {"light_tracing", "light_derivatives", "light_contribution_scale", "light_sample_weight",
                      "light_mis_path_count", "light_emission", "light_primal", "light_interior"}
# Camera sample i of a pass with seed s is the reference path tracer's frame s * FRAMES_PER_SEED + i (interior.slang).
FRAMES_PER_SEED = 65536
# Samples per pixel per dispatch of the per-pixel derivative kernels (long dispatches hit the GPU timeout).
SAMPLES_PER_DISPATCH = 32


class PSDR:
    """Renders a falcor2 scene from a camera: the primal image, forward-mode derivative images and backward-mode
    gradients (see falcor2.psdr).

    The primal image is falcor2's forward renderer: the camera paths are ReferencePathTracerNode's (next-event
    estimation and MIS on, its other defaults: interior tracking with homogeneous media, Russian roulette from
    ``rr_depth`` (None: off), material opacity (cut-outs)) and, with light tracing, the light paths
    AdjointPathTracerNode's. ``forward(spp, seed)["primal"]`` is the AccumulatorNode mean of the forward renderer's
    frames seed * 65536 + i, i < spp, at max_depth = max_bounces + 1 (pixel-identical up to the compiled arithmetic and
    the order of the light splats' float atomics). The derivatives (interior term) are those of the same path samples in
    material form; volume scattering vertices, medium weights and Russian roulette are detached factors.

    ``light_sampler`` chooses the lights of next-event estimation (default: f2.PowerLightSampler, as the reference
    path tracer). Its emitted-power weights starve small lights next to a distant, physically large one (a sun
    disk); f2.UniformLightSampler samples every light alike.

    ``photon_count`` (default PHOTON_EMISSIONS): edge photons per pass of the secondary-edge term, spread over
    ``photon_edge_count`` edges of total length ``photon_edge_length`` (boundary and convex sharp edges of every
    mesh) in proportion to length. More photons per pass cut its variance; e.g. one per edge, or about one per
    photon_radius of edge length.

    ``vertex_grad_clamp`` (default VERTEX_GRAD_CLAMP; 0: off): backward mode bounds each path sample's interior-term
    vertex gradient to vertex_grad_clamp / scene extent x its dL/dI-weighted contribution. Sharp glossy lobes make
    the material-form vertex derivative heavy-tailed (rare samples 10^3-10^4 x the mean); the clamp trades a small
    bias for bounded variance. Forward-mode derivative images are not clamped.

    Light tracing (set_light_tracing(), off by default): light paths from the scene's lights connected to the camera
    (falcor2's adjoint light tracer), combined with the camera paths' emission and next-event estimates by multiple
    importance sampling as in bidirectional path tracing. They add the light that reaches the camera paths' surfaces
    only through delta or peaked transmission (the sun through glass or sheer curtains) to the primal image and to the
    interior term (material and light-emission derivatives; no geometric derivatives along light paths). See
    slang/falcor2/psdr/light_paths.slang.
    """

    def __init__(
        self,
        device: spy.Device,
        scene: f2.Scene,
        camera: f2.Camera,
        max_bounces: int = 2,
        ray_epsilon: Optional[float] = None,
        photon_radius: Optional[float] = None,
        vertex_grad_clamp: Optional[float] = None,
        check_visibility: bool = False,
        light_sampler: Optional[f2.LightSampler] = None,
        photon_count: int = PHOTON_EMISSIONS,
        rr_depth: Optional[int] = 3,
    ):
        if not 0 <= max_bounces <= MAX_DEPTH:
            raise ValueError(f"max_bounces must be in [0, {MAX_DEPTH}]")
        scene.update()
        self.device, self.scene, self.max_bounces = device, scene, max_bounces
        self.width, self.height = camera.width, camera.height
        self.camera = camera.calc_uniforms(self.width, self.height)
        self._helper = SceneShaderHelper(device)
        generate_openpbr_headers()
        module = self._helper.get_module(scene, "falcor2.psdr.psdr")
        conformances = scene.requirements.type_conformances
        type_ids = {c.type_name: c.id for c in conformances if c.interface_name == "IMaterial"}
        self._light_sampler = light_sampler or f2.PowerLightSampler()
        prelude = (self._light_sampler.shader_specialization_source
                   + f"export struct LightSampler : ILightSampler = {self._light_sampler.slang_type_name};\n")
        has_openpbr = "OpenPBRMaterial" in type_ids
        has_spongecake = "SpongeCakeMaterial" in type_ids

        def query(name, backward=False):
            constants = {"has_openpbr": has_openpbr, "has_spongecake": has_spongecake, "normal_maps": self._normal_maps,
                         "backward": backward}
            function = getattr(module, name).constants(constants).prelude(prelude).type_conformances(conformances)
            return function.write(self._helper.bind_scene)

        self._query = query
        self._normal_maps = False

        # Geometry instances, their vertex ranges and materials.
        self.instances = {}
        for component in scene.components:
            if isinstance(component, f2.GeometryInstance):
                for i in range(component.geometry_instance_count):
                    self.instances[component.geometry_instance_id + i] = (component, i)
        self._info = {gid: np.asarray(query("instance_info")(gid), dtype=np.uint32) for gid in self.instances}
        self.vertex_count = int(max((int(i[0] + i[1]) for i in self._info.values()), default=0))
        materials = [m for m in scene.materials]
        self.material_slots = {m.material_id: m.material_id * MATERIAL_SLOTS for m in materials}
        light_slot_base = (max((m.material_id for m in materials), default=-1) + 1) * MATERIAL_SLOTS

        # Lights in LightID order (as in LightSystem): the active analytic lights, then the active environment
        # lights, each in component order. The first environment map is differentiable.
        lights = [c for c in scene.components if isinstance(c, f2.Light) and c.active]
        self.lights = [light for light in lights if not isinstance(light, ENVIRONMENT_LIGHTS)]
        self.lights += [light for light in lights if isinstance(light, ENVIRONMENT_LIGHTS)]
        self.light_slots = {id(light): light_slot_base + LIGHT_SLOTS * i for i, light in enumerate(self.lights)}
        env_maps = [i for i, light in enumerate(self.lights) if isinstance(light, f2.EnvMapLight)]
        env_map_id = env_maps[0] if env_maps else NO_SLOT

        # Texels of the materials' and the environment map's textures, by texture index (shared textures share
        # their texels): four slots (RGBA) each, rows from the top.
        self._textures, sizes = {}, {}
        ids = [type_ids.get(name, NO_SLOT) for name in ("StandardMaterial", "OpenPBRMaterial")]
        for m in materials:
            for role, name in enumerate(MATERIAL_TEXTURES.get(m.slang_type_name, ())):
                if m.slang_type_name == "SpongeCakeMaterial":
                    info = query("spongecake_texture")(m.material_id, role, type_ids["SpongeCakeMaterial"])
                else:
                    info = query("material_texture")(m.material_id, role, *ids)
                index, width, height = np.asarray(info, np.uint32).tolist()
                if index != NO_SLOT:
                    self._textures[m.material_id, name], sizes[index] = index, (height, width)
        if env_maps:
            index, width, height = np.asarray(query("env_map_texture")(env_map_id), np.uint32).tolist()
            if index != NO_SLOT:
                self._textures[id(self.lights[env_map_id])], sizes[index] = index, (height, width)
        self._normal_maps = any(isinstance(key, tuple) and key[1] == "normal" for key in self._textures)
        self._texel_base = light_slot_base + LIGHT_SLOTS * len(self.lights)
        self._texture_sizes, self._texel_request = sizes, None
        texel_slots = self._layout_texels(None)
        # Gradient buffers holding accumulate=True passes not yet read by gradients().
        self._grads_pending = False

        self._topology = {}
        self._vertex_query = None
        self._edges_stale = False
        self._build_edges()
        if ray_epsilon is None:
            extent = max((np.abs(self._positions[g]).max() for g in self._positions), default=1.0)
            ray_epsilon = max(1e-5, 1e-6 * float(extent))
        if photon_radius is None:
            photon_radius = 1.25e-3 * self._scene_extent
        self.photon_radius = photon_radius
        if vertex_grad_clamp is None:
            vertex_grad_clamp = VERTEX_GRAD_CLAMP
        self.vertex_grad_clamp = vertex_grad_clamp

        buffer = self._buffer
        self._buffers = {
            "vertex_tangent": buffer(np.zeros((max(1, self.vertex_count), 3), np.float32)),
            "param_tangent": buffer(np.zeros(max(1, self.param_count), np.float32)),
            "vertex_grad": buffer(np.zeros(max(1, self.vertex_count) * 3, np.float32)),
            "param_grad": buffer(np.zeros(max(1, self.param_count), np.float32)),
            "dl_di": buffer(np.zeros((self.width * self.height, 3), np.float32)),
            "texel_slots": buffer(texel_slots),
        }
        for name in ("primal", "interior", "pixel_boundary", "primary_boundary", "secondary_boundary", "light_primal",
                     "light_interior", "primal_sum", "primal_residual"):
            self._buffers[name] = buffer(np.zeros((self.width * self.height, 3), np.float32))

        self._globals = {
            "max_bounces": max_bounces,
            "rr_depth": 0xFFFFFFFF if rr_depth is None else max(1, int(rr_depth)),
            "ray_epsilon": float(ray_epsilon),
            "standard_type_id": type_ids.get("StandardMaterial", 0xFFFFFFFF),
            "openpbr_type_id": type_ids.get("OpenPBRMaterial", 0xFFFFFFFF),
            "spongecake_type_id": type_ids.get("SpongeCakeMaterial", 0xFFFFFFFF),
            "light_slot_base": light_slot_base,
            "texel_slot_count": len(texel_slots),
            "env_map_id": env_map_id,
            "photon_radius": float(photon_radius),
            "max_relative_vertex_derivative": float(vertex_grad_clamp) / self._scene_extent,
            "check_visibility": check_visibility,
            "light_tracing": False,
            "light_derivatives": False,
        }
        self._light_tracing = None
        # Kernels by (name, backward), compiled on first use.
        self._kernels = {}
        self.photon_count = photon_count
        self._upload_scene_state()

    @property
    def photon_count(self) -> int:
        return self._globals["photon_count"]

    @photon_count.setter
    def photon_count(self, count: int):
        count = int(count)
        if count < 1:
            raise ValueError("photon_count must be positive")
        self._globals["photon_count"] = count
        self._buffers["photons"] = self._buffer(np.zeros(count * 12, np.float32), struct_size=48)
        self._buffers["photon_bounds"] = self._buffer(np.zeros((2 * count, 3), np.float32))
        self._photon_accel = None

    @property
    def light_tracing(self) -> Optional[dict]:
        """Light-tracing settings (see set_light_tracing()), or None (off)."""
        return self._light_tracing

    def set_light_tracing(self, paths_per_pixel: float = 1.0, target=None, **_unused):
        """Light paths from the scene's lights (see the class docstring); paths_per_pixel <= 0 turns them off.

        paths_per_pixel: light paths per pixel and sample (spp). target: ((x, y, z), radius) sphere that finite lights
        outside it aim their light paths at, e.g. around the windows, or None for the scene's bounding sphere (light
        leaving outside it is covered by the camera paths only: its light-tracing density is zero).
        """
        if paths_per_pixel is None or paths_per_pixel <= 0:
            self._light_tracing = None
            self._globals["light_tracing"] = False
            return
        self._light_tracing = dict(paths_per_pixel=float(paths_per_pixel), target=target)
        self._globals["light_tracing"] = True
        self._update_light_emission()

    def _light_path_count(self) -> int:
        """Light paths per frame (AdjointPathTracerNode's count)."""
        return int(round(self._light_tracing["paths_per_pixel"] * self.width * self.height))

    def _update_light_emission(self):
        """Emission settings of the light term from the scene's current bounds (as AdjointPathTracerNode's)."""
        from falcor2.rendernodes.adjoint_pathtracer_node import scene_bounding_sphere

        center, radius = scene_bounding_sphere(self.scene)
        target = self._light_tracing["target"]
        if target is None:
            target = (center, radius)
        count = self._light_path_count()
        self._globals.update(
            light_mis_path_count=float(count),
            light_contribution_scale=1.0 / count if count > 0 else 0.0,
            light_emission={
                "target_center": spy.float3(*map(float, target[0])),
                "target_radius": float(target[1]),
                "scene_center": spy.float3(*map(float, center)),
                "scene_radius": float(radius),
            },
        )

    @property
    def photon_edge_count(self) -> int:
        """Number of edges the secondary-edge photons are sampled on."""
        self._update_edges()
        return len(self._photon)

    @property
    def photon_edge_length(self) -> float:
        """Total length of the edges the secondary-edge photons are sampled on."""
        self._update_edges()
        return self._photon_length

    def update(self, geometry: bool = True):
        """Refresh the scene-dependent state after ``scene.update()``: light values and, with geometry, the
        edge lists (needed after vertex or transform changes; rebuilt when a pass next samples edges)."""
        if geometry:
            self._refresh_layout()
            self._edges_stale = True
        self._upload_scene_state()

    def _update_edges(self):
        """Rebuilds stale edge lists (only the primary and secondary terms sample them)."""
        if self._edges_stale:
            self._build_edges()
            self._edges_stale = False
            self._upload_edges()

    def _refresh_layout(self):
        """falcor2 moves a mesh in the global vertex buffer when its vertices are rewritten: re-read the instances'
        vertex ranges and, if they changed, resize the vertex buffers and drop the cached edge topology (global
        vertex ids). Otherwise gradients of moved meshes would go to stale ranges or past the buffer's end."""
        gids = list(self.instances)
        if not gids:
            return
        fn = self._query("instance_info").return_type(spy.Tensor)
        rows = fn(spy.Tensor.from_numpy(self.device, np.asarray(gids, np.uint32))).to_numpy().reshape(len(gids), 4)
        info = {gid: np.asarray(row, dtype=np.uint32) for gid, row in zip(gids, rows)}
        if all(np.array_equal(info[gid], self._info[gid]) for gid in gids):
            return
        if self._grads_pending:
            raise RuntimeError("The vertex layout changed with accumulated gradients unread: call gradients() first")
        self._info = info
        self.vertex_count = int(max(int(i[0] + i[1]) for i in info.values()))
        n = max(1, self.vertex_count)
        if self._buffers["vertex_grad"].size < n * 12:
            self._buffers["vertex_tangent"] = self._buffer(np.zeros((n, 3), np.float32))
            self._buffers["vertex_grad"] = self._buffer(np.zeros(n * 3, np.float32))
        self._topology = {}
        self._vertex_query = None

    def set_camera(self, camera: f2.Camera):
        """Render from another camera with the same resolution, e.g. for multi-view optimization."""
        if (camera.width, camera.height) != (self.width, self.height):
            raise ValueError("The camera resolution must not change")
        self.camera = camera.calc_uniforms(self.width, self.height)
        self.update()

    def _upload_scene_state(self):
        self._light_sampler.update(self.scene)
        values = [light_emission(light) for light in self.lights] or [[0.0, 0.0, 0.0]]
        self._write_buffer("light_value", np.asarray(values, np.float32))
        if not self._edges_stale:
            self._upload_edges()

    def _upload_edges(self):
        self._write_buffer("primary_edges", self._primary)
        self._write_buffer("photon_edges", self._photon)
        self._globals.update(primary_edge_count=len(self._primary), primary_edge_length=self._primary_length,
                             photon_edge_count=len(self._photon), photon_edge_length=self._photon_length)

    # ------------------------------------------------------------------
    # Parameter layout

    def vertex_range(self, instance: f2.GeometryInstance, sub_mesh: int = 0) -> slice:
        """Global vertex range of a geometry instance's sub-mesh."""
        info = self._info[instance.geometry_instance_id + sub_mesh]
        return slice(int(info[0]), int(info[0] + info[1]))

    def material_slot(self, material: f2.Material, name: str) -> int:
        """First parameter slot of a material factor (see MATERIAL_PARAMS)."""
        return self.material_slots[material.material_id] + MATERIAL_PARAMS[material.slang_type_name][name]

    def light_slot(self, light: f2.Light, name: str = "emission") -> int:
        """First of the three parameter slots of an analytic light's emission (intensity or radiance) or
        world-space translation (LIGHT_PARAMS), or of an EnvMapLight's intensity or rotation (ENV_MAP_PARAMS)."""
        params = ENV_MAP_PARAMS if isinstance(light, f2.EnvMapLight) else LIGHT_PARAMS
        return self.light_slots[id(light)] + params[name]

    def texture_slot(self, owner, name: str = "base_color") -> slice:
        """Parameter slots of the texels of a material's texture (see MATERIAL_TEXTURES) or of an EnvMapLight's
        map: (height, width, 4) linear RGBA values in row-major order."""
        key = id(owner) if isinstance(owner, f2.Light) else (owner.material_id, name)
        index = self._textures[key]
        if index not in self._texels:
            raise KeyError(f"Texture {key} has no texel parameters (see set_texel_gradients())")
        first, height, width = self._texels[index]
        return slice(first, first + 4 * width * height)

    def set_texel_gradients(self, owners=None):
        """Texel parameters (gradients, tangents, texture_slot()) only for the textures of ``owners``, keys as
        texture_slot() takes them: (material, name) pairs or EnvMapLights; None (the default): every texture. The other
        textures cost no texel derivatives or gradient memory (a scene's textures can need gigabytes of slots)."""
        indices = None
        if owners is not None:
            keys = [id(o) if isinstance(o, f2.Light) else (o[0].material_id, o[1]) for o in owners]
            indices = frozenset(self._textures[k] for k in keys if k in self._textures)
        if indices == self._texel_request:
            return
        if self._grads_pending:
            raise RuntimeError("The texel layout changed with accumulated gradients unread: call gradients() first")
        self._texel_request = indices
        self._write_buffer("texel_slots", self._layout_texels(indices))
        n = max(1, self.param_count)
        for name in ("param_tangent", "param_grad"):
            if self._buffers[name].size != 4 * n:
                self._buffers[name] = self._buffer(np.zeros(n, np.float32))

    def _layout_texels(self, indices) -> np.ndarray:
        """Texel slots of the textures with these indices (None: every texture), packed after the light slots in
        texture order; returns the texel slot table (NO_SLOT: no texel derivatives) and sets param_count."""
        slot, self._texels = self._texel_base, {}
        for index, (height, width) in sorted(self._texture_sizes.items()):
            if indices is None or index in indices:
                self._texels[index] = (slot, height, width)
                slot += 4 * width * height
        self.param_count = slot
        table = np.full(max(self._texture_sizes, default=0) + 1, NO_SLOT, np.uint32)
        for index, (first, _, _) in self._texels.items():
            table[index] = first
        return table

    def set_tangents(self, vertex: Optional[np.ndarray] = None, params: Optional[np.ndarray] = None):
        """Forward-mode direction: object-space vertex tangents (V, 3) and parameter tangents (P,)."""
        vertex = np.zeros((self.vertex_count, 3)) if vertex is None else np.asarray(vertex)
        params = np.zeros(self.param_count) if params is None else np.asarray(params)
        if vertex.shape != (self.vertex_count, 3) or params.shape != (self.param_count,):
            raise ValueError("Tangent shapes must be (vertex_count, 3) and (param_count,)")
        self._buffers["vertex_tangent"].copy_from_numpy(np.ascontiguousarray(vertex, np.float32))
        self._buffers["param_tangent"].copy_from_numpy(np.ascontiguousarray(params, np.float32))

    # ------------------------------------------------------------------
    # Rendering

    def forward(self, spp: int = 1, seed: int = 0, terms=TERMS, antithetic=True) -> dict:
        """Primal image and forward-mode derivative images (H, W, 3) of each term and their total.

        ``antithetic``: antithetic sampling. Each pixel-boundary sample comes with its mirror images on the pixel's
        perimeter (four samples; 2 selects two, its reflection through the pixel's center), and a camera path whose
        first vertex samples a glossy reflection is averaged with a path whose sample there is turned by half a
        turn about the lobe (interior term and primal image). False turns it off.
        """
        self._run(False, spp, seed, terms, antithetic)
        images = {"primal": self._image("primal")}
        for term, name in zip(TERMS, ("interior", "pixel_boundary", "primary_boundary", "secondary_boundary")):
            images[term] = self._image(name) if term in terms else np.zeros_like(images["primal"])
        if self._light_tracing is not None and "interior" in terms:
            images["interior"] += self._image("light_interior")
        images["total"] = sum(images[term] for term in TERMS)
        return images

    def backward(self, dl_di: np.ndarray, spp: int = 1, seed: int = 0, terms=TERMS, antithetic=True,
                 accumulate: bool = False):
        """Gradients (vertex (V, 3), params (P,)) of L given dL/dI (H, W, 3); ``antithetic`` as in forward().

        ``accumulate``: the pass's gradients stay on the GPU, added to those of the previous accumulate passes, and
        backward() returns None; gradients() reads the sum (e.g. once per multi-view iteration, instead of reading
        param_count floats back per view)."""
        if self._grads_pending and not accumulate:
            raise RuntimeError("Accumulated gradients unread: call gradients() first")
        dl_di = np.ascontiguousarray(dl_di, np.float32).reshape(-1, 3)
        self._buffers["dl_di"].copy_from_numpy(dl_di)
        self._run(True, spp, seed, terms, antithetic, clear_gradients=not self._grads_pending)
        if accumulate:
            self._grads_pending = True
            return None
        return self._read_gradients()

    def gradients(self):
        """The gradients (vertex (V, 3), params (P,)) summed over the backward(accumulate=True) passes since the last
        call; the next accumulate pass starts from zero."""
        self._grads_pending = False
        return self._read_gradients()

    def _read_gradients(self):
        vertex = self._buffers["vertex_grad"].to_numpy().view(np.float32)[: 3 * self.vertex_count]
        params = self._buffers["param_grad"].to_numpy().view(np.float32)[: self.param_count]
        return vertex.reshape(-1, 3), params  # views of fresh readback arrays (no copy)

    def _run(self, backward: bool, spp: int, seed: int, terms, antithetic, clear_gradients: bool = True):
        if spp <= 0:
            raise ValueError("spp must be positive")
        # Pixel-boundary samples per perimeter position; camera paths have antithetic partners with any of them.
        boundary_samples = (4 if antithetic else 1) if isinstance(antithetic, (bool, np.bool_)) else int(antithetic)
        if boundary_samples not in (1, 2, 4):
            raise ValueError("antithetic must be True, False, 2 or 4")
        paths = boundary_samples > 1
        if "primary" in terms or "secondary" in terms:
            self._update_edges()
        names = ["primal", "interior", "pixel_boundary", "primary_boundary", "secondary_boundary", "light_primal",
                 "light_interior"]
        if backward and clear_gradients:  # only backward passes write them (forward ones keep accumulated gradients)
            names += ["vertex_grad", "param_grad"]
        encoder = self.device.create_command_encoder()
        for name in names:
            encoder.clear_buffer(self._buffers[name])
        self.device.submit_command_buffer(encoder.finish())
        pixels = spy.grid((self.height, self.width))
        light = self._light_tracing is not None and (not backward or "interior" in terms)
        if light:
            self._update_light_emission()
            self._globals.update(light_derivatives="interior" in terms, light_sample_weight=1.0 / spp)
            count = self._light_path_count()
            if count > 1 << 24:
                raise ValueError("At most 2^24 light paths per frame")
        # Frame by frame, as the forward renderer: the frame's light paths (splats into light_primal; their derivatives
        # into light_interior or the gradients), then the camera paths, accumulated into the primal image.
        for i in range(spp):
            frame = (seed * FRAMES_PER_SEED + i) & 0xFFFFFFFF
            if light:
                if not backward:
                    encoder = self.device.create_command_encoder()
                    encoder.clear_buffer(self._buffers["light_primal"])
                    self.device.submit_command_buffer(encoder.finish())
                if count > 0:
                    self._call("render_light", backward, seed, tid=spy.grid((count,)), count=count, frame=frame)
            if not backward:
                self._call("render_primal", False, seed, pixel=pixels, frame=frame, sample_index=i)
        if "interior" in terms:
            for first in range(0, spp, SAMPLES_PER_DISPATCH):
                self._call("render_interior", backward, seed, pixel=pixels, spp=spp, first=first,
                           count=min(SAMPLES_PER_DISPATCH, spp - first), antithetic=paths)
        if "pixel" in terms:
            for first in range(0, spp, SAMPLES_PER_DISPATCH):
                self._call("render_pixel_boundary", backward, seed, pixel=pixels, spp=spp, first=first,
                           count=min(SAMPLES_PER_DISPATCH, spp - first), antithetic=boundary_samples)
        if "primary" in terms and len(self._primary):
            samples = self.width * self.height * spp
            for offset in range(0, samples, 1 << 22):
                count = min(1 << 22, samples - offset)
                self._call("render_primary", backward, seed * 4096 + offset // (1 << 22),
                           tid=spy.grid((count,)), samples=samples)
        if "secondary" in terms and len(self._photon):
            for i in range(spp):
                self._call("sample_photons", False, seed * spp + i, tid=spy.grid((self.photon_count,)))
                self._build_photon_bvh()
                self._call("render_secondary", backward, seed, pixel=pixels, spp_index=i, spp=spp)

    def _call(self, name: str, backward: bool, seed: int, **kwargs):
        def bind(cursor):
            cursor["camera"] = self.camera
            cursor["seed"] = seed
            cursor["light_sampler"] = self._light_sampler
            # The light-tracing state is bound only with light tracing on (older shader modules lack it).
            light = self._light_tracing is not None
            for key, value in self._globals.items():
                if light or key not in LIGHT_TRACING_KEYS:
                    cursor[key] = value
            for key, value in self._buffers.items():
                if light or key not in LIGHT_TRACING_KEYS:
                    cursor[key] = value
            if self._photon_accel is not None:
                cursor["photon_tlas"] = self._photon_accel[2]

        if (name, backward) not in self._kernels:
            self._kernels[name, backward] = self._query(name, backward)
        self._kernels[name, backward].write(bind)(**kwargs)

    def _image(self, name: str) -> np.ndarray:
        return self._buffers[name].to_numpy().view(np.float32).reshape(self.height, self.width, 3).copy()

    def _write_buffer(self, name: str, data: np.ndarray):
        """Writes data to a named buffer, which is reallocated only when it is too small."""
        data = np.ascontiguousarray(data)
        buffer = self._buffers.get(name)
        if buffer is None or buffer.size < max(1, data.nbytes):
            self._buffers[name] = self._buffer(data)
        elif data.nbytes:
            buffer.copy_from_numpy(data.view(np.uint8).reshape(-1))

    def _buffer(self, data: np.ndarray, struct_size: Optional[int] = None):
        data = np.ascontiguousarray(data)
        if data.size == 0:
            data = np.zeros(1, dtype=data.dtype)
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        struct_size = struct_size or data.dtype.itemsize * int(np.prod(data.shape[1:]))
        return self.device.create_buffer(data=data.view(np.uint8).reshape(-1), struct_size=struct_size, usage=usage)

    def _build_photon_bvh(self):
        """Rebuilds the photon BVH (a BLAS of photon spheres in one TLAS instance) in place."""
        if self._photon_accel is None:
            builds = []
            inputs = [spy.AccelerationStructureBuildInputProceduralPrimitives({
                "aabb_buffers": [self._buffers["photon_bounds"]], "aabb_stride": 24,
                "primitive_count": self.photon_count})]
            for kind in (spy.AccelerationStructureKind.bottom_level, spy.AccelerationStructureKind.top_level):
                desc = spy.AccelerationStructureBuildDesc(
                    {"inputs": inputs, "flags": spy.AccelerationStructureBuildFlags.prefer_fast_build})
                sizes = self.device.get_acceleration_structure_sizes(desc)
                accel = self.device.create_acceleration_structure(kind=kind, size=sizes.acceleration_structure_size)
                scratch = self.device.create_buffer(size=sizes.scratch_size, usage=spy.BufferUsage.unordered_access)
                builds.append((desc, accel, scratch))
                if kind == spy.AccelerationStructureKind.bottom_level:
                    instances = self.device.create_acceleration_structure_instance_list(1)
                    instances.write(0, {"transform": spy.float3x4.identity(), "instance_mask": 0xFF,
                                        "acceleration_structure": accel.handle})
                    inputs = [instances.build_input_instances()]
            self._photon_accel = (builds, instances, builds[-1][1])
        encoder = self.device.create_command_encoder()
        for desc, accel, scratch in self._photon_accel[0]:
            encoder.build_acceleration_structure(desc=desc, dst=accel, src=None, scratch_buffer=scratch)
        self.device.submit_command_buffer(encoder.finish())

    # ------------------------------------------------------------------
    # Edges

    def _edge_topology(self, gid: int, triangles: int):
        """Unique vertex indices (ids, and the triangles' indices into them) and edges (see edges.edge_topology())
        of a geometry instance."""
        tri = self._vector_indices(gid, np.arange(triangles, dtype=np.uint32))
        ids, inverse = np.unique(tri, return_inverse=True)
        inverse = inverse.reshape(-1, 3)
        positions = self._vector_positions(*(spy.Tensor.from_numpy(self.device, np.asarray(x, np.uint32))
                                              for x in (np.full(len(ids), gid), ids)))
        return (ids, inverse) + edge_topology(inverse, positions)

    def _build_edges(self):
        origin = np.asarray(self.camera.position, dtype=np.float64)
        gids = [gid for gid, info in self._info.items() if int(info[3]) > 0]
        for gid in gids:
            if gid not in self._topology:
                self._topology[gid] = self._edge_topology(gid, int(self._info[gid][3]))
        # World positions of the vertices of all instances, in one query with cached arguments.
        counts = [len(self._topology[gid][0]) for gid in gids]
        if self._vertex_query is None and gids:
            self._vertex_query = [spy.Tensor.from_numpy(self.device, np.asarray(x, np.uint32)) for x in (
                np.repeat(gids, counts), np.concatenate([self._topology[gid][0] for gid in gids]))]
        positions = np.split(self._vector_positions(*self._vertex_query), np.cumsum(counts)[:-1]) if gids else []
        primary, photon, self._positions = [], [], {}
        for gid, vertices in zip(gids, positions):
            ids, inverse, *edges = self._topology[gid]
            world = vertices[inverse]
            self._positions[gid] = world.reshape(-1, 3)
            silhouettes, creases = classify_edges(gid, world, *edges, origin)
            primary.append(silhouettes)
            photon.append(creases)
        self._scene_extent = float(max((np.ptp(p, axis=0).max() for p in self._positions.values()), default=1.0))
        self._primary, self._primary_length = normalize_cdf(primary)
        self._photon, self._photon_length = normalize_cdf(photon)

    def _vector_indices(self, gid, prims):
        fn = self._query("triangle_indices").return_type(spy.Tensor)
        return fn(gid, spy.Tensor.from_numpy(self.device, prims)).to_numpy().reshape(-1, 3)

    def _vector_positions(self, gids, ids):
        fn = self._query("instance_vertex").return_type(spy.Tensor)
        return fn(gids, ids).to_numpy().reshape(-1, 3)
