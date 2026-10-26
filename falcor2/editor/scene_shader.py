# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import slangpy as spy
import falcor2 as fc


# Light samplers that programs select at run time through SelectableLightSampler (slang/falcor2/render/
# light_sampler.slang): their kind there. Programs for them are linked once for all of them.
SELECTABLE_LIGHT_SAMPLERS = {"UniformLightSampler": 0, "PowerLightSampler": 1}


def light_sampler_prelude(sampler: fc.LightSampler) -> str:
    """Prelude exporting the LightSampler type of a program: SelectableLightSampler for the selectable samplers
    (the same program for each of them), otherwise the sampler's own type and specialization constants."""
    if sampler.slang_type_name in SELECTABLE_LIGHT_SAMPLERS:
        return "export struct LightSampler : ILightSampler = SelectableLightSampler;\n"
    return (sampler.shader_specialization_source
            + f"export struct LightSampler : ILightSampler = {sampler.slang_type_name};\n")


def light_sampler_key(sampler: fc.LightSampler) -> Any:
    """Cache key of the program specialization light_sampler_prelude() makes for a sampler."""
    if sampler.slang_type_name in SELECTABLE_LIGHT_SAMPLERS:
        return "SelectableLightSampler"
    return (sampler.slang_type_name, sampler.shader_generation)


def bind_light_sampler(cursor, sampler: fc.LightSampler):
    """Binds a light sampler to a program's light_sampler (see light_sampler_prelude())."""
    kind = SELECTABLE_LIGHT_SAMPLERS.get(sampler.slang_type_name)
    if kind is None:
        cursor["light_sampler"] = sampler
        return
    block = cursor["light_sampler"]
    block["kind"] = kind
    if sampler.slang_type_name == "PowerLightSampler":
        block["power"] = sampler


class SceneShaderHelper:
    """Load and cache a slang module linked with a scene's render module and requirements."""

    def __init__(self, device: spy.Device):
        self._device = device
        self._utils = spy.Module(device.load_module("falcor2.utils"))
        self._module: Optional[spy.Module] = None
        self._module_key: Optional[Any] = None
        self._requirements_generation: Optional[int] = None
        self._scene = None

    def get_module(self, scene: fc.Scene, module: str | spy.Module) -> spy.Module:
        requirements_generation = scene.requirements_generation
        module_key = module
        if (
            self._module is None
            or module_key != self._module_key
            or requirements_generation != self._requirements_generation
        ):
            reqs = scene.requirements
            render_module = spy.Module(scene.render_module)
            source_module = (
                self._device.load_module(module)
                if isinstance(module, str)
                else module.device_module
            )
            self._module = spy.Module(
                source_module,
                link=[self._utils, render_module] + [spy.Module(m) for m in reqs.modules],
            )
            self._requirements_generation = requirements_generation
            self._module_key = module_key
        self._scene = scene
        return self._module

    def bind_scene(self, cursor):
        self._scene.bind(cursor)
