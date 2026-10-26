# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The differentiable copy of the OpenPBR headers that slang/falcor2/psdr/openpbr.slang includes."""

import re
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[2]

# Value-preserving edits of the differentiable copy that cut its compiled size: every call site of a function is
# inlined (with its derivative code) into each kernel, so the coating lobe evaluates its base lobe at one call site
# instead of one in each branch; and PSDR only evaluates the BSDF (its sampling and pdf use the render module's build),
# so the aggregate lobe's sampling weights (read only by its pdf and lobe selection) are not estimated.
PATCHES = {
    "impl/openpbr_coating_lobe.h": [(
        """{
    if (lobe.inside)
    {
        const float odotn = dot(light_direction, lobe.normal_ff);
        const vec3 base_layer_scale = openpbr_coat_passage_color_multiplier(lobe, -odotn);  // negative to tint only transmission

        return openpbr_scale_diffuse_specular(openpbr_calculate_lobe_value(lobe.base_lobe, view_direction, light_direction), base_layer_scale);
    }
    else
    {
        return openpbr_combine_coat_and_base_evals(lobe,
                                                   openpbr_calculate_lobe_value(lobe.coat_reflection_lobe, view_direction, light_direction),
                                                   openpbr_calculate_lobe_value(lobe.base_lobe, view_direction, light_direction),
                                                   light_direction);
    }
}""",
        """{
    const OpenPBR_DiffuseSpecular base_value = openpbr_calculate_lobe_value(lobe.base_lobe, view_direction, light_direction);
    if (lobe.inside)
    {
        const float odotn = dot(light_direction, lobe.normal_ff);
        const vec3 base_layer_scale = openpbr_coat_passage_color_multiplier(lobe, -odotn);  // negative to tint only transmission

        return openpbr_scale_diffuse_specular(base_value, base_layer_scale);
    }
    else
    {
        return openpbr_combine_coat_and_base_evals(lobe,
                                                   openpbr_calculate_lobe_value(lobe.coat_reflection_lobe, view_direction, light_direction),
                                                   base_value,
                                                   light_direction);
    }
}""",
    )],
    "impl/openpbr_aggregate_lobe.h": [(
        """{
    lobe.lobe_weights[OpenPBR_SpecularLobeIndex] = openpbr_estimate_lobe_contribution(lobe.specular_lobe, view_direction, path_throughput);
    if (OPENPBR_GET_SPECIALIZATION_CONSTANT(EnableTranslucency))
        lobe.lobe_weights[OpenPBR_DielectricMMSLobeIndex] =
            openpbr_estimate_lobe_contribution(lobe.dielectric_mms_lobe, view_direction, path_throughput);
    if (OPENPBR_GET_SPECIALIZATION_CONSTANT(EnableMetallic))
        lobe.lobe_weights[OpenPBR_MetalMMSLobeIndex] = openpbr_estimate_lobe_contribution(lobe.metal_mms_lobe, view_direction, path_throughput);
    lobe.lobe_weights[OpenPBR_DiffuseLobeIndex] = openpbr_estimate_lobe_contribution(lobe.diffuse_lobe, view_direction, path_throughput);
    lobe.lobe_weights[OpenPBR_ThinWallSpecularTransLobeIndex] =
        openpbr_estimate_lobe_contribution(lobe.thin_wall_specular_trans_lobe, view_direction, path_throughput);
    lobe.lobe_weights[OpenPBR_ThinWallDiffuseTransLobeIndex] =
        openpbr_estimate_lobe_contribution(lobe.thin_wall_diffuse_trans_lobe, view_direction, path_throughput);
}""",
        """{
    for (int i = 0; i < OpenPBR_NumBaseLobes; ++i)
        lobe.lobe_weights[i] = 0.0f;  // (PSDR copy: sampling weights unused, see PATCHES in falcor2/psdr/openpbr.py)
}""",
    )],
}


def generate_openpbr_headers():
    """Write the differentiable OpenPBR copy included by slang/falcor2/psdr/openpbr.slang.

    The headers of external/openpbr-bsdf are copied to slang/falcor2/psdr/generated/ unchanged, except that every
    function definition not declared through an OpenPBR function macro (openpbr.slang defines those as
    [Differentiable]) is prefixed with [Differentiable], and the PATCHES below are applied. Only changed files are
    written, since shader hot reload recompiles the loaded modules when files change.
    """
    source = _REPOSITORY / "external" / "openpbr-bsdf"
    output = _REPOSITORY / "slang" / "falcor2" / "psdr" / "generated" / "openpbr-bsdf"
    definition = re.compile(r"^(?!OPENPBR_)([A-Za-z_]\w*\s+openpbr_\w+\s*\()", re.M)
    for path in sorted(source.rglob("*.h")):
        target = output / path.relative_to(source)
        text = definition.sub(r"[Differentiable]\n\1", path.read_text(encoding="utf-8"))
        for old, new in PATCHES.get(path.relative_to(source).as_posix(), ()):
            if old not in text:
                raise RuntimeError(f"OpenPBR patch does not apply to {path}: {old[:60]!r}")
            text = text.replace(old, new)
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
