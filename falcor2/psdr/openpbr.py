# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The differentiable copy of the OpenPBR headers that slang/falcor2/psdr/openpbr.slang includes."""

import re
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[2]


def generate_openpbr_headers():
    """Write the differentiable OpenPBR copy included by slang/falcor2/psdr/openpbr.slang.

    The headers of external/openpbr-bsdf are copied to slang/falcor2/psdr/generated/ unchanged, except that every
    function definition not declared through an OpenPBR function macro (openpbr.slang defines those as
    [Differentiable]) is prefixed with [Differentiable]. Only changed files are written, since shader hot reload
    recompiles the loaded modules when files change.
    """
    source = _REPOSITORY / "external" / "openpbr-bsdf"
    output = _REPOSITORY / "slang" / "falcor2" / "psdr" / "generated" / "openpbr-bsdf"
    definition = re.compile(r"^(?!OPENPBR_)([A-Za-z_]\w*\s+openpbr_\w+\s*\()", re.M)
    for path in sorted(source.rglob("*.h")):
        target = output / path.relative_to(source)
        text = definition.sub(r"[Differentiable]\n\1", path.read_text(encoding="utf-8"))
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
