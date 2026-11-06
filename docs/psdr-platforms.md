# PSDR platforms and tests

Status (2026-11-04).

## Supported backends

| Platform | Backend | `traversal="auto"` selects | Status |
|---|---|---|---|
| Windows | D3D12 | `ray_query` (inline ray queries) | supported, tested |
| Linux | Vulkan | `ray_query` if the GPU has it, else `trace_ray` (e.g. GB300) | supported, tested |
| Windows | Vulkan | `ray_query` | **not supported**: known failures, see below |

SlangPy's automatic device type (falcor2_inverse's `create_device()`, the scene toolkit) already picks D3D12 on
Windows and Vulkan on Linux. Tests default to every backend of the platform (`falcor2.testing.helpers`), so select
the supported one explicitly:

```
# Windows
set SLANGPY_DEVICE=d3d12
python -m pytest tests/python/render -k psdr
# Linux
SLANGPY_DEVICE=vulkan python -m pytest tests/python/render -k psdr
```

## Scene traversal

`PSDR(traversal=...)` chooses how kernels trace scene rays: `ray_query` uses inline ray queries in compute kernels,
and `trace_ray` uses a ray-tracing pipeline, so the tracing kernels run as ray-generation shaders. The
implementation is the link-time type `SceneTraversal` (`slang/falcor2/psdr/traversal.slang`). The renderer's prelude
exports `InlineTraversal` or `PipelineTraversal`, so each program contains only one of them. `TraceRay` must not
reach a compute kernel: Slang rejects it there for HLSL and SPIR-V (E36107, "uses features that are not available in
'compute' stage"). Before the link-time type, a branch on a link-time constant left `TraceRay` in the inline
programs, and PSDR with `ray_query` did not compile on Windows (D3D12 or Vulkan).

## Known failures and limits

Windows, Vulkan (RTX 5090), not supported:
- With `ray_query`, NVIDIA's driver rejects the `backward_vertices` pipeline that the interior term uses:
  `CreatePipeline: failed to validate local memory`. This happened at 223d570 too, before the traversal option
  existed. The forward pass and the pixel, primary and secondary terms work. `test_traversals_match[vulkan]` fails.
- The precise-mode parity tests (`test_psdr_primal_matches_path_tracer[*-precise]`, `..._deep`) assert bit-exact
  equality with the reference path tracer. On Vulkan PSDR differs in the last bits: up to 4.8e-6 with `ray_query`
  and 6e-8 with `trace_ray`, on values around 0.4. The default-precision variants, which use a tolerance, pass.

Linux, Vulkan (GB300, driver 615.71.09):
- The same precise-mode parity tests fail at the last-bit level with `trace_ray`: 99.4-99.95 % of pixels are
  identical, and the largest difference is 4.8e-7. Bit-exact parity holds only on D3D12.

Windows, D3D12, compile time: DXC compiles the largest kernels slowly. On a cold shader cache, a single
`test_psdr_material_batches` test can take over 50 minutes, whichever traversal is used. Vulkan builds the same
kernels in minutes.
