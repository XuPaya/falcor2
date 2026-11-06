# PSDR platforms and tests

Status (2026-11-05).

## Supported backends

| Platform | Backend | `traversal="auto"` selects | Status |
|---|---|---|---|
| Windows | D3D12 | `ray_query` (inline ray queries) | supported, tested; two D3D12-only test failures, see below |
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

On D3D12 the material-batch and shading-frame kernels now compile in minutes (see "D3D12 compile time"), but two
D3D12-only failures remain (see "Known failures and limits"); deselect them:

```
set SLANGPY_DEVICE=d3d12
python -m pytest tests/python/render -k "psdr and not split_unsplit and not frame_gradients_match"
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

Windows, D3D12 (RTX 5090), found 2026-11-05 once the kernels compiled; both pass on Windows Vulkan:
- Frame gradients (`backward(frame_gradients=True)`) are wrong on D3D12 for StandardMaterial and OpenPBRMaterial:
  `test_frame_gradients_match_finite_differences_and_forward_mode[*-StandardMaterial|OpenPBRMaterial]` fail
  (Standard: reverse -0.0188 vs forward and finite difference +0.0035; OpenPBR: reverse 0, its gradients land on
  the first instance). SpongeCake is right. A Standard-only scene shows the same error with no material-family split,
  at DXC -Od and -O1, so this predates the compile-time changes; the forward-mode frame derivatives are correct.
  Tangent gradients appear for an isotropic material, as if `bwd_vertex_attribute()` received the wrong `k`.
- `test_mixed_split_unsplit_and_accumulation`: the unsplit interior backward pass (`render_interior`, backward,
  per material family) hangs the GPU (`DXGI_ERROR_DEVICE_HUNG`, TDR). Not yet known whether -Od is involved.

Linux, Vulkan (GB300, driver 615.71.09):
- The same precise-mode parity tests fail at the last-bit level with `trace_ray`: 99.4-99.95 % of pixels are
  identical, and the largest difference is 4.8e-7. Bit-exact parity holds only on D3D12.
- `test_psdr_primal_matches_adjoint_tracer[precise|default]`: the driver cannot compile the reference
  `AdjointPathTracerNode` pipeline (`NVVM compilation failed: 3`). PSDR is not involved.

## D3D12 compile time

DXC (`dxcompiler.dll`) inlines each entry point into one function. The PSDR programs that hold the reverse-mode BSDF
code of all three material families (`backward_vertices`, `render_light_frames`, and the unsplit `render_interior` /
unmerged `render_light` backward) are 450-610 k lines of HLSL once inlined. At DXC `-O1`, which is what Slang's default
optimization level passes, SimplifyCFG (`passingValueIsAlwaysUndefined`, quadratic in block size) runs on them for
hours; DXC v1.9.2609 behaves the same. Standalone `dxc.exe` on the dumped HLSL (Ryzen 9 5950X, seconds):

| kernel (all three families) | `-O1 -Zi` | `-Od -Zi` | `-Od` |
|---|---|---|---|
| `render_primal` (forward, for scale) | 102 | 66 | 48 |
| `backward_vertices` | > 4997, stopped | 1305 | 1158 |
| `render_light_frames` | > 5000, stopped | 2441 | 1661 |

Three changes bring the `trace_ray` programs down to minutes:
- The heavy backward kernels run one program per material family on every backend when all three families are present
  (`PSDR._backward_material_families`; this used to be Vulkan only).
- `backward_material_family()` is `[ForceInline]`, so Slang folds the link-time family test and each family's program
  keeps only that family's reverse-mode code (`backward_vertices`: 0.65 / 0.67 / 1.98 MB of HLSL for Standard /
  SpongeCake / OpenPBR instead of 2.54 MB each). Without it the split programs are as large as the combined one.
- Test devices compile D3D12 shaders at Slang optimization `none` (DXC `-Od`, `falcor2.testing.helpers.get_device`).
  `downstream_args=["-Od"]` does not work: Slang appends its own `-O1` after it and DXC keeps the last level. Even
  split, the OpenPBR programs spend over an hour in DXC at `-O1`. At `-Od` they take 9 (`backward_vertices`) and 24
  (`render_light_frames`) minutes standalone, and PSDR runs about 5 % slower (forward 37.8 -> 39.4 ms, backward
  interior 83.7 -> 87.6 ms on a 128x128 Standard scene at 64 spp; same primal, gradients equal to 1e-6).
  Devices outside the tests (falcor2_inverse's `create_device()`) still use `-O1`.

`ray_query` programs are shared across scenes and always contain all three families (the `HAS_*_MATERIAL`
constants and the family split apply to `trace_ray` only), so on D3D12 their backward kernels still take tens of
minutes each at `-Od`.

Before these changes (measured 2026-11-05 on an RTX 5090, cold shader cache, one test per process):
- No result after 8 hours, DXC still compiling: six of the seven `test_psdr_material_batches` tests (all but
  `test_primary_environment_gradient_is_counted_once[False]`, which passes in 11 minutes) and
  `test_traversals_match`.
- `test_psdr_shading_frames` (frame gradients): no result after 30 minutes with either traversal; one ran 5.5 hours
  without finishing.
- `test_trace_ray_material_backward_matches_finite_difference[OpenPBRMaterial-base_color]` passes, after 86 minutes.

Vulkan builds the same kernels in minutes. These tests were added after PSDR stopped compiling on D3D12 (58b099b and
454babf), so DXC had never compiled them before the link-time traversal type. The other 29 of the 40 D3D12 PSDR tests
pass, including bit-exact precise parity. Both traversals therefore pass on D3D12 separately (the parity tests use
`ray_query`, the `test_trace_ray_*` tests `trace_ray`), but `test_traversals_match` has not completed there.
