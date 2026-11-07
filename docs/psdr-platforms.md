# PSDR platforms and tests

Status (2026-11-06).

## Supported backends

| Platform | Backend | `traversal="auto"` selects | Status |
|---|---|---|---|
| Windows | D3D12 | `ray_query` (inline ray queries) | supported; merge validation described in `psdr-dxc-merge.md` |
| Linux | Vulkan | `ray_query` if the GPU has it, else `trace_ray` (e.g. GB300) | supported; previous branch tested, integrated branch awaits Linux validation |
| Windows | Vulkan | `ray_query` | supported; not the automatic device type |

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

The integration keeps default shader optimization, compact D3D12 derivative loops, and the Windows frame-record
fix while retaining Vulkan's unrolled loops and the broader per-family specialization. See
[DXC comparison and validation](psdr-dxc-merge.md) for the compiler probes and current Windows results.
The failure descriptions below are historical observations from the incoming branch before this integration.

## Scene traversal

`PSDR(traversal=...)` chooses how kernels trace scene rays: `ray_query` uses inline ray queries in compute kernels,
and `trace_ray` uses a ray-tracing pipeline, so the tracing kernels run as ray-generation shaders. The
implementation is the link-time type `SceneTraversal` (`slang/falcor2/psdr/traversal.slang`). The renderer's prelude
exports `InlineTraversal` or `PipelineTraversal`, so each program contains only one of them. `TraceRay` must not
reach a compute kernel: Slang rejects it there for HLSL and SPIR-V (E36107, "uses features that are not available in
'compute' stage"). Before the link-time type, a branch on a link-time constant left `TraceRay` in the inline
programs, and PSDR with `ray_query` did not compile on Windows (D3D12 or Vulkan).

## Material-family programs

The reverse-mode BSDF code of the differentiable material families is large. The SPIR-V of `backward_vertices` holds
24 / 146 / 184 KB of function-scope variables with only Standard / SpongeCake / OpenPBR, and 335 KB with all three.
NVIDIA's Vulkan driver on Windows (RTX 5090, 591.86) rejects a program with the OpenPBR and SpongeCake reverse modes
together, with either traversal (`CreatePipeline: failed to validate local memory`); Standard + OpenPBR builds. These
kernels trace no rays, so ray queries are not the cause. DXC (D3D12) compiles all three in one program for hours (see
"D3D12 compile time").

The heavy backward kernels (`MATERIAL_BACKWARD_KERNELS`: `backward_vertices`, `render_interior`, `render_light`,
`render_light_frames`) therefore run one program per material family present in the scene
(`PSDR._backward_material_families`, link-time constant `BACKWARD_MATERIAL_FAMILY`):
- `trace_ray`: when two or more families are present. With one, `HAS_*_MATERIAL` already leaves only its code.
- `ray_query`: always. Its programs are shared across scenes (no `HAS_*_MATERIAL`), so each holds one family; a scene
  runs only its own families' passes.

Materials of other types get a pass of their own (`MATERIAL_OTHER`, no reverse-mode BSDF), as does a `ray_query` scene
without a differentiable family. The first pass also owns the derivatives that do not belong to a surface vertex. The
path-tracing kernels (`render_interior` unsplit, `render_light`, `render_light_frames`) trace the paths again in each
pass, so a scene with k families runs k backward passes of them; `backward_vertices` passes only re-read the recorded
vertices. Until 2026-11-06 the split applied to `trace_ray` with all three families only, so `ray_query` backward
failed on Windows Vulkan (`test_traversals_match[vulkan]`), and so did `trace_ray` with OpenPBR + SpongeCake.

## Historical failures and remaining limits

Windows, Vulkan (RTX 5090, driver 591.86): no known failures.
- Precise-mode parity with `ray_query` is not bit-exact. The parity tests (`test_psdr_primal_matches_path_tracer[*-precise]`,
  `..._deep`) need rtol 8.7e-7 at max_depth 6 and 2.4e-6 at max_depth 16 and 32 (largest absolute difference 6.7e-6,
  on bright pixels); their tolerance (rtol 1e-5, atol 1e-8) allows that. `trace_ray` differs by 6e-8.

Before the integration, Windows D3D12 (RTX 5090), found 2026-11-05 and 2026-11-06 once the kernels compiled; the first two pass on Windows
Vulkan:
- Frame gradients (`backward(frame_gradients=True)`) are wrong on D3D12 for StandardMaterial and OpenPBRMaterial:
  `test_frame_gradients_match_finite_differences_and_forward_mode[*-StandardMaterial|OpenPBRMaterial]` fail
  (Standard: reverse -0.0188 vs forward and finite difference +0.0035; OpenPBR: reverse 0, its gradients land on
  the first instance). SpongeCake is right. A Standard-only scene shows the same error with no material-family split,
  at DXC -Od and -O1, so this predates the compile-time changes; the forward-mode frame derivatives are correct.
  Tangent gradients appear for an isotropic material, as if `bwd_vertex_attribute()` received the wrong `k`.
- `test_mixed_split_unsplit_and_accumulation`: the unsplit interior backward pass (`render_interior`, backward,
  per material family) hangs the GPU (`DXGI_ERROR_DEVICE_HUNG`, TDR). Not yet known whether -Od is involved.
- `test_traversals_match[d3d12]` fails when both persistent test caches (module and shader) are warm: creating the
  `ray_query` `backward_vertices` pipeline returns `SLANG_FAIL` with no message (slang-rhi reports any failed
  `NvAPI_D3D12_CreateComputePipelineState` that way). It passes without caches, from a cold cache and with either
  cache alone (2026-11-06: 3 failures, 4 passes). A reduced scene that compiles both traversals' interior backward
  passes with both caches warm passes, so the trigger is not known yet. One run of the test fills most of the
  128 MB RHI cache; the warm run logs `MDB_MAP_FULL`. This was the pre-integration workaround; the integrated validation checks cache reuse explicitly.

Linux, Vulkan (GB300, driver 615.71.09): no known failures. Precise-mode parity with `trace_ray` is not bit-exact
(99.4-99.95 % of pixels are identical, the largest difference is 4.8e-7), which the parity tests' tolerance allows.
Bit-exact parity holds only on D3D12.

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

The incoming branch used three changes to bring the backward programs down to minutes:
- The heavy backward kernels run one program per material family on every backend (see "Material-family programs";
  this used to be Vulkan only).
- `backward_material_family()` is `[ForceInline]`, so Slang folds the link-time family test and each family's program
  keeps only that family's reverse-mode code (`backward_vertices`: 0.65 / 0.67 / 1.98 MB of HLSL for Standard /
  SpongeCake / OpenPBR instead of 2.54 MB each). Without it the split programs are as large as the combined one.
- The incoming branch compiled test devices at Slang optimization `none` (DXC `-Od`, `falcor2.testing.helpers.get_device`).
  `downstream_args=["-Od"]` does not work: Slang appends its own `-O1` after it and DXC keeps the last level. Even
  split, the OpenPBR programs spend over an hour in DXC at `-O1`. At `-Od` they take 9 (`backward_vertices`) and 24
  (`render_light_frames`) minutes standalone, and PSDR runs about 5 % slower (forward 37.8 -> 39.4 ms, backward
  interior 83.7 -> 87.6 ms on a 128x128 Standard scene at 64 spp; same primal, gradients equal to 1e-6).
  Devices outside the tests (falcor2_inverse's `create_device()`) still used `-O1`.

The integrated branch keeps the first two changes but replaces the global `-Od` workaround with the Windows
branch's bounded two-direction loops and disabled debug metadata for heavy PSDR tests. Other tests retain their
existing debug settings. Normal optimization stays enabled for every backend. The original Windows and integrated
OpenPBR backward-vertex probes emit identical HLSL and DXIL; the incoming unrolled probe produces a 37% larger DXIL
blob with the same compiler options. See [the measured comparison](psdr-dxc-merge.md).

Until 2026-11-06 the `ray_query` programs (shared across scenes) held all three families, so on D3D12 their backward
kernels took tens of minutes each at `-Od`: 18 minutes for the backward interior term of a Standard-only scene. Split
per family, the Standard program takes about 1.5 minutes, and `test_traversals_match[d3d12]` completes in 10-18
minutes from a cold cache.

Before these changes (measured 2026-11-05 on an RTX 5090, cold shader cache, one test per process):
- No result after 8 hours, DXC still compiling: six of the seven `test_psdr_material_batches` tests (all but
  `test_primary_environment_gradient_is_counted_once[False]`, which passes in 11 minutes) and
  `test_traversals_match`.
- `test_psdr_shading_frames` (frame gradients): no result after 30 minutes with either traversal; one ran 5.5 hours
  without finishing.
- `test_trace_ray_material_backward_matches_finite_difference[OpenPBRMaterial-base_color]` passes, after 86 minutes.

Vulkan builds the same kernels in minutes. These tests were added after PSDR stopped compiling on D3D12 (58b099b and
454babf), so DXC had never compiled them before the link-time traversal type. Before integration, 29 of the 40 D3D12
PSDR tests passed, including precise parity, and `test_traversals_match` first completed from a cold cache with the
`ray_query` family split. The integrated tests now cover both traversals explicitly. Current Windows measurements,
cache checks, numerical tolerances, and validation results are in [the integration report](psdr-dxc-merge.md).
