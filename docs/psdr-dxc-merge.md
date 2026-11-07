# DXC compilation fixes: Windows/Vulkan integration

The integration combines `39fb0f7` (`lifanw/merge_all_fixes` before the merge) and
`06b5cf1` (`lifanw/dxc_compile_fix`). The former fixes D3D12 compilation and frame-gradient
destinations; the latter adds Vulkan/TraceRay portability and broader material-family dispatch.

## What differs

| Change | Windows branch | Vulkan branch | Combined implementation |
|---|---|---|---|
| Reverse-mode material specialization | D3D12 `backward_vertices` | All four heavy backward kernels, both backends | Keep the broader specialization |
| Family predicate | Force-inlined before HLSL emission | Force-inlined before HLSL emission | Keep it force-inlined |
| Two-direction OpenPBR/SpongeCake evaluation | Bounded loop, one derivative call site | Force-unrolled to fix Vulkan reverse mode | Bounded loop on D3D12; force-unrolled on Vulkan |
| D3D12 test optimization | Default optimization | Globally disabled (`-Od`) | Default optimization |
| Shader debug information | Disabled for the heavy frame tests | Standard D3D12 debug information | Disabled for heavy PSDR derivative tests; other tests retain their defaults |
| Frame-gradient destination | Original vertex hit, outside AD context | Hit recovered through custom derivative context | Original vertex hit |
| Scene updates | Refresh material-family dispatch | Families captured at construction | Refresh dispatch; TraceRay still requires reconstruction when a new material type is introduced |

The four heavy kernels are `backward_vertices`, `render_interior`, `render_light`, and
`render_light_frames`. Family selection filters the differentiated surface vertex, while
material decoding, emission, and gradient destinations remain scene-wide. The first family
pass owns shared camera-miss and light-emission derivatives, so those contributions occur once.
The recorded-path implementation records each chunk once, processes each family, and then
chains the accumulated shading-frame adjoints once.

The shader-test dispatcher also keeps its original uniform-scalar selector on D3D12.
The incoming constant-buffer selector caused a reproducible CBV/root-signature mismatch
in `test_interior_list.slang`, including with both persistent caches disabled. All 13
interior-list tests passed after restoring the D3D12 selector. Other backends retain the
constant buffer, which avoids CUDA's invalid constant-memory `__ldg()` emission.

## Recompute versus checkpointing

There are no explicit `[PreferRecompute]` or `[PreferCheckpoint]` attributes in
`slang/falcor2/psdr`. Some shared ray-origin helpers use `[PreferRecompute]`.
These are attributes on differentiable callees; adding one to a backward dispatch entry point
is not a global policy switch for its call graph.

The installed compiler is Slang 2026.16.1. Its
[checkpoint policy](https://github.com/shader-slang/slang/blob/v2026.16.1/source/slang/slang-ir-autodiff-primal-hoist.cpp#L2752)
honors explicit preferences, recomputes trivial arithmetic and calls proven to have no side
effects or nonlocal loads, and retains other required values. The default is therefore not
equivalent to applying `[PreferCheckpoint]` to every function.

An offline OpenPBR `backward_vertices` experiment added `[PreferRecompute]` to the three
`OPENPBR_*_FUNCTION` macros in the compact-loop variant. The resulting HLSL was byte-identical:
1,206,750 bytes, 28,124 lines, SHA-256
`0022e7d50a092ea5f661181cd4c18be4ea9363fcb63c4a288c4df59386cc194e`.
This experiment covers those OpenPBR functions, not every possible annotation placement.
It provides no evidence that blanket annotations improve this kernel, so they are not added.
Recomputation can reduce saved state but adds execution work and can enlarge generated code;
compile time and GPU memory must be measured separately.

## Measurement method

Measurements use the installed Slang 2026.16.1 and bundled DXC on Windows with an RTX A6000.
The offline comparison compiles the same OpenPBR-family `backward_vertices` wrapper, type
conformances, precise floating-point mode, and shader model 6.6 against snapshots of each
branch. It creates DXIL without GPU dispatch. Logs include commands, emitted HLSL, elapsed
times, and exit codes. These are individual measurements, not hardware-independent guarantees.

| OpenPBR backward-vertex variant | DXC options | DXC seconds | DXIL bytes |
|---|---|---:|---:|
| Original Windows branch | `-O1`, no debug information | 95.86 | 548,168 |
| Incoming Vulkan branch, same compiler options | `-O1`, no debug information | 144.69 | 752,688 |
| Incoming branch with only the two bounded loops restored | `-O1`, no debug information | 119.14 | 548,280 |
| Incoming branch's D3D12 test configuration | `-Od`, standard debug information | 367.61 | 74,084,916 |
| Combined implementation | `-O1`, no debug information | 131.89 | 548,168 |

These probes ran alongside correctness tests, so differences include host-load variation.
The original Windows and combined probes emit byte-identical HLSL and DXIL; their timing
difference is measurement variation, not a compiler-code difference. The merged DXIL SHA-256 is
`7fcc4c1a81f0f3f6ad01983eba6504d91cc1a2b8b7a26107c476ad8499525093`.
The same-options comparison and loop experiment support retaining the compact D3D12 loops;
debug metadata and the global optimization setting are additional differences between the
original branches. Full application cold-start time includes multiple shader compilations.
For the incoming debug-enabled probe, Slang `-O0` and its default optimization emitted
byte-identical HLSL (1,617,340 bytes); the `-Od` row therefore also covers the branch's
Slang optimization setting for this probe.
DXIL blob size includes metadata and is not a measurement of GPU working memory.

A follow-up same-options comparison ran the Windows and incoming DXC commands sequentially
in Windows/incoming/incoming/Windows order. Windows took 127.04 and 106.12 seconds (mean
116.58); incoming took 183.31 and 192.92 seconds (mean 188.11). The compact Windows kernel
took approximately 38% less DXC time in these repeats. Both variants reproduced their
original DXIL hashes. Other correctness jobs were still active, so this small sample is a
machine-specific comparison, not a throughput guarantee. The merged kernel is byte-identical
to the Windows kernel in this probe.

The original Windows branch's four shading-frame tests passed with an empty module/shader
cache: 409.08 seconds in pytest, 415.86 seconds including process startup and shutdown.
Compilation telemetry separates Slang, DXC, and driver pipeline creation.

Validation artifacts are recorded under `C:/agentic_psdr/logs/merge-fixes-20261105`.

## Windows validation

Validation uses an RTX A6000 (48 GiB), NVIDIA driver 610.47, CPython 3.12.15,
SlangPy 0.43.0, and Slang 2026.16.1. The Python runs include slow and image tests
(`--all`). The focused PSDR and throughput runs use separate persistent cache
directories for D3D12 and Vulkan; the initial broad runs used the repository's
shared test-cache root. Linux was
not exercised during this integration. The CUDA-only coverage rasterizer and
explicitly CUDA-only OptiX/Torch cases are outside this backend matrix.

The native executable linked with the updated `scene_picker.cpp` passed 603 tests
and 230,572 assertions on each backend; two CUDA-only cases were excluded.

The expanded PSDR matrix has 61 cases per backend: material-family splitting,
mixed-material derivatives, material reassignment, camera and adjoint parity,
deep paths, unique photon gathering, frame gradients for both traversals, all
backward terms, opacity, and direct traversal comparisons. The complete Vulkan
matrix passed in 2,391.33 seconds of pytest time. Its final traversal cache check
passed in two fresh processes (9.56 and 8.84 seconds wall time); the latter
reported 56/56 shader cache hits and 28/28 pipeline cache hits.

Parity tests now exercise both `ray_query` and `trace_ray`. Matched precise
TraceRay paths retain `rtol=1e-5, atol=1e-8`, as do both D3D12 paths. On Vulkan,
ray queries versus the reference TraceRay renderer can change an intersection
enough to flip a later sampling decision. That comparison uses the existing
stochastic bounds: fewer than 1% of pixels exceeding 1e-4 relative error and
mean absolute error below 1% of the different-seed noise estimate. This is
paired with strict matched-traversal coverage, rather than relaxing every
precise comparison.

The diagnostic Vulkan media comparison differed at 3/4,096 pixels, with mean
absolute error 2.68e-5 (6.59e-5 of the different-seed noise). The original strict
deep-path failures were rounding-level differences, with maximum absolute error
4.77e-6. Forcing matched TraceRay traversal passed all three original strict
assertions without changing their tolerances.

The full matrix exceeds the native shader cache's 128 MiB default. Python test
devices now allow 2 GiB to reduce eviction between fresh-process runs. The
focused traversal checks verify reuse on both backends. Device validation,
normal shader optimization, and numerical gradient
tolerances remain enabled.

The D3D12 matrix was validated in two partitions: 42 completed cases from the
expanded run and the earlier successful 19-case frame/traversal run. Before
stopping duplicate compilation, the renderer, PSDR shaders, frame tests, and
traversal tests were checked against the earlier run's Git tree and were
unchanged. `verified-prefix.json` preserves the actual 42 pytest pass records;
that interrupted process is not reported as a successful full-suite process.
The subsequent D3D12 traversal check passed twice in fresh processes: 705.42
seconds while filling missing cache entries, then 13.33 seconds with 56/56
shader cache hits and 23/28 pipeline cache hits.

The normal Windows Release build of `falcor2_tests` and `falcor2_ext` also passed.

The broad Python runs plus focused rechecks give:

| Backend | Passed | Failed | Skipped | Expected failures |
|---|---:|---:|---:|---:|
| D3D12 | 1,341 | 2 | 19 | 2 |
| Vulkan | 1,339 | 0 | 23 | 2 |

The D3D12 Python-material scalar evaluation fails after earlier material tests
with `DXGI_ERROR_DEVICE_HUNG`; running it alone passes. A fresh archive of the
starting branch (`39fb0f7`) reproduced the same failure in the unmodified material
test file: 33 passed, one failed. This failure predates the merge. The baseline
uses the shared native build; its only integrated native change is the scene
picker selection, which this test does not use. No speculative material-shader
workaround is included.

Adjoint frames now use one command submission for the path tracer, accumulation
clear, light tracer, and resolve. A caller-provided command encoder remains owned
by the caller. The six adjoint correctness tests, including two new encoder
ownership cases, and both adjoint parity cases passed on each backend.

The original serial throughput checks measured 48.6 FPS on D3D12 and 104.6 FPS on
Vulkan at 640x360. A D3D12 control/batched/batched/control experiment measured
51.6 / 60.7 / 65.6 / 58.1 FPS. After integrating batching, the primary validation
run measured 59.6 FPS on D3D12 and 122.4 FPS on Vulkan. The unchanged 60 FPS
threshold still failed on D3D12, so the table retains that failure. Later
repeats passed at 70.2 and 69.8 FPS, making this an intermittent performance
failure rather than a reliably green check. The pristine starting branch also
failed the same test at 49.8 FPS on this machine, so the threshold failure
predates the merge. Writing the settings struct in one operation did not
improve the surrounding controls and was not integrated.
