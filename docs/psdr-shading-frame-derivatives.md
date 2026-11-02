# PSDR vertex gradients ignored the shading normals of smooth meshes

Status (2026-11-02): fixed on falcor2 / falcor2-inverse branch `lifanw/silk_zoom_out`. PSDR can now return the
gradients of the stored vertex normals and tangents (`backward(frame_gradients=True)`, `frame_gradients()`), and
falcor2_inverse chains them through the normals and tangents it recomputes after every vertex write. Vertex-position
and instance rotation / scale gradients of smooth-shaded meshes now agree with finite differences (FD).

## Summary

The interior term differentiates each path vertex in material form (`vertex_terms()`, `paths.slang`): positions,
the geometric normal, the area Jacobian and the incident / outgoing directions all follow the vertex positions.
The shading frame did not. `make_surface()` built it `no_diff` from `SurfaceInteraction.shading_frame_ws`, the
barycentric interpolation of the *stored* vertex normals and tangents, and `shading_frame()` returned it detached
unless the material had a normal map. So PSDR's vertex gradient was the derivative of the image with the stored
normals held fixed.

That is not what a vertex update renders. falcor2_inverse's `VertexPositions.write()` recomputes area-weighted
normals and tangents from the new positions (`recompute_normals=True`, the default and the scene toolkit's setting).
On a smooth mesh, shading is dominated by the normal: a vertex bump tilts the normals around it by
O(displacement / spacing), far more than it moves the light and view directions. The missing term is the main one,
so AD and FD disagreed in size and sign.

`MeshTransform` rotation / scale had the same gap. A rotated instance has rotated world normals
(`M^-T n`), but the gradient was chained from vertex positions only.

## How it showed up

The silk cloth reconstruction (`silk_zoom_out_Nov_1_opus_5.5_extra`, task `r4-fold-shading`) tried to fit the drape's
folds to the highlights with an RGB vertex loss. The folds did not change, and an AD vs FD check of the RGB loss for
6 cm Gaussian normal bumps of the cloth (view 0195) was off by 3-20x, with a sign flip at the dome.

## Evidence

A smooth icosphere (subdivision 4, r = 0.5, glossy StandardMaterial, roughness 0.3) under a small sphere light, a
weighted image loss, and Gaussian normal bumps (σ = 0.12) at three places: the lit side, the centre and the
terminator side. AD (interior term) vs same-seed central FD, before the fix:

| bump | AD / FD, normals recomputed on write | AD / FD, normals frozen |
|---|---|---|
| light side | **-1.08** | 0.96 |
| centre | **1.36** | 0.99 |
| dark side | **0.16** | 1.03 |

AD matched FD only when the stored normals were frozen during FD: the derivative was exact for a different function
from the one being rendered. After the fix (16 repeats; frozen-normal control in the second column, unchanged
code path):

| bump | AD / FD, recomputed (fixed) | control, frozen |
|---|---|---|
| light side | 1.061 (3.0σ) | 0.979 (3.2σ) |
| centre | 0.963 (0.4σ) | 1.003 (0.3σ) |
| dark side | 1.005 (2.4σ) | 1.027 (10σ) |

Other configurations after the fix (forward mode through `set_tangents(normals=..., frame_tangents=...)`, or all
PSDR terms where silhouettes move):

| case | light side | centre | dark side | before (constant frames) |
|---|---|---|---|---|
| anisotropic OpenPBR (aniso 0.7), forward mode | 1.031 | 1.053 (0.5σ) | 1.004 | -0.65 / 1.73 / 0.14 |
| diffuse, rotated + non-uniformly scaled instance, all terms | 1.011 (0.8σ) | 0.976 (0.7σ) | 1.045* | -0.90 / 2.54 / 0.57 |

\* The frozen-normal control gives 1.079 for that bump, so the remaining gap is not from the shading frame.

falcor2_inverse `MeshTransform` on a glossy ellipsoid instance (all terms):

| parameter | AD / FD, fixed | before |
|---|---|---|
| rotation z | 0.932 (3σ) | **-2.40** |
| rotation x | 0.952 (1.2σ) | **-5.01** |
| scale x | 1.000 (0.1σ) | 0.853 |

**The cloth scene itself** (the session's `adcheck.py`: view 0195, glossy proxy, 3 bounces, clipped RGB loss in the
cloth region, 6 cm Gaussian normal bumps, FD ±4 mm), AD with and without the frame term on the same samples:

| bump | FD | AD / FD, fixed | AD / FD, before |
|---|---|---|---|
| mid-left, light tracing on (8 / 4 repeats) | -3.65e-3 ± 0.33e-3 | 0.99 (0.1σ) | 0.44 (4.8σ) |
| mid-left, light tracing off (16 / 8) | -3.19e-3 ± 0.37e-3 | 1.10 (0.7σ) | 0.43 (4.0σ) |
| dome, light tracing on | +2.3e-4 ± 0.7e-4 | -1.42 (3.3σ) | **-7.2** (11.8σ) |
| dome, light tracing off | +1.6e-4 ± 0.9e-4 | -1.14 (2.4σ) | **-11.7** (15σ) |
| front fold, light tracing off | +1.31e-3 ± 0.45e-3 | 1.31 (0.4σ) | 0.78 (0.3σ) |

The dome derivative is nearly zero (FD within 2σ of 0), so its ratio is noise-dominated; the toolkit's self-test
criterion would call it inconclusive. The front fold's AD is heavy-tailed (±1e-3, the glossy-lobe issue that
`vertex_grad_clamp` addresses). The session's original 3-20x mismatch came from the missing frame term.

**Scene-toolkit self-test** (`scene_pipeline_f2/selftest.py`): `appearance` and `pose` pass unchanged. `pose_rgb`
still fails as "inconclusive (noise)": unclamped, the sharp glossy coats make AD heavy-tailed (e.g. -13 ± 7 vs FD
0.004). With `vertex_grad_clamp` 10, rotation_pair (1.24) and translation_gold (1.01) pass, and the pair's
translation / scale stay inconclusive because even FD is within noise of 0. The results are identical with and
without the frame term, apart from rotation_pair (1.16 without). Sphere normals do not change under translation or
uniform scale, and barely under rotation, so this check cannot see the bug.

The new test `tests/python/render/test_psdr_shading_frames.py` perturbs the stored normals and tangents of smooth
patches of each material family directly (no falcor2_inverse): reverse vs forward mode within 3e-3, reverse vs FD
1.005 (StandardMaterial), 1.033 (anisotropic OpenPBRMaterial), 1.067 (SpongeCakeMaterial).

## Fix

**falcor2 (PSDR).** The stored per-vertex normal and tangent are differentiable inputs:

- `vertex_attribute()` (`geometry.slang`) reads a stored normal or tangent; its custom derivatives read
  `frame_tangent` (forward) and accumulate into `frame_grad` (backward, clamped like vertex gradients, with the
  bound scaled by the scene extent).
- `interpolated_frame()` rebuilds the shading frame as `make_triangle_surface_interaction()` does (barycentric
  interpolation, inverse-transpose / world transforms, Gram-Schmidt with the primal handedness).
- `vertex_shading_frame()` returns the primal frame unchanged (primal values stay bit-identical) with custom
  derivatives: `interpolated_frame()`'s in forward mode; in backward mode, the frame adjoint is kept in a
  thread-local and `backward_vertices()` appends one `FrameItem` per vertex. A new kernel, `backward_frames()`,
  chains the items to `frame_grad` after each `backward_vertices()` chunk.
- `shading_frame()` (`materials.slang`) starts from `vertex_shading_frame()` when the surface's `vertex_frame` is set:
  camera-path vertices with `frame_derivatives` on. Light-path vertices keep constant frames, consistent with "no
  geometric derivatives along light paths". With normal maps, `normal_map_derivatives=False` now detaches only the
  normal map's perturbation.

Why a separate kernel: with all three material families on Vulkan (`traversal="trace_ray"`), PSDR compiles one
reverse-mode vertex kernel per family. Any variant that compiled `interpolated_frame()`'s reverse code into those
kernels made the SpongeCake kernel hang (Xid 109, `CTX SWITCH TIMEOUT`, `VK_ERROR_DEVICE_LOST`), even with the
switch off at run time so that code never executed. Variants tried: inline, hand-unrolled, `[noinline]`
(then the OpenPBR kernel hung too), plain atomics instead of the wave-reduced `atomic_add`, no new buffer binding,
and the chain at the kernel's top level. Only moving it out of the vertex kernels works. SpongeCake-only scenes
did not hang.

**Python API (`falcor2/psdr/renderer.py`).**

- `backward(..., frame_gradients=False)`: with True, the pass also accumulates the stored-frame gradients;
  `frame_gradients()` returns `(normals (V, 3), tangents (V, 3))` after `backward()` / `gradients()`, else None.
  Needs `split_interior` (the default).
- `set_tangents(vertex, params, normals=None, frame_tangents=None)`: forward-mode tangents of the stored frames.
- The vertex gradients still hold the stored frames fixed. Whoever recomputes normals from positions chains the
  frame gradients through that computation: PSDR cannot know whether, or how, normals are recomputed.
- When off, the custom derivatives return at once and no frame items are written; the vertex gradients of a
  three-family test scene were bit-identical to the old code's. When on: one small kernel per backward chunk and a
  frame-item buffer of one 60-byte item per vertex item (about 57 MB, allocated on first use).

**falcor2_inverse.** `VertexPositions` (with `recompute_normals`) and `MeshTransform` (rotation, scale) have
`turns_frames`; the renderer asks PSDR for frame gradients when a wanted parameter has it.
`VertexPositions.grad` adds `frame_vertex_grad()` (`geometry.py`), the torch vector-Jacobian product of
`vertex_normals()` / `tangents()`. `MeshTransform.grad` adds the frames' rotation with the instance:
`M0^T M^-T n` and `M0^-1 M t`.

## Impact

Every falcor2 build before the fix. Shading (RGB) gradients with respect to geometry missed the frame term for
every mesh whose normals are recomputed on vertex writes. This is worst on smooth-shaded meshes (wrong in size and
often in sign), and applies to flat-shaded ones too, whose stored face normals turn with the face. Affected: vertex
fits with an RGB term, pose fits with `rgb_weight > 0` (the toolkit poses by writing vertex positions), and
falcor2_inverse `MeshTransform` rotation / scale. Silhouette (mask) losses and material, texture and light
gradients are unaffected.

## Not covered

- Light paths (light tracing) have no frame derivatives, as they have no geometric derivatives at all.
- `adjust_shading_normal()`'s grazing-angle blend (cos < 0.1) still uses the detached geometric normal and primal
  incident direction.
- Forward mode through falcor2_inverse: it has none; PSDR callers pass the frame tangents themselves.
- The scene toolkit restores UV-aligned tangents (tiled / rotated-anisotropy shapes) only in `update()`; during
  vertex fits the stored tangents are falcor2_inverse's generic ones, which the chain matches.

## Files

- falcor2: `slang/falcor2/psdr/{geometry,globals,interior,materials,paths,psdr}.slang`, `falcor2/psdr/renderer.py`,
  `tests/python/render/test_psdr_shading_frames.py`
- falcor2-inverse: `falcor2_inverse/{geometry,params,render}.py`
