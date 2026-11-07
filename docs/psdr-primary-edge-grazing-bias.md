# PSDR primary-edge term drops samples at grazing occluder edges

Status (2026-10-31): fixed in `slang/falcor2/psdr/primary.slang`. The edge rays now start just off the
occluder. With the default `ray_epsilon`, every occlusion case is within 1.6σ of FD (see [Fix](#fix)). The
self-test's `ray_epsilon=1e-3` workaround is removed.

## Summary

Before the fix, PSDR's primary-edge (occlusion-boundary) term `render_primary` (`slang/falcor2/psdr/primary.slang`)
underestimated the derivative wherever one object hides part of another. It samples a point `xb` on the occluder's
silhouette edge and sent two rays from it:

- `visible(camera.position, xb)` (`geometry.slang`): `tmin = ray_epsilon`, `tmax = |xb - camera| - ray_epsilon`
- `trace_path(xb, d, ...)` for the surface behind (`paths.slang`): `tmin = ray_epsilon`

Both rays lie in a plane that only touches the occluder along the edge, so they meet the edge's two neighbouring
faces at grazing angles between 0 and the dihedral angle (about edge length / radius). `xb = lerp(p0, p1, t)` is
float32. When rounding puts it a distance δ inside the occluder, a ray meets a neighbouring face about δ / sin φ
from `xb`. When that exceeds `ray_epsilon`, the sample is lost. Either `visible()` fails, or the ray "behind" the
edge sees the occluder's own face (black in a silhouette scene, so `value == 0` and an early return). The loss is
silent and always in the same direction: the edge term comes out too small.

PSDR's default `ray_epsilon` is `max(1e-5, 1e-6 * scene extent)`. The dropped fraction grows as the occluder's
mesh gets finer (smaller dihedral angles) and shrinks as `ray_epsilon` grows. The hidden surface's tessellation
doesn't matter.

Untested: which of the two rays drops most samples. The shadow-edge term (`secondary.slang`) is photon-based. It
gathers edge photons along path segments against photon bounding boxes and casts no ray from an edge point against
the occluder's own faces, so this failure mode doesn't carry over directly. That term hasn't been tested either.

## How it showed up

The scene-toolkit backend self-test (`scene_pipeline_f2/selftest.py`, check `pose`) compares AD with same-seed
central FD of the silhouette loss `mean(silhouette)` of the pair `s_coated` + `s_velvet`. Every other self-test
sphere is a black occluder. `s_gold` hides part of `s_coated`.

| run | `silhouette scale_pair` AD / FD | gap |
|---|---|---|
| spp 64 (default) | 0.0514 / 0.0576 = **0.891** | 6.4σ |
| spp 256 | 0.0513 / 0.0577 = **0.888** | 9.2σ |
| spp 64, `ray_epsilon=1e-3` | 0.0562 / 0.0576 = **0.975** | 1.3σ |
| spp 64, default `ray_epsilon`, after the fix | 0.0562 / 0.0576 = **0.976** | 1.3σ |

The gap didn't depend on the sample count. The check still passed because its tolerance is
`3σ + 0.15 |FD|`. The x-translation checks moved from 0.98 to 1.00 with the larger offset.

## Evidence

All numbers are AD vs FD at width 160, spp 64 and 8 repeats, with the self-test camera. "σ" is the gap over the
combined standard error.

**Unoccluded motion is correct** (`scale_diag.py`). For a lone sphere, the primary term is exactly 0. A sphere's own
silhouette is handled by the interior and pixel terms in material form.

| case | ratio | gap |
|---|---|---|
| `s_velvet` scale about its centre | 1.003 | 0.6σ |
| `s_coated` scale about its centre | 0.995 | 1.3σ |
| `s_velvet` scale about the camera (image invariant) | 0 = 0 | — |
| `s_velvet` translate z / y (vertical ramp) / x (horizontal ramp) | 0.995 / 0.999 / 1.004 | ≤ 0.5σ |
| pair scale, occluders removed | 0.964 | 0.5σ |
| pair scale, occluders present (the self-test) | 0.913 | 5.0σ |

**The occlusion edge is short, and more so for finer occluders** (`occ_diag.py`: white emitter sphere W at the
`s_coated` position, black sphere B at the `s_gold` position, both r = 0.3, icosphere subdivision as given):

| case | FD | AD | gap |
|---|---|---|---|
| B (sub 4) moves in x over W | +0.0743 | +0.0672 | **−9.6%**, 19σ |
| B (sub 4) moves in y over W | −0.0232 | −0.0221 | −4.9%, 4.5σ |
| **B (sub 6)** moves in x over W | +0.0745 | +0.0542 | **−27%**, 52σ |
| W moves in x behind B (sub 4) | +0.0164 | +0.0235 | +0.0071, 6.6σ (edge term −10% in magnitude) |
| W moves in x behind B (sub 6) | +0.0163 | +0.0352 | +0.0189, 15.6σ |
| W (sub 6) moves behind B (sub 4) | +0.0165 | +0.0236 | +0.0071, 6.0σ (same as W sub 4) |
| W alone moves in x | +0.0804 | +0.0803 | 0.0σ |

**A larger `ray_epsilon` closes the gap**, B moving in x over W:

| `ray_epsilon` | B sub 4 | B sub 6 |
|---|---|---|
| 1e-5 (default) | −9.6% (19σ) | −27% (52σ) |
| 1e-4 | −0.3% (0.5σ) | −3.1% (5.6σ) |
| 1e-3 | +0.4% (0.7σ) | −0.8% (1.5σ) |

These rule out other explanations:

- **Sample count:** the gap is the same at spp 64 and 256.
- **FD step:** projected area is quadratic in scale, so a central difference is exact.
- **`vertex_grad_clamp`:** its default is 0 (off).
- **Pivot placement:** scaling about each sphere's own centre is correct.

## Impact

This applied to every falcor2 build before the fix. Results from fits run with those builds carry it.

- **Pose fitting (mask loss).** `fit_manifest_pose`'s mask loss uses `MeshScene.silhouette_scene`. An object that
  another object partly hides gets too little gradient from the hidden part of its outline, and the shortfall
  grows with the occluder's mesh density. A 0.3 m sphere at subdivision 6 (about 5 mm edges) already loses 27%.
  Dense, smoothly curved meshes, such as SAM 3D reconstructions, chairs, lamps or plants, likely lose more.
  Sharp-edged occluders like boxes and walls (large dihedral angles) are barely affected.
- **RGB fits.** Every occlusion edge in an RGB render goes through this term, so geometry and pose derivatives
  from RGB losses are biased the same way. The `pose_rgb` self-test check already fails for other reasons
  (heavy-tailed interior-term vertex derivatives).
- **Adam-based fits.** These absorb a constant gradient scale, but this loss isn't constant. It depends on
  which edges are occlusion edges and how finely the occluder is meshed, so joint fits under-weight those parts
  of the silhouette.

## Workaround (removed)

Until the fix, the self-test's `check_pose` built its silhouette scene with `ray_epsilon=1e-3`. That override is
gone. The pass-throughs it used remain, for experiments:

- `falcor2-inverse/falcor2_inverse/render.py`: `Renderer(..., ray_epsilon=None)` passes the value through to
  `PSDR` (None means PSDR's default).
- `scene_pipeline_f2/f2_mesh_scene.py`: `MeshScene.silhouette_scene(names, spp, ray_epsilon=None)` passes it to
  its renderer.

A large `ray_epsilon` is not a general fix. In multi-bounce renders it skips surfaces within that distance of a ray
origin, which causes leaks and wrong visibility at contacts.

## Fix

Applied 2026-10-31. In `render_primary`, both rays are cast from a point moved off the occluder, perpendicular to the
edge plane, instead of from `xb`:

```diff
     float3 xb = lerp(p[0], p[1], t);
     float3 d = normalize(xb - camera.position);
     float z = dot(d, camera_forward());
-    if (z <= 0 || !visible(camera.position, xb))
+    if (z <= 0)
+        return;
+    // Edge-plane normal, pointing away from the edge's faces (for a silhouette edge both lie on the other side).
+    float3 nb = normalize(cross(p[0] - p[1], d));
+    nb *= -sign(dot(nb, p[2] - p[0]));
+    // The rays below run along the edge's own faces at grazing angles. Float32 rounding can put xb just inside the
+    // occluder, and a hit on those faces then drops the sample (see docs/psdr-primary-edge-grazing-bias.md), so
+    // cast them from a point just off the occluder.
+    float3 xo = xb + nb * max(ray_epsilon, 1e-5 * max(length(xb), length(camera.position)));
+    if (!visible(camera.position, xo))
         return;
     float2 film = camera.project_world_to_pixel(xb);
     ...
     PathRecord rec;
-    trace_path(xb, d, NO_SLOT, sg, rec);
+    trace_path(xo, d, NO_SLOT, sg, rec);
     float3 value = rec.radiance;
     if (all(value == 0))
         return;
     Background b = background(rec, xb, d);
-    float3 nb = normalize(cross(p[0] - p[1], d));
-    nb *= -sign(dot(nb, p[2] - p[0]));
     float weight = sensor_importance(b.p) * boundary_factor(b.p, xb, normalize(p[0] - p[1]), b.n)
```

Why this works:

- For a silhouette edge, both neighbouring faces lie on the `-nb` side of the edge plane. A ray at distance η on
  the `+nb` side can't reach them, whatever the grazing angle.
- The hit/miss margin in the triangle test is at least η, so η only needs to beat float32 rounding (about
  1e-7 × coordinate size). `ray_epsilon` (1e-5 at unit scale) is about 100 times that, and the relative term
  covers scenes far from the origin.
- The pixel index, `boundary_factor` and the velocity still use `xb`. `boundary_velocity` recomputes the hit on
  the background triangle from the exact edge point, so the velocity isn't shifted. Only `b.p` (importance and
  Jacobian) moves, by O(η).

An alternative is to ignore hits on the edge's own two triangles (`e.prim`, `e.prim1` of `e.gid`) in both rays.
That needs candidate filtering in `visible()` and in `trace_path()`'s first segment, so it's more invasive.

**Result** (`occ_diag.py`, default `ray_epsilon`, same seeds as the tables above):

| case | FD | AD before | AD after | gap after |
|---|---|---|---|---|
| B (sub 4) moves in x over W | +0.0743 | +0.0672 (−9.6%) | +0.0746 | +0.4%, 0.7σ |
| B (sub 4) moves in y over W | −0.0232 | −0.0221 (−4.9%) | −0.0236 | +1.8%, 1.6σ |
| **B (sub 6)** moves in x over W | +0.0745 | +0.0542 (**−27%**) | +0.0742 | −0.4%, 0.7σ |
| W moves in x behind B (sub 4) | +0.0164 | +0.0235 | +0.0166 | 0.3σ |
| W moves in x behind B (sub 6) | +0.0163 | +0.0352 | +0.0171 | 0.7σ |
| W (sub 6) moves behind B (sub 4) | +0.0165 | +0.0236 | +0.0169 | 0.3σ |
| W alone moves in x | +0.0804 | +0.0803 | +0.0803 | 0.0σ |
| W scales alone / behind B (sub 4) | +0.1178 / +0.0969 | — | +0.1172 / +0.0964 | 1.3σ / 1.1σ |

The gap no longer grows with the occluder's mesh density, and it is as small as in the `ray_epsilon=1e-3` runs
(+0.4% and −0.8%).

The self-test changed with the fix (`selftest.py`):

- `check_pose` no longer overrides `ray_epsilon`.
- New pose check `silhouette occluder edge (subdivision 6) x`: the B sub 6 case above.
- `verdict` tolerance tightened from `0.15 |FD|` to `0.05 |FD|`. The original scale result (gap 0.00628 vs
  0.00581 allowed) would have failed it.

Self-test `compile,appearance,pose` after the fix (run `primary-edge-fix`, spp 64): all checks pass under the
tighter tolerance. The pose checks:

| check | AD / FD | gap |
|---|---|---|
| silhouette translation_pair[0] at 0 / at y 0.03 | 1.002 / 0.996 | 0.2σ / 0.3σ |
| silhouette scale_pair | 0.976 | 1.3σ |
| silhouette occluder edge (subdivision 6) x | 0.996 | 0.7σ |

Two appearance checks pass only because their noise term is large: `coat_weight_coated` (ratio 1.39, 1.3σ) and
`light_emission_0` (0.94, 1.2σ). These are the ones the tighter tolerance could fail on an unlucky seed.

Not yet tested after the fix: the shadow-edge term, `pose_rgb`, and real scenes (multi-bounce RGB fits).

## Reproduction

Scripts are in `docs/psdr-primary-edge-grazing-bias/` and run with the falcor2 environment:

```bash
R=/localhome/local-lifanw/projects/agentic_psdr
export PYTHONPATH=$R/falcor2-inverse:$R/scene-toolkit-release/workspace/local/experiments/scene_pipeline_f2:$R/scene-toolkit-release/workspace/local/experiments/scene_pipeline
$R/falcor2/.venv/bin/python $R/falcor2/docs/psdr-primary-edge-grazing-bias/occ_diag.py --output /tmp/occ --only B_tx_over_W,B6_tx_over_W
$R/falcor2/.venv/bin/python $R/falcor2/docs/psdr-primary-edge-grazing-bias/occ_diag.py --output /tmp/occ_eps --ray-eps 1e-3 --only B_tx_over_W,B6_tx_over_W
$R/falcor2/.venv/bin/python $R/falcor2/docs/psdr-primary-edge-grazing-bias/scale_diag.py --output /tmp/scale --spp 64
```

- `occ_diag.py`: W/B occlusion cases (table above).
- `scale_diag.py`: the self-test's own silhouette scene with chosen shapes, pivots and motions.

Both print FD, AD and the AD split into the interior, primary and pixel terms, and write JSON to `--output`.
Seeds are fixed, so the numbers above reproduce exactly. Each case takes about 10 s.
