"""Occlusion-edge diagnostic: a white emitter sphere W partly hidden by a black sphere B (the self-test's coated and
gold spheres, same camera), each moved in turn.  AD per PSDR term vs same-seed central FD of mean(silhouette)."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

F2 = Path("/localhome/local-lifanw/projects/agentic_psdr/scene-toolkit-release/workspace/local/experiments/scene_pipeline_f2")
ap = argparse.ArgumentParser()
ap.add_argument("--output", required=True)
ap.add_argument("--width", type=int, default=160)
ap.add_argument("--spp", type=int, default=64)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--only", default="")
ap.add_argument("--ray-eps", type=float, default=0.0, help="PSDR ray_epsilon (0: PSDR's default)")
a = ap.parse_args()
O = Path(a.output)
O.mkdir(parents=True, exist_ok=True)
SC = O / "scene"
if not SC.exists():
    shutil.copytree(F2 / "selftest_scene", SC)

from runtime import load  # noqa: E402
import falcor2_inverse as fi  # noqa: E402

TRUE = json.loads((SC / "parameters_true.json").read_text())
(SC / "_true.json").write_text(json.dumps(TRUE))
_, s, _ = load(SC / "manifest.json", a.width, str(SC / "_true.json"))
WC, BC = (-0.9, 0.3, 0.0), (-0.3, 0.3, 0.3)


def scene(spheres):
    ms = fi.Scene(recompute_normals=True)
    wm = ms.add_material("white", base_color_factor=(0, 0, 0), emissive_factor=(1, 1, 1), roughness_factor=1.0, ior=1.0)
    bm = ms.add_material("black", base_color_factor=(0, 0, 0), roughness_factor=1.0, ior=1.0)
    for name, c, sub, white in spheres:
        v, f = fi.sphere(c, 0.3, sub)
        ms.add_mesh(name, v, f.astype(np.int64), wm if white else bm)
    ms.add_constant_light("none", (0.0, 0.0, 0.0))
    RW, RH, fov_y = s._cam_geo[:3]
    ms.add_camera("camera", s.p["camera"], s.p["target"], up=s.p.get("up", [0, 1, 0]), fov_y=fov_y, resolution=(RW, RH))
    ms.update()
    return ms, fi.Renderer(ms, spp=a.spp, max_bounces=1, terms=("interior", "primary", "pixel"),
                           ray_epsilon=a.ray_eps or None)


def experiment(name, spheres, moving, kind, comp, pivot, h):
    t0 = time.time()
    ms, r = scene(spheres)
    base = {n: torch.as_tensor(np.asarray(ms.mesh_arrays(n)[0], np.float32), device="cuda") for n in moving}
    pivot = torch.as_tensor(np.asarray(pivot, np.float32), device="cuda")

    def vals(x):
        if kind == "scale":
            return {f"{n}.vertex_positions": (b - pivot) * x + pivot for n, b in base.items()}
        t = torch.zeros(3, device="cuda").index_put((torch.tensor([comp], device="cuda"),), x.reshape(1))
        return {f"{n}.vertex_positions": b + t for n, b in base.items()}

    loss = lambda x, seed: s.crop(r(vals(x), seed=seed)).mean()
    x0 = 1.0 if kind == "scale" else 0.0
    out = {"interior": [], "primary": [], "pixel": [], "total": [], "fd": []}
    for rep in range(a.reps):
        for term in ("interior", "primary", "pixel", "total"):
            r.terms = ("interior", "primary", "pixel") if term == "total" else (term,)
            x = torch.tensor(x0, device="cuda", requires_grad=True)
            loss(x, 101 + rep).backward()
            out[term].append(float(x.grad))
        r.terms = ("interior", "primary", "pixel")
        with torch.no_grad():
            lp, lm = (float(loss(torch.tensor(x0 + sg * h, device="cuda"), 900 + rep)) for sg in (1, -1))
        out["fd"].append((lp - lm) / (2 * h))
    res = {k: [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))] for k, v in out.items()}
    (fd, fdse), (tot, totse) = res["fd"], res["total"]
    res["ratio"] = tot / fd if abs(fd) > 1e-9 else None
    res["gap"] = tot - fd
    res["gap_sigma"] = abs(tot - fd) / max(float(np.hypot(totse, fdse)), 1e-12)
    res["seconds"] = round(time.time() - t0, 1)
    print(f"{name:22s} FD {fd:+.5f}±{fdse:.5f}  AD {tot:+.5f}±{totse:.5f}  gap {tot - fd:+.5f} ({res['gap_sigma']:.1f}σ)"
          f" | int {res['interior'][0]:+.5f} prim {res['primary'][0]:+.5f}±{res['primary'][1]:.5f}"
          f" pix {res['pixel'][0]:+.5f}  ({res['seconds']} s)", flush=True)
    return res


W4, B4 = ("W", WC, 4, True), ("B", BC, 4, False)
W6, B6 = ("W", WC, 6, True), ("B", BC, 6, False)
EXPS = {
    "W_tx_alone":   ([W4], ["W"], "translate", 0, WC, 0.01),
    "W_tx_behind_B": ([W4, B4], ["W"], "translate", 0, WC, 0.01),
    "B_tx_over_W":  ([W4, B4], ["B"], "translate", 0, BC, 0.01),
    "B_ty_over_W":  ([W4, B4], ["B"], "translate", 1, BC, 0.01),
    "B6_tx_over_W": ([W4, B6], ["B"], "translate", 0, BC, 0.01),
    "W_tx_behind_B6": ([W4, B6], ["W"], "translate", 0, WC, 0.01),
    "W6_tx_behind_B": ([W6, B4], ["W"], "translate", 0, WC, 0.01),
    "W_scale_alone": ([W4], ["W"], "scale", 0, WC, 0.02),
    "W_scale_behind_B": ([W4, B4], ["W"], "scale", 0, WC, 0.02),
}
results = {"width": a.width, "spp": a.spp, "reps": a.reps}
for name, args in EXPS.items():
    if a.only and name not in a.only.split(","):
        continue
    results[name] = experiment(name, *args)
    (O / "occ_diag.json").write_text(json.dumps(results, indent=2))
print("DONE ->", O / "occ_diag.json", flush=True)
