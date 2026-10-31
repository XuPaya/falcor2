"""Diagnose the low AD/FD ratio of the self-test's silhouette scale gradient.

Builds silhouette scenes like MeshScene.silhouette_scene (chosen shapes emit white, others black occluders, no
lights) from the self-test scene, poses chosen shapes with an explicit pivot / scale / translation, and compares
the AD gradient, split into PSDR terms (interior, primary, pixel), with same-seed central FD.
"""
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
ap.add_argument("--spp", type=int, default=128)
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
ALL = list(s.shapes)
EYE = np.asarray(s.p["camera"], np.float64)
CENTRES = {"s_coated": (-0.9, 0.3, 0.0), "s_velvet": (0.9, 0.3, 0.3)}
print("shapes:", ALL, flush=True)


def sil_scene(white, black):
    ms = fi.Scene(recompute_normals=True)
    wm = ms.add_material("white", base_color_factor=(0, 0, 0), emissive_factor=(1, 1, 1), roughness_factor=1.0, ior=1.0)
    bm = ms.add_material("black", base_color_factor=(0, 0, 0), roughness_factor=1.0, ior=1.0)
    for n in white + black:
        pos, fc = s.scene.mesh_arrays(n)
        ms.add_mesh(n, pos, fc, wm if n in white else bm)
    ms.add_constant_light("none", (0.0, 0.0, 0.0))
    RW, RH, fov_y = s._cam_geo[:3]
    ms.add_camera("camera", s.p["camera"], s.p["target"], up=s.p.get("up", [0, 1, 0]), fov_y=fov_y, resolution=(RW, RH))
    ms.update()
    base = {n: torch.as_tensor(np.asarray(ms.mesh_arrays(n)[0], np.float32), device="cuda") for n in white}
    r = fi.Renderer(ms, spp=a.spp, max_bounces=1, terms=("interior", "primary", "pixel"), ray_epsilon=a.ray_eps or None)
    return ms, r, base


def experiment(name, white, black, kind, comp, pivot, weight, h):
    """kind: 'scale' (uniform about pivot) or 'translate' (component comp); weight: 'one' | 'hramp' | 'vramp'."""
    t0 = time.time()
    _, r, base = sil_scene(white, black)
    pivot = torch.as_tensor(np.asarray(pivot, np.float32), device="cuda")
    H, W = s.height, s.width
    w = {"one": 1.0,
         "hramp": torch.linspace(0.2, 1.0, W, device="cuda")[None, :, None],
         "vramp": torch.linspace(0.2, 1.0, H, device="cuda")[:, None, None]}[weight]

    def vals(x):
        if kind == "scale":
            return {f"{n}.vertex_positions": (b - pivot) * x + pivot for n, b in base.items()}
        t = torch.zeros(3, device="cuda")
        t = t.index_put((torch.tensor([comp], device="cuda"),), x.reshape(1))
        return {f"{n}.vertex_positions": b + t for n, b in base.items()}

    def loss(x, seed):
        return (s.crop(r(vals(x), seed=seed)) * w).mean()

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
            lp = float(loss(torch.tensor(x0 + h, device="cuda"), 900 + rep))
            lm = float(loss(torch.tensor(x0 - h, device="cuda"), 900 + rep))
        out["fd"].append((lp - lm) / (2 * h))
    res = {}
    for k, v in out.items():
        v = np.asarray(v, np.float64)
        res[k] = [float(v.mean()), float(v.std() / np.sqrt(len(v)))]
    fd, fdse = res["fd"]
    tot, totse = res["total"]
    res["ratio"] = tot / fd if fd else None
    res["gap_sigma"] = abs(tot - fd) / float(np.hypot(totse, fdse))
    res["terms_sum"] = res["interior"][0] + res["primary"][0] + res["pixel"][0]
    res["seconds"] = round(time.time() - t0, 1)
    print(f"{name:28s} FD {fd:+.5f}±{fdse:.5f}  AD {tot:+.5f}±{totse:.5f}  ratio {res['ratio'] if res['ratio'] is None else round(res['ratio'], 3)}"
          f"  gap {res['gap_sigma']:.1f}σ | interior {res['interior'][0]:+.5f} primary {res['primary'][0]:+.5f}"
          f" pixel {res['pixel'][0]:+.5f}  ({res['seconds']} s)", flush=True)
    return res


pair = ["s_coated", "s_velvet"]
others = [n for n in ALL if n not in pair]
lo = np.min([np.asarray(s.scene.mesh_arrays(n)[0]).min(0) for n in pair], 0)
hi = np.max([np.asarray(s.scene.mesh_arrays(n)[0]).max(0) for n in pair], 0)
PAIR_PIVOT = [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]          # MeshScene's default pivot
V = CENTRES["s_velvet"]
EXPS = {
    "base_pair_scale":        (pair, others, "scale", 0, PAIR_PIVOT, "one", 0.02),
    "pair_scale_no_occluders": (pair, [], "scale", 0, PAIR_PIVOT, "one", 0.02),
    "velvet_scale_centre":    (["s_velvet"], [], "scale", 0, V, "one", 0.02),
    "coated_scale_centre":    (["s_coated"], [], "scale", 0, CENTRES["s_coated"], "one", 0.02),
    "velvet_scale_eye":       (["s_velvet"], [], "scale", 0, EYE, "one", 0.02),
    "velvet_tz":              (["s_velvet"], [], "translate", 2, V, "one", 0.01),
    "velvet_ty_vramp":        (["s_velvet"], [], "translate", 1, V, "vramp", 0.01),
    "velvet_tx_hramp":        (["s_velvet"], [], "translate", 0, V, "hramp", 0.01),
}
results = {"width": a.width, "spp": a.spp, "reps": a.reps, "pair_pivot": [float(x) for x in PAIR_PIVOT]}
for name, args in EXPS.items():
    if a.only and name not in a.only.split(","):
        continue
    results[name] = experiment(name, *args)
    (O / "scale_diag.json").write_text(json.dumps(results, indent=2))
print("DONE ->", O / "scale_diag.json", flush=True)
