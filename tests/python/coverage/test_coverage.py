# SPDX-License-Identifier: Apache-2.0
"""Tests of falcor2.coverage (deterministic differentiable coverage rasterizer): primal vs a supersampled ray-cast
reference, gradients vs central finite differences, determinism and timing.

Run: python -m pytest --noconftest tests/python/coverage/test_coverage.py -s   (needs a CUDA GPU; --noconftest skips
the repository conftest, whose falcor2 test devices are not needed here)
"""

import math
import time

import numpy as np
import pytest
import torch

pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from falcor2.coverage import render  # noqa: E402
from falcor2.coverage.rasterizer import camera_frame  # noqa: E402

CAM = dict(eye=(0.3, 0.4, 3.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov_y=40.0)
DEV = torch.device("cuda")


# ----------------------------------------------------------------------------------------------------------------------
# Reference: S x S regular samples per pixel, 3D ray casting (Moller-Trumbore), nearest hit.

def reference(V, F, C, cam, W, H, S=64, chunk=1 << 19):
    V = torch.as_tensor(V, dtype=torch.float64, device=DEV)
    F = torch.as_tensor(F, dtype=torch.int64, device=DEV)
    C = torch.as_tensor(C, dtype=torch.float64, device=DEV)
    eye, rot, f, cx, cy = camera_frame(cam, W, H)
    eye, rot = eye.to(DEV), rot.to(DEV)
    o = (torch.arange(S, dtype=torch.float64, device=DEV) + 0.5) / S
    px = (torch.arange(W, dtype=torch.float64, device=DEV)[:, None] + o[None]).reshape(-1)
    py = (torch.arange(H, dtype=torch.float64, device=DEV)[:, None] + o[None]).reshape(-1)
    yy, xx = torch.meshgrid(py, px, indexing="ij")
    d_cam = torch.stack([(xx - cx) / f, (yy - cy) / f, torch.ones_like(xx)], -1).reshape(-1, 3)
    dirs = d_cam @ rot  # world directions (unnormalised; t is then view depth)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = b - a, c - a
    out = torch.zeros(dirs.shape[0], C.shape[1], dtype=torch.float64, device=DEV)
    for s in range(0, dirs.shape[0], chunk):
        d = dirs[s:s + chunk]
        best_t = torch.full((d.shape[0],), float("inf"), dtype=torch.float64, device=DEV)
        best_f = torch.full((d.shape[0],), -1, dtype=torch.int64, device=DEV)
        for k in range(0, F.shape[0], 64):
            sl = slice(k, k + 64)
            pvec = torch.linalg.cross(d[:, None], e2[None, sl].expand(d.shape[0], -1, -1))
            det = (e1[None, sl] * pvec).sum(-1)
            inv = 1 / torch.where(det == 0, torch.ones_like(det), det)
            tvec = eye[None, None] - a[None, sl]
            u = (tvec * pvec).sum(-1) * inv
            qvec = torch.linalg.cross(tvec.expand(d.shape[0], -1, -1), e1[None, sl].expand(d.shape[0], -1, -1))
            v = (d[:, None] * qvec).sum(-1) * inv
            t = (e2[None, sl] * qvec).sum(-1) * inv
            hit = (det != 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            t = torch.where(hit, t, torch.full_like(t, float("inf")))
            tm, im = t.min(1)
            better = tm < best_t
            best_t = torch.where(better, tm, best_t)
            best_f = torch.where(better, im + k, best_f)
        col = torch.where((best_f >= 0)[:, None], C[best_f.clamp(min=0)], torch.zeros(1, dtype=torch.float64,
                                                                                      device=DEV))
        out[s:s + chunk] = col
    return out.reshape(H, S, W, S, -1).mean((1, 3)).float()


# ----------------------------------------------------------------------------------------------------------------------
# Scenes

def tri_scene():
    V = np.array([[-0.5, -0.4, 0.0], [0.6, -0.3, 0.1], [0.1, 0.5, -0.1]])
    return V, np.array([[0, 1, 2]]), np.array([[1.0, 0.5, 0.25]])


def two_tri_scene():
    V = np.array([[-0.6, -0.4, 0.0], [0.6, -0.5, 0.05], [0.0, 0.6, -0.05],
                  [-0.3, 0.3, 0.5], [0.7, 0.2, 0.55], [0.2, -0.6, 0.45]])
    F = np.array([[0, 1, 2], [3, 5, 4]])  # opposite windings
    return V, F, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def box_mesh(sx=0.5, sy=0.4, sz=0.3, rot=(0.3, 0.5, 0.1), center=(0.0, 0.0, 0.0)):
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], np.float64)
    q = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    f = np.array([[a, b, c] for a, b, c, d in q] + [[a, c, d] for a, b, c, d in q])
    fq = np.array(list(range(6)) + list(range(6)))  # quad of each triangle
    rx, ry, rz = rot
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return v @ (Rz @ Ry @ Rx).T + np.asarray(center), f, fq


def blob_mesh(n_lat=10, n_lon=16, seed=0, center=(0.0, 0.0, 0.0), r=0.45):
    """Jittered UV sphere (closed, irregular)."""
    rng = np.random.default_rng(seed)
    verts = [[0, r, 0]]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * (j + 0.3 * (i % 2)) / n_lon
            verts.append([r * math.sin(th) * math.cos(ph), r * math.cos(th), r * math.sin(th) * math.sin(ph)])
    verts.append([0, -r, 0])
    v = np.array(verts) * (1 + 0.08 * rng.standard_normal((len(verts), 1))) * np.array([1.2, 0.9, 1.0])
    f = []
    for j in range(n_lon):
        f.append([0, 1 + (j + 1) % n_lon, 1 + j])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + (j + 1) % n_lon
            c, d = a + n_lon, b + n_lon
            f += [[a, b, d], [a, d, c]]
    last = len(verts) - 1
    for j in range(n_lon):
        a, b = 1 + (n_lat - 2) * n_lon + j, 1 + (n_lat - 2) * n_lon + (j + 1) % n_lon
        f.append([last, a, b])
    return v + np.asarray(center), np.array(f)


def random_scene(seed, n=6):
    """Random triangles in disjoint depth slabs (no interpenetration), random colours, some black."""
    rng = np.random.default_rng(seed)
    V, F, C = [], [], []
    for k in range(n):
        z = 0.6 - 0.25 * k
        c = rng.uniform(-0.6, 0.6, 2)
        p = np.c_[c + rng.uniform(-0.6, 0.6, (3, 2)), z + rng.uniform(-0.08, 0.08, 3)]
        V.append(p)
        F.append([3 * k, 3 * k + 1, 3 * k + 2])
        C.append(rng.uniform(0, 1, 3) if k % 3 else np.zeros(3))
    return np.concatenate(V), np.array(F), np.array(C)


def normal_colors(V, F):
    """Per-face flat normal colours 0.5 + 0.5 n, binned (like the fit's normals view)."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                     [1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]], np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    b = np.argmax(n @ dirs.T, 1)
    return 0.5 + 0.5 * dirs[b]


def scenes():
    out = {"triangle": tri_scene(), "two_triangles": two_tri_scene()}
    v, f, fq = box_mesh()
    out["box_flat"] = (v, f, np.tile([[0.2, 0.7, 0.9]], (len(f), 1)))
    rng = np.random.default_rng(3)
    qc = rng.uniform(0.1, 1, (6, 3))
    out["box_face_colours"] = (v, f, qc[fq])
    bv, bf = blob_mesh()
    occ_v, occ_f, _ = box_mesh(1.2, 1.0, 0.05, rot=(0.1, 0.2, 0.0), center=(0.1, 0.0, -0.8))
    out["blob_normals_occluder"] = (np.concatenate([bv, occ_v]), np.concatenate([bf, occ_f + len(bv)]),
                                    np.concatenate([normal_colors(bv, bf), np.zeros((len(occ_f), 3))]))
    for s in range(3):
        out[f"random{s}"] = random_scene(s)
    # a floor reaching behind the camera (near-plane and guard-band clipping) under a box, both coloured
    fv = np.array([[-20, -1.0, -20], [20, -1.0, -20], [20, -1.0, 20], [-20, -1.0, 20]], np.float64)
    out["box_on_floor"] = (np.concatenate([v, fv]), np.concatenate([f, np.array([[0, 1, 2], [0, 2, 3]]) + len(v)]),
                           np.concatenate([qc[fq], np.tile([[0.1, 0.3, 0.9]], (2, 1))]))
    return out


def _t(x, grad=False):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=DEV, requires_grad=grad)


# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(scenes()))
def test_primal_vs_supersampled(name):
    V, F, C = scenes()[name]
    W = H = 24
    cam = dict(CAM, fov_y=45.0)
    img = render(_t(V), torch.as_tensor(F), _t(C), cam, W, H).detach()
    ref = reference(V, F, C, cam, W, H, S=64)
    err = (img - ref).abs().max().item()
    print(f"\n[primal] {name}: max |I - I_ref(64x64)| = {err:.2e}, sum I = {img.sum().item():.4f} "
          f"ref {ref.sum().item():.4f}")
    # The 64x64 reference itself has error ~ edge length / 64 per pixel (<= ~1/64 per boundary crossing).
    assert err < 0.03, err
    assert abs(img.sum().item() - ref.sum().item()) < 0.02 * max(1.0, ref.sum().item()) + 0.05
    # The worst pixels against a 1024 x 1024-sample reference of just that pixel (same camera, shifted principal
    # point): the coverage is exact up to float32.
    f = 0.5 * H / math.tan(math.radians(cam["fov_y"]) / 2)
    d = (img - ref).abs().max(-1).values.reshape(-1)
    fine = 0.0
    for k in torch.argsort(d, descending=True)[:3].tolist():
        j, i = divmod(k, W)
        crop = dict(eye=cam["eye"], target=cam["target"], up=cam["up"], focal=f, cx=0.5 * W - i, cy=0.5 * H - j)
        fine = max(fine, (img[j, i] - reference(V, F, C, crop, 1, 1, S=1024)[0, 0]).abs().max().item())
    print(f"[primal] {name}: worst 3 pixels vs 1024x1024 reference: max error {fine:.2e}")
    assert fine < 2e-4, fine


def _fd_check(name, V, F, C, cam, W, H, h, max_coords=60, seed=0):
    torch.manual_seed(seed)
    Vt = _t(V, True)
    Ft = torch.as_tensor(F)
    Ct = _t(C)
    img = render(Vt, Ft, Ct, cam, W, H)
    adj = torch.randn_like(img)
    (img * adj).sum().backward()
    g = Vt.grad.detach().double().reshape(-1)
    rng = np.random.default_rng(seed)
    idx = np.arange(g.numel())
    if idx.size > max_coords:
        # prefer coordinates with non-zero gradient, plus some random ones
        nz = idx[g.abs().cpu().numpy() > 0]
        idx = np.unique(np.concatenate([rng.choice(nz, min(len(nz), max_coords - 10), replace=False),
                                        rng.choice(idx, 10, replace=False)]))
    fd = torch.zeros(len(idx), dtype=torch.float64)
    base = torch.tensor(np.asarray(V), dtype=torch.float64)
    for n, k in enumerate(idx):
        vals = []
        for sgn in (1, -1):
            Vp = base.clone().reshape(-1)
            Vp[k] += sgn * h
            I = render(Vp.reshape(-1, 3).float().to(DEV), Ft, Ct, cam, W, H).detach().double()
            vals.append((I * adj.double()).sum().item())
        fd[n] = (vals[0] - vals[1]) / (2 * h)
    ga = g[idx].cpu()
    rel = (ga - fd).norm().item() / max(fd.norm().item(), 1e-12)
    worst = (ga - fd).abs().max().item() / max(fd.abs().max().item(), 1e-12)
    cos = (ga @ fd).item() / max(ga.norm().item() * fd.norm().item(), 1e-12)
    print(f"\n[grad] {name}: {len(idx)} coords, |g - fd| / |fd| = {rel:.2e}, max err / max |fd| = {worst:.2e}, "
          f"cos = {cos:.6f}")
    return rel, worst


GRAD_CASES = ["triangle", "two_triangles", "box_flat", "box_face_colours", "blob_normals_occluder", "random0",
              "box_on_floor"]


@pytest.mark.parametrize("name", GRAD_CASES)
def test_gradient_vs_fd(name):
    V, F, C = scenes()[name]
    W = H = 32
    cam = dict(CAM, fov_y=45.0)
    # world step ~ 0.003 px of screen motion: the coverage is piecewise smooth (C^1 kinks where a vertex crosses a
    # pixel line or an edge), so larger steps pick up the kinks
    f = 0.5 * H / math.tan(math.radians(cam["fov_y"]) / 2)
    h = 0.003 * 3.0 / f
    rel, worst = _fd_check(name, V, F, C, cam, W, H, h)
    assert rel < 1e-2, rel


def test_deterministic():
    V, F, C = scenes()["blob_normals_occluder"]
    out = []
    for _ in range(2):
        Vt = _t(V, True)
        img = render(Vt, torch.as_tensor(F), _t(C), CAM, 64, 64)
        adj = torch.linspace(-1, 1, img.numel(), device=DEV).reshape(img.shape)
        (img * adj).sum().backward()
        out.append((img.detach().clone(), Vt.grad.clone()))
    assert torch.equal(out[0][0], out[1][0])
    assert torch.equal(out[0][1], out[1][1])
    print("\n[determinism] image and vertex gradient bitwise equal across calls")


def test_near_clip_plane():
    """A floor that extends behind the camera (clipped at the near plane) plus a box: primal vs reference."""
    v, f, _ = box_mesh(center=(0.0, 0.0, 0.0))
    floor_v = np.array([[-20, -1.0, -20], [20, -1.0, -20], [20, -1.0, 20], [-20, -1.0, 20]], np.float64)
    floor_f = np.array([[0, 1, 2], [0, 2, 3]])
    V = np.concatenate([v, floor_v])
    F = np.concatenate([f, floor_f + len(v)])
    C = np.concatenate([np.tile([[1.0, 0.2, 0.2]], (len(f), 1)), np.tile([[0.1, 0.3, 0.9]], (2, 1))])
    W = H = 24
    img = render(_t(V), torch.as_tensor(F), _t(C), CAM, W, H).detach()
    ref = reference(V, F, C, CAM, W, H, S=64)
    err = (img - ref).abs().max().item()
    print(f"\n[near clip] max |I - I_ref| = {err:.2e}")
    assert err < 0.03


def _grid_mesh(n, x0, x1, y0, y1, z, jitter, rng):
    xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)
    X, Y = np.meshgrid(xs, ys)
    V = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
    V[:, 2] += jitter * rng.standard_normal(X.size)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return V, np.r_[np.c_[a, b, d], np.c_[a, d, c]]


@pytest.mark.parametrize("size", ["100k", "300k"])
def test_timing(size):
    """384x384 render + backward of a normal-binned blob (20k / 50k triangles) over a jittered occluder backdrop
    (80k / 250k triangles)."""
    rng = np.random.default_rng(0)
    lat, lon, n = (71, 144, 200) if size == "100k" else (125, 200, 353)
    bv, bf = blob_mesh(n_lat=lat, n_lon=lon, seed=1, r=0.45)
    gv, gf = _grid_mesh(n, -3, 3, -3, 3, -1.0, 0.01, rng)
    V = np.concatenate([bv, gv])
    F = np.concatenate([bf, gf + len(bv)])
    C = np.concatenate([normal_colors(bv, bf), np.zeros((len(gf), 3))])
    Vt, Ft, Ct = _t(V, True), torch.as_tensor(F), _t(C)
    cam = dict(eye=(0.2, 0.3, 2.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov_y=40.0)
    img, info = render(Vt, Ft, Ct, cam, 384, 384, return_info=True)
    img.sum().backward()
    times = []
    for _ in range(5):
        Vt.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = render(Vt, Ft, Ct, cam, 384, 384)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        (img * img).sum().backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        times.append((t1 - t0, t2 - t1))
    fwd, bwd = np.median(np.array(times), 0)
    print(f"\n[timing] {len(F)} triangles, 384x384: forward {fwd * 1e3:.1f} ms, backward {bwd * 1e3:.1f} ms; {info}")
    assert fwd + bwd < 0.5
