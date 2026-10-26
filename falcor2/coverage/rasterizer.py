# SPDX-License-Identifier: Apache-2.0
"""Deterministic differentiable coverage rasterizer: torch preprocessing, binning and the autograd Function (see
falcor2.coverage for the algorithm)."""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import slangpy as spy
import torch

MAX_CHANNELS = 8  # MAXC in coverage.slang
SLANG_ROOT = Path(__file__).resolve().parents[2] / "slang"

_DEVICE = None
_MODULE = None


def device() -> spy.Device:
    """The CUDA slangpy device shared with PyTorch (created on first use)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = spy.create_torch_device(type=spy.DeviceType.cuda, include_paths=[SLANG_ROOT],
                                          enable_hot_reload=False)
    return _DEVICE


def module() -> spy.Module:
    global _MODULE
    if _MODULE is None:
        _MODULE = spy.Module.load_from_file(device(), "falcor2/coverage/coverage.slang")
    return _MODULE


_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.shared


def _buffer(t: torch.Tensor) -> spy.Buffer:
    t = t.contiguous()
    if t.dtype == torch.int64 or t.dtype == torch.int32:
        t = t.to(torch.int32)
    else:
        t = t.to(torch.float32)
    n = max(t.numel(), 1)
    buf = device().create_buffer(size=4 * n, usage=_USAGE)
    if t.numel():
        spy.copy_torch_tensor_to_buffer(t.reshape(-1), buf)
    return buf


def _zeros(n: int) -> spy.Buffer:
    buf = device().create_buffer(size=4 * max(n, 1), usage=_USAGE)
    spy.copy_torch_tensor_to_buffer(torch.zeros(max(n, 1), dtype=torch.float32, device="cuda"), buf)
    return buf


def _to_torch(buf: spy.Buffer, n: int) -> torch.Tensor:
    out = torch.empty(n, dtype=torch.float32, device="cuda")
    spy.copy_buffer_to_torch_tensor(buf, out)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Camera and geometry preprocessing (torch, differentiable where it matters)

def camera_frame(camera: dict, width: int, height: int):
    """eye, rows (right, down, forward) of the view matrix, focal length and principal point in pixels."""
    eye = torch.as_tensor(camera["eye"], dtype=torch.float64)
    target = torch.as_tensor(camera["target"], dtype=torch.float64)
    up = torch.as_tensor(camera.get("up", (0.0, 1.0, 0.0)), dtype=torch.float64)
    fwd = target - eye
    fwd = fwd / fwd.norm()
    right = torch.linalg.cross(fwd, up)
    right = right / right.norm()
    upv = torch.linalg.cross(right, fwd)
    if "focal" in camera:
        f = float(camera["focal"])
    else:
        f = 0.5 * height / math.tan(math.radians(float(camera["fov_y"])) / 2)
    cx = float(camera.get("cx", 0.5 * width))
    cy = float(camera.get("cy", 0.5 * height))
    return eye, torch.stack([right, -upv, fwd]), f, cx, cy


def weld(vertices: torch.Tensor, faces: torch.Tensor):
    """Vertices at identical positions merged (their mean, so gradients split evenly); returns (positions, faces)."""
    _, inv = torch.unique(vertices.detach(), dim=0, return_inverse=True)
    n = int(inv.max()) + 1 if inv.numel() else 0
    cnt = torch.zeros(n, dtype=vertices.dtype, device=vertices.device).index_add_(
        0, inv, torch.ones_like(vertices[:, 0]))
    pos = torch.zeros(n, 3, dtype=vertices.dtype, device=vertices.device).index_add(0, inv, vertices) / cnt[:, None]
    return pos, inv[faces]


def clip_plane(pc: torch.Tensor, faces: torch.Tensor, fid: torch.Tensor, dist: torch.Tensor):
    """Clips the indexed mesh (positions pc, faces with source face ids fid) to dist >= 0 (dist: per-vertex signed
    distance, linear in position). Crossing vertices are shared by the faces of a crossing edge (the mesh stays
    indexed); partially clipped faces become 1 or 2 triangles."""
    behind = dist.detach() < 0
    nb = behind[faces].sum(1)
    if not bool((nb > 0).any()):
        return pc, faces, fid
    keep = nb == 0
    part = (nb > 0) & (nb < 3)
    pf, pid = faces[part], fid[part]
    e = pf[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    lo, hi = e.min(1).values, e.max(1).values
    cross = behind[lo] != behind[hi]
    nv = pc.shape[0]
    ukey, uinv = torch.unique((lo * nv + hi)[cross], return_inverse=True)
    a, b = ukey // nv, ukey % nv
    t = (dist[a] / (dist[a] - dist[b])).unsqueeze(1)
    pos = torch.cat([pc, pc[a] + t * (pc[b] - pc[a])])
    xid = torch.full((e.shape[0],), -1, dtype=torch.int64, device=faces.device)
    xid[cross] = nv + uinv
    xid = xid.reshape(-1, 3)  # crossing vertex of edge (k, k+1) of each partial face
    # Polygon of each partial face: for k = 0..2 the vertex k (if kept), then the crossing of edge k (if any).
    bk = behind[pf]
    polys = []
    for k in range(3):
        polys.append(torch.where(bk[:, k], torch.full_like(pf[:, k], -1), pf[:, k]))
        polys.append(xid[:, k])
    poly = torch.stack(polys, 1)  # (P, 6) with -1 holes; 3 or 4 valid entries in order
    valid = poly >= 0
    poly = torch.gather(poly, 1, torch.argsort((~valid).to(torch.int8), dim=1, stable=True))
    quad = valid.sum(1) == 4
    new_faces = torch.cat([faces[keep], poly[:, [0, 1, 2]], poly[quad][:, [0, 2, 3]]])
    new_fid = torch.cat([fid[keep], pid, pid[quad]])
    return pos, new_faces, new_fid


def frustum_clip(pc: torch.Tensor, faces: torch.Tensor, near: float, f: float, cx: float, cy: float, width: int,
                 height: int, guard: float):
    """Clips camera-space geometry (x right, y down, z forward) to z >= near and to the guard band
    [-guard, width + guard] x [-guard, height + guard] (pixels) around the image, which bounds the screen coordinates
    (float32 precision) and drops off-screen geometry. Returns (positions, faces, source face ids)."""
    fid = torch.arange(faces.shape[0], device=faces.device)
    pc, faces, fid = clip_plane(pc, faces, fid, pc[:, 2] - near)
    for sx, sy, off in ((1, 0, cx + guard), (-1, 0, width + guard - cx), (0, 1, cy + guard),
                        (0, -1, height + guard - cy)):
        pc, faces, fid = clip_plane(pc, faces, fid, f * (sx * pc[:, 0] + sy * pc[:, 1]) + off * pc[:, 2])
    return pc, faces, fid


def project(pc: torch.Tensor, f: float, cx: float, cy: float):
    """Screen x, y (pixels, y down) and inverse view depth of camera-space points (z forward)."""
    z = pc[:, 2]
    iz = 1.0 / torch.where(z > 0, z, torch.ones_like(z))  # unused vertices behind the camera stay finite
    return torch.stack([cx + f * pc[:, 0] * iz, cy + f * pc[:, 1] * iz], 1), iz


def candidate_edges(xy: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor):
    """Unique edges that can carry a discontinuity: boundary / non-manifold edges, edges between faces of different
    colours and silhouette edges (both faces on the same side of the edge in screen space)."""
    F = faces.shape[0]
    e = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    opp = faces[:, [2, 0, 1]].reshape(-1)
    fid = torch.arange(F, device=faces.device).repeat_interleave(3)
    lo, hi = e.min(1).values, e.max(1).values
    nv = xy.shape[0]
    key = lo * nv + hi
    ukey, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
    order = torch.argsort(inv, stable=True)
    start = torch.cumsum(counts, 0) - counts
    o1 = order[start]
    o2 = order[torch.clamp(start + 1, max=order.numel() - 1)]
    a, b = ukey // nv, ukey % nv
    two = counts == 2
    f1, f2 = fid[o1], fid[o2]
    same = (colors[f1] == colors[f2]).all(1)
    pa, pb = xy[a], xy[b]
    d = pb - pa

    def side(v):
        r = xy[v] - pa
        return d[:, 0] * r[:, 1] - d[:, 1] * r[:, 0]

    opposite = side(opp[o1]) * side(opp[o2]) < 0
    cull = two & same & opposite
    cand = ~cull & (d.abs().sum(1) > 0)
    return torch.stack([a[cand], b[cand]], 1)


def bin_edges(xy: torch.Tensor, edges: torch.Tensor, width: int, height: int, margin: float = 1e-3):
    """(pixel offsets (H*W+1), edge ids) of the pixels each edge may touch (conservative)."""
    dev = xy.device
    if edges.shape[0] == 0:
        return torch.zeros(width * height + 1, dtype=torch.int64, device=dev), torch.zeros(0, dtype=torch.int64,
                                                                                            device=dev)
    p0, p1 = xy[edges[:, 0]].double(), xy[edges[:, 1]].double()
    d = p1 - p0
    # Liang-Barsky clip to the image (expanded by the margin).
    t0 = torch.zeros(d.shape[0], dtype=torch.float64, device=dev)
    t1 = torch.ones_like(t0)
    ok = torch.ones_like(t0, dtype=torch.bool)
    for p, q in ((-d[:, 0], p0[:, 0] + margin), (d[:, 0], width + margin - p0[:, 0]),
                 (-d[:, 1], p0[:, 1] + margin), (d[:, 1], height + margin - p0[:, 1])):
        zero = p == 0
        ok &= ~(zero & (q < 0))
        r = q / torch.where(zero, torch.ones_like(p), p)
        t0 = torch.where(~zero & (p < 0), torch.maximum(t0, r), t0)
        t1 = torch.where(~zero & (p > 0), torch.minimum(t1, r), t1)
    ok &= t1 >= t0
    ids = torch.nonzero(ok).squeeze(1)
    q0, q1 = p0[ids] + t0[ids, None] * d[ids], p0[ids] + t1[ids, None] * d[ids]
    out_pix, out_edge = [], []
    xmajor = (q1 - q0)[:, 0].abs() >= (q1 - q0)[:, 1].abs()
    for major in (0, 1):
        sel = xmajor if major == 0 else ~xmajor
        if not bool(sel.any()):
            continue
        a, b, eid = q0[sel], q1[sel], ids[sel]
        mn = 1 - major
        lo = torch.minimum(a[:, major], b[:, major]) - margin
        hi = torch.maximum(a[:, major], b[:, major]) + margin
        size = width if major == 0 else height
        osize = height if major == 0 else width
        c0 = torch.clamp(torch.floor(lo), 0, size - 1).long()
        c1 = torch.clamp(torch.floor(hi), 0, size - 1).long()
        n = c1 - c0 + 1
        rep = torch.repeat_interleave(torch.arange(a.shape[0], device=dev), n)
        start = torch.cumsum(n, 0) - n
        c = c0[rep] + torch.arange(rep.numel(), device=dev) - start[rep]
        # minor-axis range of the segment over [c, c+1] (clamped to the segment)
        A, B = a[rep], b[rep]
        dd = B - A
        den = torch.where(dd[:, major] == 0, torch.ones_like(dd[:, major]), dd[:, major])
        ta = torch.clamp((c.double() - margin - A[:, major]) / den, 0, 1)
        tb = torch.clamp((c.double() + 1 + margin - A[:, major]) / den, 0, 1)
        flat = dd[:, major] == 0
        ta = torch.where(flat, torch.zeros_like(ta), ta)
        tb = torch.where(flat, torch.ones_like(tb), tb)
        ma, mb = A[:, mn] + ta * dd[:, mn], A[:, mn] + tb * dd[:, mn]
        r0 = torch.clamp(torch.floor(torch.minimum(ma, mb) - margin), 0, osize - 1).long()
        r1 = torch.clamp(torch.floor(torch.maximum(ma, mb) + margin), 0, osize - 1).long()
        m = r1 - r0 + 1
        rep2 = torch.repeat_interleave(torch.arange(c.numel(), device=dev), m)
        start2 = torch.cumsum(m, 0) - m
        r = r0[rep2] + torch.arange(rep2.numel(), device=dev) - start2[rep2]
        cc = c[rep2]
        col, row = (cc, r) if major == 0 else (r, cc)
        out_pix.append(row * width + col)
        out_edge.append(eid[rep[rep2]])
    if not out_pix:
        return torch.zeros(width * height + 1, dtype=torch.int64, device=dev), torch.zeros(0, dtype=torch.int64,
                                                                                            device=dev)
    pix = torch.cat(out_pix)
    edg = torch.cat(out_edge)
    # unique (pixel, edge) pairs sorted by pixel
    key = torch.unique(pix * edges.shape[0] + edg)
    pix, edg = key // edges.shape[0], key % edges.shape[0]
    offsets = torch.zeros(width * height + 1, dtype=torch.int64, device=dev)
    offsets[1:] = torch.cumsum(torch.bincount(pix, minlength=width * height), 0)
    return offsets, edg


def bin_faces(xy: torch.Tensor, faces: torch.Tensor, width: int, height: int, tile: int, zkey: torch.Tensor,
              margin: float = 0.01):
    """(tile offsets, face ids) of the tiles (tile x tile pixels) each face's screen triangle may overlap (bbox +
    separating-edge test), each tile's faces sorted by decreasing zkey (nearest first)."""
    dev = xy.device
    tx, ty = (width + tile - 1) // tile, (height + tile - 1) // tile
    v = xy[faces].double()  # (F, 3, 2)
    lo, hi = v.min(1).values - margin, v.max(1).values + margin
    x0 = torch.clamp(torch.floor(lo[:, 0] / tile), 0, tx - 1).long()
    x1 = torch.clamp(torch.floor(hi[:, 0] / tile), 0, tx - 1).long()
    y0 = torch.clamp(torch.floor(lo[:, 1] / tile), 0, ty - 1).long()
    y1 = torch.clamp(torch.floor(hi[:, 1] / tile), 0, ty - 1).long()
    on = (hi[:, 0] >= 0) & (lo[:, 0] <= width) & (hi[:, 1] >= 0) & (lo[:, 1] <= height)
    area = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 2, 0] - v[:, 0, 0])
    on &= area != 0
    nx = torch.where(on, x1 - x0 + 1, torch.zeros_like(x0))
    ny = torch.where(on, y1 - y0 + 1, torch.zeros_like(y0))
    n = nx * ny
    fid = torch.repeat_interleave(torch.arange(faces.shape[0], device=dev), n)
    start = torch.cumsum(n, 0) - n
    k = torch.arange(fid.numel(), device=dev) - start[fid]
    cx = x0[fid] + k % nx[fid]
    cy = y0[fid] + k // nx[fid]
    # Separating-axis test against the face's edges: drop tiles entirely outside one edge.
    big = n[fid] > 1
    if bool(big.any()):
        vv = v[fid]
        sgn = torch.sign(area[fid])
        keep = torch.ones_like(fid, dtype=torch.bool)
        corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for k0 in range(3):
            a, b = vv[:, k0], vv[:, (k0 + 1) % 3]
            e = b - a
            outside = torch.ones_like(keep)
            for ox, oy in corners:
                px = (cx + ox).double() * tile + (margin if ox else -margin)
                py = (cy + oy).double() * tile + (margin if oy else -margin)
                w = (e[:, 0] * (py - a[:, 1]) - e[:, 1] * (px - a[:, 0])) * sgn
                outside &= w < 0
            keep &= ~outside
        keep |= ~big
        fid, cx, cy = fid[keep], cx[keep], cy[keep]
    tid = cy * tx + cx
    rank = torch.empty_like(zkey, dtype=torch.int64)
    rank[torch.argsort(-zkey, stable=True)] = torch.arange(zkey.numel(), device=dev)
    order = torch.argsort(tid * faces.shape[0] + rank[fid])
    tid, fid = tid[order], fid[order]
    offsets = torch.zeros(tx * ty + 1, dtype=torch.int64, device=dev)
    offsets[1:] = torch.cumsum(torch.bincount(tid, minlength=tx * ty), 0)
    return offsets, fid, tx, ty


# ----------------------------------------------------------------------------------------------------------------------
# Kernels

class _Raster:
    """Device buffers of one prepared frame (screen vertices, faces, colours, bins)."""

    def __init__(self, xy, iz, faces, colors, edges, pix_off, pix_edges, tile_off, tile_faces, tx, ty, width, height,
                 tile, eps, zmax):
        self.nv = xy.shape[0]
        self.width, self.height, self.channels = width, height, colors.shape[1]
        verts = torch.cat([xy.detach().float(), iz.detach().float()[:, None]], 1)
        self.scene = {
            "verts": _buffer(verts), "faces": _buffer(faces), "colors": _buffer(colors.detach()),
            "edges": _buffer(edges), "pix_offsets": _buffer(pix_off), "pix_edges": _buffer(pix_edges),
            "tile_offsets": _buffer(tile_off), "tile_faces": _buffer(tile_faces), "face_zmax": _buffer(zmax),
            "width": width, "height": height, "channels": self.channels, "tile_size": tile, "tiles_x": tx,
            "tiles_y": ty, "eps": float(eps), "grad_scale": 0.0}

    def forward(self) -> torch.Tensor:
        n = self.width * self.height * self.channels
        out = device().create_buffer(size=4 * n, usage=_USAGE)
        dummy = _zeros(2)
        module().render_pixels.dispatch(thread_count=spy.uint3(self.width, self.height, 1), s=self.scene,
                                        image=out, dummy=dummy)
        return _to_torch(out, n).reshape(self.height, self.width, self.channels)

    def backward(self, grad_image: torch.Tensor) -> torch.Tensor:
        gmax = float(grad_image.detach().abs().max()) if grad_image.numel() else 0.0
        if gmax == 0 or not math.isfinite(gmax):
            return torch.zeros(self.nv, 2, dtype=torch.float32, device="cuda")
        # 2^-32 max|dL/dI| per contribution resolution, 2^30 max|dL/dI| range (64-bit fixed point)
        scale = 2.0 ** 32 / gmax
        g = _buffer(grad_image.detach().float())
        gxy = _zeros(4 * self.nv)  # int64 per coordinate
        module().backward_pixels.dispatch(thread_count=spy.uint3(self.width, self.height, 1),
                                          s=dict(self.scene, grad_scale=scale), grad_image=g, grad_xy=gxy)
        out = torch.empty(2 * self.nv, dtype=torch.int64, device="cuda")
        spy.copy_buffer_to_torch_tensor(gxy, out)
        return (out.double() / scale).float().reshape(self.nv, 2)


class _CoverageFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, raster):
        ctx.raster = raster
        return raster.forward()

    @staticmethod
    def backward(ctx, grad):
        return ctx.raster.backward(grad).to(grad.dtype), None


def render(vertices: torch.Tensor, faces: torch.Tensor, face_colors: torch.Tensor, camera: dict, width: int,
           height: int, *, near: float = 1e-3, guard: float = 1.0, tile: int = 2, eps: float = 1e-3,
           weld_vertices: bool = True, return_info: bool = False):
    """Box-filtered image (H, W, C) of flat-coloured triangles, differentiable w.r.t. vertices (see falcor2.coverage).

    vertices (N, 3) world positions (CUDA tensor; may require grad), faces (F, 3) int, face_colors (F, C) with
    C <= 8 (zeros = black occluders), camera {eye, target, up, fov_y (deg, vertical) or focal (pixels), optional
    cx, cy (principal point, pixels)}. near: near-plane distance (camera units); guard: guard band around the image
    (in image sizes) that geometry is clipped to; tile: query cell size (pixels); eps: depth tie-break offset (pixels);
    weld_vertices: merge vertices at identical positions. return_info: also return counts of the prepared frame.
    """
    dev = torch.device("cuda")
    vertices = vertices.to(dev)
    faces = torch.as_tensor(faces, device=dev).long()
    colors = torch.as_tensor(face_colors, device=dev, dtype=torch.float32)
    if colors.ndim == 1:
        colors = colors[:, None]
    if colors.shape[1] > MAX_CHANNELS:
        raise ValueError(f"at most {MAX_CHANNELS} colour channels")
    eye, rot, f, cx, cy = camera_frame(camera, width, height)
    pos, fc = weld(vertices, faces) if weld_vertices else (vertices, faces)
    pc = (pos.double() - eye.to(dev)) @ rot.to(dev).T
    pc, fc, src = frustum_clip(pc, fc, near, f, cx, cy, width, height, guard * max(width, height))
    xy64, iz = project(pc, f, cx, cy)
    xy = xy64.float()
    cols = colors[src]
    with torch.no_grad():
        edges = candidate_edges(xy.detach(), fc, cols)
        pix_off, pix_edges = bin_edges(xy.detach(), edges, width, height)
        zmax = iz.detach()[fc].max(1).values.float()  # nearest point of each face (inverse depth)
        tile_off, tile_faces, tx, ty = bin_faces(xy.detach(), fc, width, height, tile, zmax)
    raster = _Raster(xy, iz, fc, cols, edges, pix_off, pix_edges, tile_off, tile_faces, tx, ty, width, height, tile,
                     eps, zmax)
    img = _CoverageFn.apply(xy, raster)
    if return_info:
        info = {"faces": int(fc.shape[0]), "edges": int(edges.shape[0]), "pixel_edge_entries": int(pix_edges.numel()),
                "tile_face_entries": int(tile_faces.numel()), "vertices": int(xy.shape[0])}
        return img, info
    return img
