# SPDX-License-Identifier: Apache-2.0
"""Deterministic differentiable coverage rasterizer for flat-coloured triangle meshes.

    img = falcor2.coverage.render(vertices, faces, face_colors, camera, width, height)   # (H, W, C) torch tensor
    img.backward(dL_dimg)                                                                 # -> vertices.grad

vertices (N, 3) world positions (CUDA torch tensor, may require grad), faces (F, 3), face_colors (F, C) per-face
constant colours (C <= 8; black occluders = zeros; empty pixels are 0), camera dict(eye, target, up, fov_y [deg,
vertical] or focal [px], optional cx, cy principal point [px]): a pinhole camera with falcor2's conventions (row 0 =
top). Both windings rasterize; the nearest surface wins. No lighting, no Monte Carlo: image and gradient are
deterministic (two calls are bitwise identical, gradients included).

Primal. Pixel (i, j) covers [i, i+1) x [j, j+1); its value is the exact box-filtered (area-averaged) visible colour,
computed from the discontinuity network of the visible colour field f inside the pixel by integration by parts
(Green's theorem with the field (x - i, 0)):

    I_ij = f(corner) + sum_{pieces ending on x = i+1} sign(dx) (y_m - j) (c+ - c-)
                     - sum_{pieces} (c+ - c-) (y1 - y0) ((x0 + x1)/2 - i)

corner = (i+1, j+1) approached from inside the pixel. The pieces are the straight sub-segments of the candidate edges
(boundary and non-manifold edges, edges between differently coloured faces, silhouette edges: both faces on the same
side of the projected edge) inside the pixel, split at the pixel border and at every crossing with another candidate
edge of the pixel. Between consecutive split points the visibility of the edge and the colours on its two sides are
constant (occlusion boundaries and the colour changes behind a silhouette all lie on other candidate edges), so each
piece is classified exactly by two point queries at its midpoint M: the nearest face at M approached from either side
(c+ along n = (dy, -dx)/|d|, c-), with symbolic coverage (faces that contain the edge cover the side of their third
vertex, so slivers and folds are classified correctly) and depth ties at M broken by the planes' depths at M +- eps n.
A hidden piece sees the same occluder on both sides and drops out. Queries test the faces binned to the query's cell
(tile x tile pixels), sorted nearest-first with an early exit.

Gradient. The backward kernel differentiates exactly the expression above w.r.t. the piece endpoints, with the
classification fixed (exact almost everywhere), chained to the screen positions of the vertices that define them
(edge vertices; border clips slide along the border; crossings depend on both edges), accumulated with 64-bit
fixed-point integer atomics (order-independent, 2^-32 max|dL/dI| resolution per contribution); torch then
backpropagates through the projection, the clipping and the welding. For a consistent network this is the boundary
integral sum over visible boundary pieces of (c_in - c_out) (dx/dV . n) ds; edges between same-coloured faces that
are not silhouettes are culled. Central differences of render() agree with backward() to ~1e-4 relative (float32),
except within a step of the C^1 kinks of the coverage (a vertex crossing a pixel line or another edge).

Accuracy knobs (render keyword arguments): eps (depth tie-break offset, px, default 1e-3), near (near plane, camera
units, default 1e-3) and guard (guard band in image sizes, default 1): geometry is clipped in 3D to z >= near and
to the band [-guard W, (1+guard) W] x [-guard H, (1+guard) H] of the image (indexed: crossing vertices are shared and
differentiable), which bounds the screen coordinates for float32; tile (query cell, px, default 2: speed only);
weld_vertices (merge vertices at identical positions so unwelded meshes / UV seams do not create fake boundary
edges; merged vertices take the mean position, so the gradient is split evenly between duplicates -- correct when
the duplicates move together). Residual error: float32 round-off (measured 1e-6 .. 3e-5 per pixel against a
1024 x 1024-sample reference), and misclassification only where another boundary passes within float precision of
a piece midpoint or a pixel corner.

Limitations: interpenetrating / touching triangles are not handled -- the depth test per point is exact, but no
split points or gradients are generated along intersection curves, so pixels on them are wrong (use contact-free
layouts). No depth gradients (flat colours do not depend on depth). At most 96 crossings per (edge, pixel) (more are
dropped). Cost grows with the candidate edges per pixel (quadratic per pixel) and the faces per query cell.

Files: falcor2/coverage/rasterizer.py (torch preprocessing: camera, welding, clipping, candidate edges, binning;
autograd Function) and slang/falcor2/coverage/coverage.slang (per-pixel forward / backward kernels on a CUDA slangpy
device that shares PyTorch's context). Tests: tests/python/coverage/test_coverage.py (run with pytest --noconftest).
"""

from falcor2.coverage.rasterizer import render

__all__ = ["render"]
