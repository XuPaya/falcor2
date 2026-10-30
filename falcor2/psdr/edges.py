# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge lists of the boundary terms: the silhouette edges seen from the camera (primary edges) and the edges that
can bound the visibility of path segments (photon-mapping secondary edges), sampled by length."""

import numpy as np

# Edge record (Edge in slang/falcor2/psdr/boundary.slang): the edge from vertex k to k + 1 of triangle prim of a
# geometry instance, the triangle's neighbor across it (-1 if none) and the cumulative length distribution.
EDGE = np.dtype([("gid", "<u4"), ("prim", "<u4"), ("k", "<u4"), ("prim1", "<i4"), ("cdf", "<f4")])


def edge_topology(faces: np.ndarray, positions: np.ndarray):
    """Edges of a triangle mesh with faces (F, 3) indexing positions (V, 3): edge i is the directed edge from vertex
    k0 to vertex k0 + 1 of face f0, whose third vertex is k0 + 2; f1 is its other face (-1 if none) and boundary is
    whether the edge has no other face (or several). Vertices at the same position are one vertex of the edges:
    meshes split at seams (flat-shaded or textured) would otherwise have each seam edge twice, as two boundary
    edges counted twice."""
    triangles = len(faces)
    joined = np.unique(positions, axis=0, return_inverse=True)[1].reshape(-1)[faces]
    face = np.repeat(np.arange(triangles, dtype=np.uint32), 3)
    k = np.tile(np.arange(3, dtype=np.uint32), triangles)
    a, b = joined[face, k], joined[face, (k + 1) % 3]
    key = np.minimum(a, b).astype(np.uint64) << np.uint64(32) | np.maximum(a, b).astype(np.uint64)
    order = np.argsort(key, kind="stable")
    key, face, k = key[order], face[order], k[order]
    starts = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    counts = np.diff(np.append(starts, len(key)))
    # Cast before selecting the -1 boundary sentinel; NumPy 2 rejects it for unsigned arrays.
    neighbor = face[np.minimum(starts + 1, len(key) - 1)].astype(np.int32)
    f1 = np.where(counts == 2, neighbor, -1)
    return face[starts], k[starts], f1, counts != 2


def classify_edges(gid: int, world: np.ndarray, f0, k0, f1, boundary, origin: np.ndarray):
    """Edge records of a geometry instance with world-space triangle vertices world (F, 3, 3) and edges from
    edge_topology(): the silhouettes seen from origin (primary edges), and the boundary and convex sharp edges
    (secondary edges)."""
    normals = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    length = np.linalg.norm(world[f0, (k0 + 1) % 3] - world[f0, k0], axis=1)
    n1 = normals[np.maximum(f1, 0)]
    side0 = np.einsum("ij,ij->i", normals[f0], origin - world[f0, k0])
    side1 = np.einsum("ij,ij->i", n1, origin - world[f0, k0])
    silhouette = boundary | (side0 * side1 < 0)
    third = world[f0, (k0 + 2) % 3]
    convex = np.einsum("ij,ij->i", n1, world[f0, k0] - third) >= 0
    sharp = np.linalg.norm(normals[f0] - n1, axis=1) > 1e-6
    crease = boundary | (convex & sharp)
    rows = np.zeros(len(f0), EDGE)
    rows["gid"], rows["prim"], rows["k"] = gid, f0, k0
    rows["prim1"] = np.where(boundary, -1, f1)
    rows["cdf"] = length
    return rows[silhouette], rows[crease]


def normalize_cdf(parts):
    """The edge records of all parts with their lengths turned into a cumulative distribution, and the total
    length."""
    rows = np.concatenate(parts) if parts else np.zeros(0, EDGE)
    total = float(rows["cdf"].sum())
    if len(rows):
        rows["cdf"] = np.cumsum(rows["cdf"]) / total
        rows["cdf"][-1] = 1
    return rows, total
