# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from falcor2.psdr.edges import classify_edges, edge_topology


def test_open_mesh_edge_neighbors():
    # A square split into two triangles: four boundary edges and one shared edge.
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    topology = edge_topology(faces, positions)
    f0, k0, f1, boundary = topology

    assert len(f0) == 5
    assert f1.dtype == np.int32
    assert boundary.sum() == 4
    np.testing.assert_array_equal(f1[boundary], [-1, -1, -1, -1])
    np.testing.assert_array_equal(f0[~boundary], [0])
    np.testing.assert_array_equal(f1[~boundary], [1])
    np.testing.assert_array_equal(k0[~boundary], [2])

    silhouettes, creases = classify_edges(0, positions[faces], *topology, np.array([0.5, 0.5, 1]))
    for edges in (silhouettes, creases):
        assert len(edges) == 4
        np.testing.assert_array_equal(edges["prim1"], [-1, -1, -1, -1])
        np.testing.assert_allclose(edges["cdf"], 1)
