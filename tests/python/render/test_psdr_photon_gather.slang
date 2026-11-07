// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Exported so the generated kernel module sees IPsdrTraversal and PipelineTraversal for its SceneTraversal export.
__exported import falcor2.psdr.psdr;

RWStructuredBuffer<uint> result_indices;
RWStructuredBuffer<float> result_distances;

public void probe_photon_gather(float3 origin, float3 dir, float distance)
{
    Neighbors neighbors = gather(origin, dir, distance);
    result_indices[0] = neighbors.count;
    result_distances[0] = neighbors.max_dist;
    for (uint i = 0; i < neighbors.count; i++) {
        result_indices[i + 1] = neighbors.index[i];
        result_distances[i + 1] = neighbors.dist[i];
    }
}
