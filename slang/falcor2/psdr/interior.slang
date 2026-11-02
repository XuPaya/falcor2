// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing psdr;

// The primal image and the interior term (derivatives of path contributions in material form).

/// Frames per seed: camera sample i of a pass with seed s is frame s * FRAMES_PER_SEED + i, as the reference path
/// tracer's `iteration` (falcor2 toolkits render frame seed * 65536 + i).
static const uint FRAMES_PER_SEED = 65536;

/// Running compensated sums of the primal image (the AccumulatorNode's single_compensated history).
public RWStructuredBuffer<float3> primal_sum;
public RWStructuredBuffer<float3> primal_residual;

/// Camera path of pixel sample i (the same paths in all kernels: the reference path tracer's path of frame
/// seed * FRAMES_PER_SEED + i), recorded into `slot`; `antithetic` and `dual` as in trace_path().
void trace_camera_path(uint2 pixel, uint i, uint spp, bool antithetic, bool dual, uint slot, out PathRecord rec)
{
    UniformSampleGenerator sg = UniformSampleGenerator(pixel, seed * FRAMES_PER_SEED + i);
    trace_path(primary_ray(pixel, sg), slot, sg, rec, antithetic, dual);
}

float neumaier_correction(float sum, float value, float next)
{
    float correction;
    if (abs(sum) >= abs(value))
        correction = fma(-1.f, next, sum) + value;
    else
        correction = fma(-1.f, next, value) + sum;
    return correction;
}

// The primal image, the interior and the pixel boundary term are separate kernels: the derivative code
// of the interior term makes its kernel much heavier than the others. With antithetic sampling, a camera path whose
// first vertex samples a glossy reflection is averaged with its antithetic partner (see trace_path()) in the interior
// term; the primal image is the plain estimate.

/// Frame `frame` (sample `sample_index` of the pass) of the primal image: the camera path's radiance plus the frame's
/// light splats (light_primal), accumulated as falcor2's AccumulatorNode accumulates the reference / adjoint path
/// tracer's frames (non-finite samples dropped, compensated sum); primal holds the running mean.
public void render_primal(uint2 pixel, uint frame, uint sample_index)
{
    uint index = pixel.y * camera.dims.x + pixel.x;
    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frame);
    PathRecord rec;
    trace_path(primary_ray(pixel, sg), NO_SLOT, sg, rec);
    float3 value = rec.radiance;
    if (light_tracing)
        value = value + asfloat(light_primal.Load3(index * 12));
    value = !all(isfinite(value)) ? float3(0.f) : value;
    if (sample_index == 0) {
        primal_sum[index] = value;
        primal_residual[index] = float3(0.f);
        primal[index] = value;
        return;
    }
    float3 sum = primal_sum[index];
    float3 next = sum + value;
    float3 residual = primal_residual[index] + float3(neumaier_correction(sum.x, value.x, next.x),
                                                      neumaier_correction(sum.y, value.y, next.y),
                                                      neumaier_correction(sum.z, value.z, next.z));
    primal_sum[index] = next;
    primal_residual[index] = residual;
    float inverse_count = 1.f / float(sample_index + 1);
    primal[index] = (next + residual) * inverse_count;
}

/// Samples first, ..., first + count - 1 of spp (one dispatch per chunk of samples: long dispatches time out) of pixel
/// (tid.x, row0 + tid.y) (one dispatch per chunk of rows: at most path_slots pixels, one record slot each).
public void render_interior(uint2 tid, uint row0, uint spp, uint first, uint count, bool antithetic)
{
    uint2 pixel = uint2(tid.x, row0 + tid.y);
    uint slot = tid.y * camera.dims.x + tid.x;
    uint index = pixel.y * camera.dims.x + pixel.x;
    float3 adjoint = backward ? dl_di[index] / spp : float3(0);
    float3 derivative = 0;
    for (uint i = first; i < first + count; i++) {
        [loop]
        for (uint branch = 0; branch < 2; branch++) {
            PathRecord rec;
            trace_camera_path(pixel, i, spp, antithetic, branch == 1, slot, rec);
            float weight = rec.antithetic ? 0.5 : 1.0;
            float3 d;
            differentiate_path(rec, true, adjoint * weight, d);
            // Non-finite derivatives of degenerate samples are dropped, as backward mode drops their gradients.
            if (all(isfinite(d)))
                derivative += d * weight;
            if (!rec.antithetic)
                break;
        }
    }
    if (!backward)
        interior[index] += derivative / spp;
}

// Split backward interior term: record_interior() traces the camera paths and writes one work item per surface vertex
// (the vertex's record and the adjoints of its terms); backward_vertices() differentiates one vertex per thread. The
// derivative code then runs in a kernel without the tracer and the path record, in short uniform threads.

/// A vertex of a camera path with the adjoints of its terms (VertexTerms.Differential: emission and nee share theirs).
struct VertexItem
{
    VertexRecord v;
    float3 d_terms;
    float3 d_step;
    float limit;
};

RWStructuredBuffer<VertexItem> vertex_items;
/// Items written (uint at offset 0; may exceed the capacity, see vertex_item_capacity).
public RWByteAddressBuffer vertex_item_count;
public uint vertex_item_capacity;

void append_vertex_item(VertexItem item)
{
    // One atomic per wave: the active lanes' items are consecutive.
    uint count = WaveActiveCountBits(true);
    uint base = 0;
    if (WaveIsFirstLane())
        vertex_item_count.InterlockedAdd(0, count, base);
    uint index = WaveReadLaneFirst(base) + WavePrefixCountBits(true);
    if (index < vertex_item_capacity)
        vertex_items[index] = item;
}

/// Backward mode: records the vertices of path sample first + tid.y / rows of pixel (tid.x, row0 + tid.y % rows) (one
/// record slot per thread: at most path_slots threads per dispatch).
public void record_interior(uint2 tid, uint spp, uint first, uint row0, uint rows, bool antithetic)
{
    uint slot = tid.y * camera.dims.x + tid.x;
    uint2 pixel = uint2(tid.x, row0 + tid.y % rows);
    uint i = first + tid.y / rows;
    uint index = pixel.y * camera.dims.x + pixel.x;
    float3 adjoint = dl_di[index] / spp;
    [loop]
    for (uint branch = 0; branch < 2; branch++) {
        PathRecord rec;
        trace_camera_path(pixel, i, spp, antithetic, branch == 1, slot, rec);
        float3 weight = adjoint * (rec.antithetic ? 0.5 : 1.0);
        float3 value = path_terms(rec);
        if (rec.length == 0 && any(rec.miss_scale > 0)) {
            // The camera sees the environment map.
            var theta = diffPair(0.0);
            bwd_diff(env_map_radiance)(theta, rec.miss_dir, weight);
        }
        float limit = max_relative_vertex_derivative > 0
            ? max(max_relative_vertex_derivative * length(weight * value), 1e-30) : 0.0;
        float3 prefix = 1;
        for (uint k = 0; k < rec.length; k++) {
            const PathVertex pv = path_vertex(rec, k);
            VertexItem item;
            item.d_terms = weight * prefix;
            item.d_step = weight * prefix * pv.tail;
            if ((pv.flags & VERTEX_VOLUME) == 0 && (any(item.d_terms != 0) || any(item.d_step != 0))) {
                item.v = vertex_record(rec, k);
                item.limit = limit;
                append_vertex_item(item);
            }
            prefix *= pv.terms.step;
        }
        if (!rec.antithetic)
            break;
    }
}

/// Backward mode: the interior-term gradients of recorded vertex tid.
public void backward_vertices(uint tid)
{
    if (tid >= min(vertex_item_count.Load(0), vertex_item_capacity))
        return;
    VertexItem item = vertex_items[tid];
    if (!backward_material_hit(item.v.hit))
        return;
    vertex_derivative_limit = item.limit;
    VertexTerms.Differential d;
    d.emission = item.d_terms;
    d.nee = item.d_terms;
    d.step = item.d_step;
    var theta = diffPair(0.0);
    bwd_diff(vertex_terms)(theta, item.v, true, d);
    if (frame_pending)
        append_frame_item();
}

/// The adjoint of a vertex's shading frame (frame_adjoint), chained to the stored vertex normals and tangents by
/// backward_frames().
struct FrameItem
{
    HitRef hit;
    float handedness;
    float limit;
    float3 d_normal;
    float3 d_tangent;
    float3 d_bitangent;
};

RWStructuredBuffer<FrameItem> frame_items;
/// Items written (uint at offset 0; at most one per vertex item, so frame_item_capacity >= vertex_item_capacity).
public RWByteAddressBuffer frame_item_count;
public uint frame_item_capacity;

void append_frame_item()
{
    frame_pending = false;
    FrameItem item;
    item.hit = frame_hit;
    item.handedness = frame_handedness;
    item.limit = vertex_derivative_limit;
    item.d_normal = frame_adjoint.normal;
    item.d_tangent = frame_adjoint.tangent;
    item.d_bitangent = frame_adjoint.bitangent;
    uint index;
    frame_item_count.InterlockedAdd(0, 1, index);
    if (index < frame_item_capacity)
        frame_items[index] = item;
}

/// Backward mode: the frame gradients (frame_grad) of frame item tid.
public void backward_frames(uint tid)
{
    if (tid >= min(frame_item_count.Load(0), frame_item_capacity))
        return;
    FrameItem item = frame_items[tid];
    vertex_derivative_limit = item.limit;
    Frame.Differential d = Frame.dzero();
    d.normal = item.d_normal;
    d.tangent = item.d_tangent;
    d.bitangent = item.d_bitangent;
    var theta = diffPair(0.0);
    bwd_diff(interpolated_frame)(theta, item.hit, item.handedness, d);
}
