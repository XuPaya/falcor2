// SPDX-License-Identifier: Apache-2.0
//
// Deterministic differentiable coverage rasterizer (falcor2.coverage): kernels.
//
// Every pixel (i, j) covers [i, i+1] x [j, j+1] in screen space (x right, y down). The image f(x, y) is the flat
// colour of the nearest triangle (0 where none). Its box-filtered pixel value is computed exactly from the visible
// discontinuity network of f inside the pixel by integration by parts (Green's theorem with the field (x - i, 0)):
//
//   I_ij = f(corner) - sum_{right-border crossings m} (y_m - j) dY_m - sum_{pieces} (c+ - c-) (y1 - y0) ((x0 + x1) / 2 - i)
//
// corner = (i+1, j+1) approached from inside the pixel; a piece is a straight sub-segment (x0, y0) -> (x1, y1) of a
// candidate edge inside the pixel between consecutive split points (pixel borders and crossings with the other
// candidate edges of the pixel), so visibility and the colours on both sides are constant along it; c+ / c- are the
// colours of the nearest faces at the piece midpoint approached along +-n, n = (dy, -dx) / |d| (query()); dY_m = -sign(dx) (c+ - c-) is the jump of f along
// the right border (x = i+1) where the piece leaves it. The backward kernel differentiates exactly this expression
// with respect to the piece endpoints (vertices, border clips, edge-edge crossings) and chains them to the screen
// positions of the edge vertices (64-bit fixed-point integer atomics into grad_xy: deterministic).

static const int MAXC = 8;    // max colour channels
static const int MAXS = 96;   // max split points per (edge, pixel)

static const int END_VA = 0;  // endpoint = edge vertex A (t = 0)
static const int END_VB = 1;  // endpoint = edge vertex B (t = 1)
static const int END_BX = 2;  // endpoint on a vertical pixel border x = const (left)
static const int END_BXR = 3; // endpoint on the right pixel border x = i+1
static const int END_BY = 4;  // endpoint on a horizontal pixel border y = const
static const int END_CROSS = 5; // endpoint = crossing with another candidate edge
static const int END_BXR_V = 6; // endpoint = edge vertex (A if t = 0, B if t = 1) lying exactly on x = i+1

public struct CoverageScene
{
    public StructuredBuffer<float> verts;        // x, y, inverse view depth per vertex
    public StructuredBuffer<uint> faces;         // 3 vertex ids per face
    public StructuredBuffer<float> colors;       // channels per face
    public StructuredBuffer<uint> edges;         // 2 vertex ids per candidate edge
    public StructuredBuffer<uint> pix_offsets;   // width * height + 1
    public StructuredBuffer<uint> pix_edges;     // candidate edge ids per pixel
    public StructuredBuffer<uint> tile_offsets;  // tiles_x * tiles_y + 1
    public StructuredBuffer<uint> tile_faces;    // face ids per tile, nearest (largest face_zmax) first
    public StructuredBuffer<float> face_zmax;    // largest vertex inverse depth per face
    public uint width;
    public uint height;
    public uint channels;
    public uint tile_size;
    public uint tiles_x;
    public uint tiles_y;
    public float eps;
    public float grad_scale;                     // backward: fixed-point scale of grad_xy
}

float2 vxy(CoverageScene s, uint v) { return float2(s.verts[3 * v], s.verts[3 * v + 1]); }

float cross2(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }

/// Nearest face at screen point p + 0 * dir (the limit from direction dir), -1 for none.
/// Coverage is decided at p itself: an edge function that is exactly zero there counts as inside iff dir points into
/// the triangle. With edge = true, p lies on the candidate edge (va, vb): faces containing that edge cover the side
/// of dir iff their third vertex lies on it (symbolic, so slivers thinner than any offset are classified correctly).
/// Depth: inverse view depth at p (a convex combination for covering faces); near-ties (faces through p: e.g. the
/// two faces of a silhouette edge) are broken by the planes' depths at p + eps * dir.
int query(CoverageScene s, float2 p, float2 dir, bool edge, uint va, uint vb)
{
    int tx = clamp(int(floor(p.x / float(s.tile_size))), 0, int(s.tiles_x) - 1);
    int ty = clamp(int(floor(p.y / float(s.tile_size))), 0, int(s.tiles_y) - 1);
    uint t = uint(ty) * s.tiles_x + uint(tx);
    int best = -1;
    float best_z = -1e30, best_zo = -1e30;
    float2 po = p + s.eps * dir;
    float2 pa = edge ? vxy(s, va) : float2(0);
    for (uint k = s.tile_offsets[t]; k < s.tile_offsets[t + 1]; k++)
    {
        uint f = s.tile_faces[k];
        // faces are sorted by their nearest point: none of the rest can be nearer than the best hit
        if (best >= 0 && s.face_zmax[f] < best_z - 1e-6 * abs(best_z))
            break;
        uint ia = s.faces[3 * f], ib = s.faces[3 * f + 1], ic = s.faces[3 * f + 2];
        float2 a = vxy(s, ia), b = vxy(s, ib), c = vxy(s, ic);
        float area = cross2(b - a, c - a);
        if (area == 0)
            continue;
        float o = area > 0 ? 1.0 : -1.0;
        float w0 = cross2(c - b, p - b) * o;
        float w1 = cross2(a - c, p - c) * o;
        float w2 = cross2(b - a, p - a) * o;
        bool covered;
        int own = -1;  // index (0..2) of the opposite vertex when the face contains the edge
        if (edge)
        {
            bool ha = ia == va || ib == va || ic == va, hb = ia == vb || ib == vb || ic == vb;
            if (ha && hb)
                own = (ia != va && ia != vb) ? 0 : ((ib != va && ib != vb) ? 1 : 2);
        }
        if (own >= 0)
        {
            float2 q = own == 0 ? a : (own == 1 ? b : c);
            covered = dot(q - pa, dir) > 0;
            // p is on the edge: the opposite vertex' weight is zero
            if (own == 0) w0 = 0; else if (own == 1) w1 = 0; else w2 = 0;
        }
        else
        {
            float g0 = cross2(c - b, dir) * o, g1 = cross2(a - c, dir) * o, g2 = cross2(b - a, dir) * o;
            covered = (w0 > 0 || (w0 == 0 && g0 > 0)) && (w1 > 0 || (w1 == 0 && g1 > 0)) &&
                      (w2 > 0 || (w2 == 0 && g2 > 0));
        }
        if (!covered)
            continue;
        float za = s.verts[3 * ia + 2], zb = s.verts[3 * ib + 2], zc = s.verts[3 * ic + 2];
        float ws = w0 + w1 + w2;
        float iz = ws > 0 ? (max(w0, 0) * za + max(w1, 0) * zb + max(w2, 0) * zc) / (max(w0, 0) + max(w1, 0) + max(w2, 0))
                          : max(za, max(zb, zc));
        float v0 = cross2(c - b, po - b), v1 = cross2(a - c, po - c), v2 = cross2(b - a, po - a);
        float izo = (v0 * za + v1 * zb + v2 * zc) / (v0 + v1 + v2);
        float tol = 1e-6 * abs(iz);
        bool nearer = iz > best_z + tol || (iz >= best_z - tol && (izo > best_zo || (izo == best_zo && int(f) < best)));
        if (best < 0 || nearer)
        {
            best_z = iz;
            best_zo = izo;
            best = int(f);
        }
    }
    return best;
}

float color(CoverageScene s, int f, uint ch) { return f < 0 ? 0.0 : s.colors[uint(f) * s.channels + ch]; }

/// Fixed-point vertex gradients (scale s.grad_scale, set per backward call): integer atomics are associative, so the
/// accumulated gradient does not depend on the order of the threads (bitwise deterministic).
int64_t fixed(float x, float scale) { return int64_t(clamp(round(x * scale), -4.0e18, 4.0e18)); }

void add_grad(CoverageScene s, RWByteAddressBuffer g, uint v, float2 w)
{
    g.InterlockedAddI64(v * 16, fixed(w.x, s.grad_scale));
    g.InterlockedAddI64(v * 16 + 8, fixed(w.y, s.grad_scale));
}

struct Endpoint
{
    float t;
    int kind;
    uint other;   // END_CROSS: the other edge
    float s;      // END_CROSS: parameter on the other edge
}

/// dL/dP of an endpoint P(t) of edge A->B chained to the vertices.
void endpoint_backward(CoverageScene s, RWByteAddressBuffer grad, uint va, uint vb, float2 D, Endpoint e, float2 w)
{
    if (e.kind == END_VA || (e.kind == END_BXR_V && e.t == 0)) { add_grad(s, grad, va, w); return; }
    if (e.kind == END_VB || e.kind == END_BXR_V) { add_grad(s, grad, vb, w); return; }
    float2 gn;
    uint vc = 0, vd = 0;
    if (e.kind == END_BX || e.kind == END_BXR)
        gn = float2(1, 0);
    else if (e.kind == END_BY)
        gn = float2(0, 1);
    else
    {
        vc = s.edges[2 * e.other];
        vd = s.edges[2 * e.other + 1];
        float2 E = vxy(s, vd) - vxy(s, vc);
        gn = float2(-E.y, E.x);
    }
    float den = dot(gn, D);
    if (den == 0)
        return;
    float q = dot(w, D) / den;
    float2 wt = w - gn * q;
    add_grad(s, grad, va, (1 - e.t) * wt);
    add_grad(s, grad, vb, e.t * wt);
    if (e.kind == END_CROSS)
    {
        add_grad(s, grad, vc, (1 - e.s) * gn * q);
        add_grad(s, grad, vd, e.s * gn * q);
    }
}

void clip_side(float p, float q, int kind, inout float t0, inout int k0, inout float t1, inout int k1, inout bool empty)
{
    if (p == 0)
    {
        if (q < 0)
            empty = true;
        return;
    }
    float r = q / p;
    if (p < 0)
    {
        if (r > t0) { t0 = r; k0 = kind; }
    }
    else
    {
        if (r < t1) { t1 = r; k1 = kind; }
    }
}

/// The pixel's value (forward) or its vertex gradients for adjoint g (backward).
void process_pixel(CoverageScene s, uint i, uint j, bool backward, float g[MAXC], RWByteAddressBuffer grad,
                   out float img[MAXC])
{
    float2 org = float2(float(i), float(j));
    int fc = query(s, org + float2(1, 1), float2(-0.70710678, -0.70710678), false, 0, 0);
    for (uint ch = 0; ch < MAXC; ch++)
        img[ch] = ch < s.channels ? color(s, fc, ch) : 0;

    uint pix = j * s.width + i;
    uint lb = s.pix_offsets[pix], le = s.pix_offsets[pix + 1];
    Endpoint sp[MAXS + 2];
    for (uint k = lb; k < le; k++)
    {
        uint e = s.pix_edges[k];
        uint va = s.edges[2 * e], vb = s.edges[2 * e + 1];
        float2 A = vxy(s, va), B = vxy(s, vb);
        float2 D = B - A;
        float len = length(D);
        if (len == 0)
            continue;
        float t0 = 0, t1 = 1;
        int k0 = END_VA, k1 = END_VB;
        bool empty = false;
        clip_side(-D.x, A.x - org.x, END_BX, t0, k0, t1, k1, empty);
        clip_side(D.x, org.x + 1 - A.x, END_BXR, t0, k0, t1, k1, empty);
        clip_side(-D.y, A.y - org.y, END_BY, t0, k0, t1, k1, empty);
        clip_side(D.y, org.y + 1 - A.y, END_BY, t0, k0, t1, k1, empty);
        if (empty || !(t1 > t0))
            continue;
        // Half-open pixels [i, i+1) x [j, j+1): a vertical edge on the right border belongs to the next pixel.
        if (D.x == 0 && A.x == org.x + 1)
            continue;
        // An edge vertex exactly on the right border is a right-border crossing as well.
        if (k0 == END_VA && A.x == org.x + 1)
            k0 = END_BXR_V;
        if (k1 == END_VB && B.x == org.x + 1)
            k1 = END_BXR_V;

        // Split points: crossings with the other candidate edges of the pixel, sorted by t (insertion sort).
        int n = 0;
        sp[n++] = { t0, k0, 0, 0 };
        for (uint k2 = lb; k2 < le; k2++)
        {
            uint e2 = s.pix_edges[k2];
            if (e2 == e)
                continue;
            uint vc = s.edges[2 * e2], vd = s.edges[2 * e2 + 1];
            if (vc == va || vc == vb || vd == va || vd == vb)
                continue;
            float2 C = vxy(s, vc), E = vxy(s, vd) - C;
            float den = cross2(D, E);
            if (abs(den) <= 1e-12 * len * length(E))
                continue;
            float2 AC = C - A;
            float t = cross2(AC, E) / den;
            float sc = cross2(AC, D) / den;
            if (!(t > t0 && t < t1 && sc > 0 && sc < 1) || n > MAXS)
                continue;
            int m = n++;
            while (m > 1 && sp[m - 1].t > t)
            {
                sp[m] = sp[m - 1];
                m--;
            }
            sp[m] = { t, END_CROSS, e2, sc };
        }
        sp[n++] = { t1, k1, 0, 0 };

        float2 nrm = float2(D.y, -D.x) / len;
        for (int p = 0; p + 1 < n; p++)
        {
            Endpoint e0 = sp[p], e1 = sp[p + 1];
            float2 P0 = A + D * e0.t - org, P1 = A + D * e1.t - org;
            float2 M = org + 0.5 * (P0 + P1);
            int fp = query(s, M, nrm, true, va, vb);
            int fm = query(s, M, -nrm, true, va, vb);
            if (fp == fm)
                continue;
            float dc[MAXC] = {};
            bool any = false;
            for (uint ch = 0; ch < MAXC; ch++)
            {
                dc[ch] = ch < s.channels ? color(s, fp, ch) - color(s, fm, ch) : 0;
                any = any || dc[ch] != 0;
            }
            if (!any)
                continue;
            float ydy = P1.y - P0.y, xm = 0.5 * (P0.x + P1.x);
            // Right-border crossing at an end of this piece (x = i+1): -(y_m - j) dY with dY = -sign(D.x) dc.
            float sgn = D.x > 0 ? 1.0 : -1.0;
            // (crossings exactly at y = j+1 belong to the pixel above: half-open in y)
            bool r0 = (e0.kind == END_BXR || e0.kind == END_BXR_V) && P0.y < 1;
            bool r1 = (e1.kind == END_BXR || e1.kind == END_BXR_V) && P1.y < 1;
            if (!backward)
            {
                for (uint ch = 0; ch < MAXC; ch++)
                {
                    img[ch] -= dc[ch] * ydy * xm;
                    if (r0) img[ch] += P0.y * sgn * dc[ch];
                    if (r1) img[ch] += P1.y * sgn * dc[ch];
                }
            }
            else
            {
                float a = 0;
                for (uint ch = 0; ch < MAXC; ch++)
                    a += g[ch] * dc[ch];
                if (a == 0)
                    continue;
                // T = -a ydy xm (+ a sgn P.y for right-border ends)
                float2 w0 = float2(-0.5 * a * ydy, a * xm);
                float2 w1 = float2(-0.5 * a * ydy, -a * xm);
                if (r0) w0.y += a * sgn;
                if (r1) w1.y += a * sgn;
                endpoint_backward(s, grad, va, vb, D, e0, w0);
                endpoint_backward(s, grad, va, vb, D, e1, w1);
            }
        }
    }
}

public void render_pixels(uint2 dispatchThreadID, CoverageScene s, RWStructuredBuffer<float> image,
                          RWByteAddressBuffer dummy)
{
    uint i = dispatchThreadID.x, j = dispatchThreadID.y;
    if (i >= s.width || j >= s.height)
        return;
    float g[MAXC] = {};
    for (uint ch = 0; ch < MAXC; ch++)
        g[ch] = 0;
    float img[MAXC];
    process_pixel(s, i, j, false, g, dummy, img);
    uint base = (j * s.width + i) * s.channels;
    for (uint ch = 0; ch < s.channels; ch++)
        image[base + ch] = img[ch];
}

public void backward_pixels(uint2 dispatchThreadID, CoverageScene s, StructuredBuffer<float> grad_image,
                            RWByteAddressBuffer grad_xy)
{
    uint i = dispatchThreadID.x, j = dispatchThreadID.y;
    if (i >= s.width || j >= s.height)
        return;
    uint pix = j * s.width + i;
    if (s.pix_offsets[pix] == s.pix_offsets[pix + 1])
        return;
    float g[MAXC] = {};
    bool any = false;
    for (uint ch = 0; ch < MAXC; ch++)
    {
        g[ch] = ch < s.channels ? grad_image[pix * s.channels + ch] : 0;
        any = any || g[ch] != 0;
    }
    if (!any)
        return;
    float img[MAXC];
    process_pixel(s, i, j, true, g, grad_xy, img);
}
