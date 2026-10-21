// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// SpongeCake BSDF (Wang, Jin, Hasan, Yan: "SpongeCake: A Layered Microflake Surface Appearance Model", ACM TOG 2023):
// a thin slab of volumetric layers without reflecting interfaces, here two layers (top and body) and an optional
// opaque Lambertian base. A layer is a medium of SGGX microflakes (specular microflake phase function; surface-like
// flakes or fibers) or a Henyey-Greenstein medium (the paper's forward-scattering layers, e.g. the translucent fibers
// of sheer fabric; anisotropy g = 1 - roughness).
//
// - Single scattering is the paper's closed form (its Eqs. 11-16), summed over the layers with the attenuation of the
//   layers in between: f = albedo p(wi, wo) sigma(wi) G / (|cos i| |cos o|) with the phase function p (SGGX:
//   D(h) / (4 sigma(wi))), G = (1 - exp(-tau (L_i + L_o))) / (L_i + L_o) for reflection and
//   G = (exp(-tau L_o) - exp(-tau L_i)) / (L_i - L_o) for transmission, L(w) = sigma(w) / |cos w| (sigma = 1 for HG).
// - Unscattered light passes as a delta transmission exp(-sum tau L(w)) (optional).
// - Multiple scattering replaces the paper's parameter-mapping network (an extra single-scattering lobe with mapped
//   parameters plus a Lambertian lobe) by per-material tables from a white random walk through the slab:
//   m(w) = 1 - E1(|cos w|) - t(w) is the energy a white slab scatters more than once (E1: its single-scattering
//   albedo), split between the sides by the walk's share w_R(|cos wi|), m_R = w_R m, m_T = (1 - w_R) m. Each side
//   leaves as a mixture of a diffuse lobe of Kulla-Conty form m_s(wi) m_s(wo) |cos o| / (pi avg(m_s)) and a forward
//   lobe beta_s m_s(wi) F(wo): an HG lobe of anisotropy g_s about the travel direction -wi (transmission) or its
//   mirror image (reflection), folded into the exit hemisphere so that it integrates to one (repeated scattering in
//   a forward medium keeps the light near its direction: n HG(g) scatterings compose to HG(g^n)). beta_s and g_s are
//   maximum-likelihood fits to the walk's exit directions. Both lobes conserve the white slab's energy exactly; colored
//   albedos scale them by c_s(gamma) = E[gamma^n | n >= 2, side] (n: the scattering order) of the walk.
// - The base: a Lambertian base below the slab (base_weight of the transmitted light), with the slab's interreflection
//   in closed form: b rho T(wi) T(wo) / (pi (1 - rho R_avg)), T(w) the slab's total transmittance.
//
// The tables (SpongeCakeMaterial::update(), src/falcor2/render/material/spongecake_material.cpp) depend on the
// optical thicknesses, roughnesses and layer types only; their derivatives with respect to the thicknesses and
// roughnesses are tabulated too and enter through `table_tangent` (falcor2.psdr). Tables without the forward lobes
// (forward_lobes = false: an older material build) use a constant w_R and diffuse lobes only.

/// Directions (|cos theta|) per table row, albedo samples of the multiple-scattering scale, bins of w_R(|cos|).
public static const uint SPONGECAKE_TABLE_MU = 32;
public static const uint SPONGECAKE_TABLE_GAMMA = 32;
public static const uint SPONGECAKE_TABLE_SHARE = 16;
/// Float offsets in the table buffer: single-scattering albedos E[layer][R/T][mu], their derivatives
/// dE[p][layer][R/T][mu] with p = (top thickness, top roughness, body thickness, body roughness), the multiple
/// scattering scales c[R/T][gamma] at gamma = 1 - (1 - g / 31)^2, the cosine-weighted averages of the delta
/// transmission (tbar, dtbar[p]) and of E (Ebar[layer][R/T], dEbar[p][layer][R/T]), w_R and the layers' shares
/// of the multiply scattered events, and the derivatives dc[p][R/T][gamma] and d(w_R, shares)[p][3].
/// Forward-lobe tables: w_R[16] per |cos wi|, the lobes (beta_R, g_R, beta_T, g_T), avg(m_R), avg(m_T), and their
/// derivatives d w_R[p][16], d lobes[p][4], d avg[p][2].
public static const uint SPONGECAKE_TAB_E = 0;
public static const uint SPONGECAKE_TAB_DE = 128;
public static const uint SPONGECAKE_TAB_MS = 640;
public static const uint SPONGECAKE_TAB_TBAR = 704;
public static const uint SPONGECAKE_TAB_EBAR = 709;
public static const uint SPONGECAKE_TAB_DEBAR = 713;
public static const uint SPONGECAKE_TAB_WR = 729;
public static const uint SPONGECAKE_TAB_DMS = 732;
public static const uint SPONGECAKE_TAB_DWR = 988;
public static const uint SPONGECAKE_TAB_SHARE = 1000;
public static const uint SPONGECAKE_TAB_LOBE = 1016;
public static const uint SPONGECAKE_TAB_MSAVG = 1020;
public static const uint SPONGECAKE_TAB_DSHARE = 1022;
public static const uint SPONGECAKE_TAB_DLOBE = 1086;
public static const uint SPONGECAKE_TAB_DMSAVG = 1102;
public static const uint SPONGECAKE_TAB_SIZE = 1110;

/// (1 - exp(-x)) / x for x >= 0.
[Differentiable]
float spongecake_phi(float x)
{
    if (detach(x) < 1e-3)
        return 1.0 - x * (0.5 - x / 6.0);
    return (1.0 - exp(-x)) / x;
}

/// Henyey-Greenstein phase function of the cosine between the travel directions before and after scattering.
[Differentiable]
float spongecake_hg(float cos_theta, float g)
{
    float d = max(1.0 + g * g - 2.0 * g * cos_theta, 1e-8);
    return (1.0 - g * g) / (4.0 * float.PI * d * sqrt(d));
}

/// HG lobe of anisotropy g about the unit direction c folded into the hemisphere of wo (integrates to 1 over it).
[Differentiable]
float spongecake_folded_hg(float3 c, float3 wo, float g)
{
    return spongecake_hg(dot(c, wo), g) + spongecake_hg(dot(c, float3(wo.x, wo.y, -wo.z)), g);
}

/// Direction with an HG(g) distributed cosine to the unit direction d.
float3 spongecake_sample_hg(float3 d, float g, float2 u)
{
    float c;
    if (abs(g) < 1e-3) {
        c = 1.0 - 2.0 * u.x;
    } else {
        float s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        c = clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
    }
    float3 t, b;
    branchless_ONB(d, t, b);
    float st = sqrt(max(1.0 - c * c, 0.0)), phi = 2.0 * float.PI * u.y;
    return normalize(st * cos(phi) * t + st * sin(phi) * b + c * d);
}

/// A slab of SGGX microflakes: surface-like flakes (S = diag(a^2, a^2, 1) in the shading frame) or fibers along the
/// tangent (S = I - (1 - a^2) t t^T); or a Henyey-Greenstein medium (hg) with anisotropy g = 1 - alpha.
public struct SpongeCakeLayer : IDifferentiable
{
    public float3 albedo;
    /// Optical thickness: the extinction along the normal for unit projected area.
    public float tau;
    /// SGGX roughness in [0.01, 1] (HG: 1 - g).
    public float alpha;
    public bool fiber;
    public bool hg;

    /// S = c.x I + c.y a a^T with the flake axis a (normal or tangent).
    [Differentiable]
    float2 sggx()
    {
        float a2 = alpha * alpha;
        return fiber ? float2(1.0, a2 - 1.0) : float2(a2, 1.0 - a2);
    }

    float3 axis() { return fiber ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 1.0); }

    /// HG anisotropy.
    [Differentiable]
    public float anisotropy() { return 1.0 - alpha; }

    /// Projected area sigma(w) = sqrt(w^T S w) (1 for HG).
    [Differentiable]
    public float sigma(float3 w)
    {
        if (hg)
            return 1.0;
        float2 c = sggx();
        float d = fiber ? w.x : w.z;
        return sqrt(max(c.x + c.y * d * d, 1e-12));
    }

    /// sigma(w) / |cos w|.
    [Differentiable]
    public float lambda(float3 w) { return sigma(w) / max(abs(w.z), 1e-4); }

    /// SGGX distribution D(m) = 1 / (pi sqrt(det S) (m^T S^-1 m)^2).
    [Differentiable]
    public float D(float3 m)
    {
        float2 c = sggx();
        float d = fiber ? m.x : m.z;
        float s = c.x + c.y;
        float q = (1.0 - d * d) / c.x + d * d / s;
        return 1.0 / (float.PI * c.x * sqrt(s) * q * q);
    }

    /// p(wi, wo) sigma(wi): D(h) / 4 (specular microflakes) or the HG phase function (h = normalize(wi + wo)).
    [Differentiable]
    public float scattering(float3 wi, float3 wo, float3 h)
    {
        if (hg)
            return spongecake_hg(-dot(wi, wo), anisotropy());
        return 0.25 * D(h);
    }

    /// Phase function p(wi, wo) (the sampling density of sample_direction()).
    public float phase(float3 wi, float3 wo, float3 h) { return scattering(wi, wo, h) / sigma(wi); }

    /// Visible normal of the flakes seen from wi (Heitz et al. 2015, "The SGGX microflake distribution").
    public float3 sample_visible_normal(float3 wi, float2 u)
    {
        float2 c = sggx();
        float3 a = axis();
        float3 wk, wj;
        branchless_ONB(wi, wk, wj);
        float ak = dot(a, wk), aj = dot(a, wj), ai = dot(a, wi);
        float s_kk = c.x + c.y * ak * ak, s_jj = c.x + c.y * aj * aj, s_ii = c.x + c.y * ai * ai;
        float s_kj = c.y * ak * aj, s_ki = c.y * ak * ai, s_ji = c.y * aj * ai;
        float det = abs(
            s_kk * s_jj * s_ii - s_kj * s_kj * s_ii - s_ki * s_ki * s_jj - s_ji * s_ji * s_kk + 2.0 * s_kj * s_ki * s_ji
        );
        float inv_sqrt_s_ii = 1.0 / sqrt(s_ii);
        float tmp = sqrt(max(s_jj * s_ii - s_ji * s_ji, 1e-20));
        float3 m_k = float3(sqrt(det) / tmp, 0.0, 0.0);
        float3 m_j = float3(-inv_sqrt_s_ii * (s_ki * s_ji - s_kj * s_ii) / tmp, inv_sqrt_s_ii * tmp, 0.0);
        float3 m_i = float3(inv_sqrt_s_ii * s_ki, inv_sqrt_s_ii * s_ji, inv_sqrt_s_ii * s_ii);
        float r = sqrt(u.x), phi = 2.0 * float.PI * u.y;
        float x = r * cos(phi), y = r * sin(phi);
        float3 m = normalize(x * m_k + y * m_j + sqrt(max(1.0 - x * x - y * y, 0.0)) * m_i);
        return m.x * wk + m.y * wj + m.z * wi;
    }

    /// Scattered direction of light arriving from wi, sampled from the phase function.
    public float3 sample_direction(float3 wi, float2 u)
    {
        if (hg)
            return spongecake_sample_hg(-wi, anisotropy(), u);
        float3 m = sample_visible_normal(wi, u);
        return 2.0 * dot(wi, m) * m - wi;
    }
};

/// The SpongeCake BSDF in the shading frame (wi.z > 0; the material orients the frame towards wi).
public struct SpongeCakeBSDF : IDifferentiable
{
    public SpongeCakeLayer top;
    public SpongeCakeLayer body;
    public float3 base_color;
    public float base_weight;
    public bool delta_transmission;
    /// The tables have the forward multiple-scattering lobes and w_R(|cos|).
    public bool forward_lobes;
    public BufferHandle tables;
    /// Derivative carriers of the tables (zero values): top thickness, top roughness, body thickness, body
    /// roughness. Only used with table_derivatives.
    public float4 table_tangent;
    public bool table_derivatives;

    /// Sets the layer types from SpongeCakeMaterial's layer_flags.
    [mutating]
    public void set_layer_flags(uint flags)
    {
        body.fiber = (flags & 1) != 0;
        top.fiber = (flags & 2) != 0;
        delta_transmission = (flags & 4) != 0;
        body.hg = (flags & 8) != 0;
        top.hg = (flags & 16) != 0;
        forward_lobes = (flags & 32) != 0;
    }

    float load(uint i) { return tables.load<float>(4 * i); }

    /// Table entry i plus its derivatives (rows of `stride` floats from `first`) along table_tangent.
    [Differentiable]
    float entry(no_diff uint i, no_diff uint first, no_diff uint stride)
    {
        float value = no_diff load(i);
        if (table_derivatives) {
            [ForceUnroll]
            for (uint p = 0; p < 4; p++)
                value += table_tangent[p] * no_diff load(first + p * stride);
        }
        return value;
    }

    /// White single-scattering albedo of `layer` (0: top, 1: body) into reflection (rt = 0) or transmission (1).
    [Differentiable]
    float single_albedo(no_diff uint layer, no_diff uint rt, float mu)
    {
        uint row = (layer * 2 + rt) * SPONGECAKE_TABLE_MU;
        float x = clamp(mu * float(SPONGECAKE_TABLE_MU) - 0.5, 0.0, float(SPONGECAKE_TABLE_MU - 1));
        uint i = min(uint(detach(x)), SPONGECAKE_TABLE_MU - 2);
        float f = x - float(i);
        float a = entry(SPONGECAKE_TAB_E + row + i, SPONGECAKE_TAB_DE + row + i, 128);
        float b = entry(SPONGECAKE_TAB_E + row + i + 1, SPONGECAKE_TAB_DE + row + i + 1, 128);
        return lerp(a, b, f);
    }

    [Differentiable]
    float single_albedo_avg(no_diff uint layer, no_diff uint rt)
    {
        uint k = layer * 2 + rt;
        return entry(SPONGECAKE_TAB_EBAR + k, SPONGECAKE_TAB_DEBAR + k, 4);
    }

    /// Unscattered transmittance along w.
    [Differentiable]
    public float transmittance(float3 w) { return exp(-(top.tau * top.lambda(w) + body.tau * body.lambda(w))); }

    /// Energy of a white slab scattered more than once (incident along w), and its cosine-weighted average.
    [Differentiable]
    float multiple(float3 w)
    {
        float mu = abs(w.z);
        float e = single_albedo(0, 0, mu) + single_albedo(0, 1, mu) + single_albedo(1, 0, mu) + single_albedo(1, 1, mu);
        return max(1.0 - e - transmittance(w), 0.0);
    }

    [Differentiable]
    float multiple_avg()
    {
        float e = single_albedo_avg(0, 0) + single_albedo_avg(0, 1) + single_albedo_avg(1, 0) + single_albedo_avg(1, 1);
        return 1.0 - e - entry(SPONGECAKE_TAB_TBAR, SPONGECAKE_TAB_TBAR + 1, 1);
    }

    /// Share w_R of the multiply scattered light leaving at the top, for light incident with |cos| = mu.
    [Differentiable]
    float reflected_share(float mu)
    {
        if (!forward_lobes)
            return entry(SPONGECAKE_TAB_WR, SPONGECAKE_TAB_DWR, 3);
        float x = clamp(mu * float(SPONGECAKE_TABLE_SHARE) - 0.5, 0.0, float(SPONGECAKE_TABLE_SHARE - 1));
        uint i = min(uint(detach(x)), SPONGECAKE_TABLE_SHARE - 2);
        float a = entry(SPONGECAKE_TAB_SHARE + i, SPONGECAKE_TAB_DSHARE + i, SPONGECAKE_TABLE_SHARE);
        float b = entry(SPONGECAKE_TAB_SHARE + i + 1, SPONGECAKE_TAB_DSHARE + i + 1, SPONGECAKE_TABLE_SHARE);
        return lerp(a, b, x - float(i));
    }

    /// Multiply scattered energy of a white slab leaving through reflection (rt = 0) or transmission (1), m_side(w).
    [Differentiable]
    float multiple_side(no_diff uint rt, float3 w, float m)
    {
        float w_r = reflected_share(abs(w.z));
        return (rt == 0 ? w_r : 1.0 - w_r) * m;
    }

    /// Cosine-weighted average of m_side.
    [Differentiable]
    float multiple_side_avg(no_diff uint rt)
    {
        if (!forward_lobes) {
            float w_r = entry(SPONGECAKE_TAB_WR, SPONGECAKE_TAB_DWR, 3);
            return (rt == 0 ? w_r : 1.0 - w_r) * multiple_avg();
        }
        return entry(SPONGECAKE_TAB_MSAVG + rt, SPONGECAKE_TAB_DMSAVG + rt, 2);
    }

    /// Forward lobe of side rt: weight beta (k = 0) or anisotropy g (k = 1); no forward lobes in older tables.
    [Differentiable]
    float lobe(no_diff uint rt, no_diff uint k)
    {
        if (!forward_lobes)
            return k == 0 ? 0.0 : 0.5;
        uint i = 2 * rt + k;
        return entry(SPONGECAKE_TAB_LOBE + i, SPONGECAKE_TAB_DLOBE + i, 4);
    }

    /// Center of the forward lobe: the travel direction -wi (transmission) or its mirror image (reflection).
    [Differentiable]
    float3 lobe_center(no_diff uint rt, float3 wi) { return rt == 1 ? -wi : float3(-wi.x, -wi.y, wi.z); }

    /// Albedo scale of the multiply scattered light leaving through reflection (rt = 0) or transmission (1):
    /// E[gamma^n | n >= 2] of the white random walk at the albedo gamma = top^f_top body^f_body (per channel).
    [Differentiable]
    float3 multiple_scale(no_diff uint rt)
    {
        float f_top = entry(SPONGECAKE_TAB_WR + 1, SPONGECAKE_TAB_DWR + 1, 3);
        float f_body = entry(SPONGECAKE_TAB_WR + 2, SPONGECAKE_TAB_DWR + 2, 3);
        float3 g = exp(f_top * log(max(top.albedo, 1e-4)) + f_body * log(max(body.albedo, 1e-4)));
        float3 u = 1.0 - sqrt(1.0 - min(g, 0.9999));
        float3 result;
        [ForceUnroll]
        for (uint c = 0; c < 3; c++) {
            float x = clamp(u[c], 0.0, 1.0) * float(SPONGECAKE_TABLE_GAMMA - 1);
            uint i = min(uint(detach(x)), SPONGECAKE_TABLE_GAMMA - 2);
            uint e = rt * SPONGECAKE_TABLE_GAMMA + i;
            float a = entry(SPONGECAKE_TAB_MS + e, SPONGECAKE_TAB_DMS + e, 2 * SPONGECAKE_TABLE_GAMMA);
            float b = entry(SPONGECAKE_TAB_MS + e + 1, SPONGECAKE_TAB_DMS + e + 1, 2 * SPONGECAKE_TABLE_GAMMA);
            result[c] = lerp(a, b, x - float(i));
        }
        return result;
    }

    /// Total transmittance of the slab (unscattered, single and multiple scattering) along w.
    [Differentiable]
    float3 total_transmittance(float3 w, float m)
    {
        float mu = abs(w.z);
        return transmittance(w) + top.albedo * single_albedo(0, 1, mu) + body.albedo * single_albedo(1, 1, mu)
            + multiple_scale(1) * multiple_side(1, w, m);
    }

    /// Value f(wi, wo) |cos wo| without the delta transmission.
    [Differentiable]
    public float3 eval(float3 wi, float3 wo)
    {
        if (wi.z < BSDF_MIN_COS_THETA || abs(wo.z) < BSDF_MIN_COS_THETA)
            return float3(0.0);
        bool reflection = wo.z > 0.0;
        float3 sum = wi + wo;
        float3 h = sum / max(length(sum), 1e-7);
        float li0 = top.lambda(wi), lo0 = top.lambda(wo), li1 = body.lambda(wi), lo1 = body.lambda(wo);
        float g0, g1;
        if (reflection) {
            g0 = top.tau * spongecake_phi(top.tau * (li0 + lo0));
            g1 = exp(-top.tau * (li0 + lo0)) * body.tau * spongecake_phi(body.tau * (li1 + lo1));
        } else {
            g0 = exp(-body.tau * lo1 - top.tau * min(li0, lo0)) * top.tau * spongecake_phi(top.tau * abs(li0 - lo0));
            g1 = exp(-top.tau * li0 - body.tau * min(li1, lo1)) * body.tau * spongecake_phi(body.tau * abs(li1 - lo1));
        }
        float3 f = (top.albedo * (top.scattering(wi, wo, h) * g0) + body.albedo * (body.scattering(wi, wo, h) * g1))
            / wi.z;

        float cos_o = abs(wo.z);
        uint rt = reflection ? 0 : 1;
        float mi = 0.0, mo = 0.0;
        float m_avg = multiple_avg();
        if (detach(m_avg) > 1e-4) {
            mi = multiple(wi);
            mo = multiple(wo);
            float msi = multiple_side(rt, wi, mi);
            float avg = multiple_side_avg(rt);
            float ms = 0.0;
            if (detach(avg) > 1e-6)
                ms = msi * multiple_side(rt, wo, mo) * cos_o / (float.PI * avg);
            float beta = lobe(rt, 0);
            if (detach(beta) > 0.0)
                ms = lerp(ms, msi * spongecake_folded_hg(lobe_center(rt, wi), wo, lobe(rt, 1)), beta);
            f += multiple_scale(rt) * ms;
        }
        if (!reflection)
            return f * (1.0 - base_weight);
        if (base_weight > 0.0) {
            float3 r_avg = top.albedo * single_albedo_avg(0, 0) + body.albedo * single_albedo_avg(1, 0)
                + multiple_scale(0) * max(multiple_side_avg(0), 0.0);
            f += base_weight * base_color * total_transmittance(wi, mi) * total_transmittance(wo, mo) * cos_o
                / (float.PI * max(1.0 - base_color * r_avg, 1e-4));
        }
        return f;
    }

    /// Lobe selection probabilities for wi: top and body single scattering, multiple scattering reflection and
    /// transmission, base and delta transmission.
    void lobe_probabilities(float3 wi, out float p[6])
    {
        float mu = wi.z;
        float t = 1.0 - base_weight;
        p[0] = luminance(top.albedo) * (single_albedo(0, 0, mu) + t * single_albedo(0, 1, mu));
        p[1] = luminance(body.albedo) * (single_albedo(1, 0, mu) + t * single_albedo(1, 1, mu));
        float m_avg = multiple_avg();
        float m = m_avg > 1e-4 ? multiple(wi) : 0.0;
        p[2] = luminance(multiple_scale(0)) * multiple_side(0, wi, m);
        p[3] = t * luminance(multiple_scale(1)) * multiple_side(1, wi, m);
        p[4] = base_weight > 0.0 ? base_weight * luminance(base_color * total_transmittance(wi, m)) : 0.0;
        p[5] = delta_transmission ? t * transmittance(wi) : 0.0;
        float total = p[0] + p[1] + p[2] + p[3] + p[4] + p[5];
        [ForceUnroll]
        for (uint i = 0; i < 6; i++)
            p[i] = total > 0.0 ? p[i] / total : 0.0;
    }

    /// Solid-angle pdf of the multiple-scattering lobe of side rt (forward lobe or cosine).
    float pdf_multiple(uint rt, float3 wi, float3 wo)
    {
        float cosine = abs(wo.z) * float.INV_PI;
        float beta = lobe(rt, 0);
        if (!(beta > 0.0))
            return cosine;
        return lerp(cosine, spongecake_folded_hg(lobe_center(rt, wi), wo, lobe(rt, 1)), beta);
    }

    /// Solid-angle pdf of the non-delta lobes.
    float pdf_lobes(float3 wi, float3 wo, float p[6])
    {
        if (wi.z < BSDF_MIN_COS_THETA || abs(wo.z) < BSDF_MIN_COS_THETA)
            return 0.0;
        float3 sum = wi + wo;
        float3 h = sum / max(length(sum), 1e-7);
        float pdf = p[0] * top.phase(wi, wo, h) + p[1] * body.phase(wi, wo, h);
        if (wo.z > 0.0)
            pdf += p[2] * pdf_multiple(0, wi, wo) + p[4] * abs(wo.z) * float.INV_PI;
        else
            pdf += p[3] * pdf_multiple(1, wi, wo);
        return pdf;
    }

    public float pdf(float3 wi, float3 wo)
    {
        float p[6];
        lobe_probabilities(wi, p);
        return pdf_lobes(wi, wo, p);
    }

    public bool sample<S : ISampleGenerator>(const float3 wi, out BSDFSample sample, inout S sg)
    {
        sample = {};
        if (wi.z < BSDF_MIN_COS_THETA)
            return false;
        float p[6];
        lobe_probabilities(wi, p);
        float u = next_float(sg);
        float2 u2 = next_float2(sg);
        float c_ms = p[0] + p[1] + p[2] + p[3];
        if (u < p[0] + p[1]) {
            // Single scattering: the phase function of the chosen layer.
            sample.wo = u < p[0] ? top.sample_direction(wi, u2) : body.sample_direction(wi, u2);
            sample.flags = sample.wo.z > 0.0 ? BSDFFlags::glossy_reflection : BSDFFlags::glossy_transmission;
        } else if (u < c_ms) {
            // Multiple scattering: the forward lobe (folded into the side's hemisphere) or a cosine lobe.
            uint rt = u < p[0] + p[1] + p[2] ? 0 : 1;
            float start = rt == 0 ? p[0] + p[1] : p[0] + p[1] + p[2];
            float v = (u - start) / max(p[2 + rt], 1e-12);
            if (v < lobe(rt, 0)) {
                sample.wo = spongecake_sample_hg(lobe_center(rt, wi), lobe(rt, 1), u2);
                if ((sample.wo.z > 0.0) != (rt == 0))
                    sample.wo.z = -sample.wo.z;
                sample.flags = rt == 0 ? BSDFFlags::glossy_reflection : BSDFFlags::glossy_transmission;
            } else {
                sample.wo = square_to_cosine_hemisphere_concentric(u2);
                if (rt == 1)
                    sample.wo.z = -sample.wo.z;
                sample.flags = rt == 0 ? BSDFFlags::diffuse_reflection : BSDFFlags::diffuse_transmission;
            }
        } else if (u < c_ms + p[4]) {
            sample.wo = square_to_cosine_hemisphere_concentric(u2);
            sample.flags = BSDFFlags::diffuse_reflection;
        } else {
            if (p[5] <= 0.0)
                return false;
            sample.wo = -wi;
            sample.pdf = p[5];
            sample.weight = float3((1.0 - base_weight) * transmittance(wi) / p[5]);
            sample.flags = BSDFFlags::delta_transmission;
            return true;
        }
        sample.pdf = pdf_lobes(wi, sample.wo, p);
        if (sample.pdf <= 0.0)
            return false;
        sample.weight = eval(wi, sample.wo) / sample.pdf;
        return true;
    }

    public BSDFFlags lobe_types()
    {
        uint flags = uint(BSDFFlags::diffuse_reflection) | uint(BSDFFlags::glossy_reflection);
        if (base_weight < 1.0) {
            flags |= uint(BSDFFlags::diffuse_transmission) | uint(BSDFFlags::glossy_transmission);
            if (delta_transmission)
                flags |= uint(BSDFFlags::delta_transmission);
        }
        return BSDFFlags(flags);
    }
};
