// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spongecake_material.h"

#include "falcor2/utils/buffer_handle.h"

#include "falcor2/render/scene.h"

#include <algorithm>
#include <cmath>

namespace falcor {

namespace {

// Table layout, see slang/falcor2/render/bsdfs/spongecake_bsdf.slang.
constexpr int TABLE_MU = 32;
constexpr int TABLE_GAMMA = 32;
constexpr int TAB_E = 0;
constexpr int TAB_DE = 128;
constexpr int TAB_MS = 640;
constexpr int TAB_TBAR = 704;
constexpr int TAB_EBAR = 709;
constexpr int TAB_DEBAR = 713;
constexpr int TAB_WR = 729;
constexpr int TAB_DMS = 732;
constexpr int TAB_DWR = 988;
constexpr int TABLE_SHARE = 16;
constexpr int TAB_SHARE = 1000;
constexpr int TAB_LOBE = 1016;
constexpr int TAB_MSAVG = 1020;
constexpr int TAB_DSHARE = 1022;
constexpr int TAB_DLOBE = 1086;
constexpr int TAB_DMSAVG = 1102;
constexpr int TAB_SIZE = 1110;
constexpr double PI = 3.14159265358979323846;

struct V3 {
    double x, y, z;
};
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(double s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 normalize(V3 a) { return (1.0 / std::sqrt(std::max(dot(a, a), 1e-300))) * a; }

/// One layer, as SpongeCakeLayer in spongecake_bsdf.slang: SGGX microflakes (surface-like or fibers) or a
/// Henyey-Greenstein medium with g = 1 - alpha.
struct Layer {
    double tau;
    double alpha;
    bool fiber;
    bool hg;

    double c1() const { return fiber ? 1.0 : alpha * alpha; }
    double c2() const { return fiber ? alpha * alpha - 1.0 : 1.0 - alpha * alpha; }
    double along(V3 w) const { return fiber ? w.x : w.z; }
    double sigma(V3 w) const
    {
        if (hg)
            return 1.0;
        double d = along(w);
        return std::sqrt(std::max(c1() + c2() * d * d, 1e-12));
    }
    double lambda(V3 w) const { return sigma(w) / std::max(std::abs(w.z), 1e-4); }

    /// Visible normal seen from wi (Heitz et al. 2015).
    V3 sample_visible_normal(V3 wi, double u1, double u2) const
    {
        double sign = wi.z < 0.0 ? -1.0 : 1.0;
        double a = -1.0 / (sign + wi.z), b = wi.x * wi.y * a;
        V3 wk{1.0 + sign * wi.x * wi.x * a, sign * b, -sign * wi.x};
        V3 wj{b, sign + wi.y * wi.y * a, -wi.y};
        V3 axis = fiber ? V3{1.0, 0.0, 0.0} : V3{0.0, 0.0, 1.0};
        double cx = c1(), cy = c2();
        double ak = dot(axis, wk), aj = dot(axis, wj), ai = dot(axis, wi);
        double s_kk = cx + cy * ak * ak, s_jj = cx + cy * aj * aj, s_ii = cx + cy * ai * ai;
        double s_kj = cy * ak * aj, s_ki = cy * ak * ai, s_ji = cy * aj * ai;
        double det = std::abs(
            s_kk * s_jj * s_ii - s_kj * s_kj * s_ii - s_ki * s_ki * s_jj - s_ji * s_ji * s_kk + 2.0 * s_kj * s_ki * s_ji
        );
        double inv_sqrt_s_ii = 1.0 / std::sqrt(s_ii);
        double tmp = std::sqrt(std::max(s_jj * s_ii - s_ji * s_ji, 1e-20));
        V3 m_k{std::sqrt(det) / tmp, 0.0, 0.0};
        V3 m_j{-inv_sqrt_s_ii * (s_ki * s_ji - s_kj * s_ii) / tmp, inv_sqrt_s_ii * tmp, 0.0};
        V3 m_i{inv_sqrt_s_ii * s_ki, inv_sqrt_s_ii * s_ji, inv_sqrt_s_ii * s_ii};
        double r = std::sqrt(u1), phi = 2.0 * PI * u2;
        double x = r * std::cos(phi), y = r * std::sin(phi);
        V3 m = normalize(x * m_k + y * m_j + std::sqrt(std::max(1.0 - x * x - y * y, 0.0)) * m_i);
        return m.x * wk + m.y * wj + m.z * wi;
    }

    /// Scattered direction (pointing away) of light arriving from wi, sampled from the phase function.
    V3 sample_direction(V3 wi, double u1, double u2) const
    {
        if (hg)
            return sample_hg(-1.0 * wi, 1.0 - alpha, u1, u2);
        V3 m = sample_visible_normal(wi, u1, u2);
        return 2.0 * dot(wi, m) * m - wi;
    }

    /// Direction around the travel direction d with HG(g) distributed cosine.
    static V3 sample_hg(V3 d, double g, double u1, double u2)
    {
        double c;
        if (std::abs(g) < 1e-3) {
            c = 1.0 - 2.0 * u1;
        } else {
            double s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
            c = std::clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
        }
        double sign = d.z < 0.0 ? -1.0 : 1.0;
        double a = -1.0 / (sign + d.z), b = d.x * d.y * a;
        V3 t{1.0 + sign * d.x * d.x * a, sign * b, -sign * d.x};
        V3 bt{b, sign + d.y * d.y * a, -d.y};
        double st = std::sqrt(std::max(1.0 - c * c, 0.0)), phi = 2.0 * PI * u2;
        return normalize((st * std::cos(phi)) * t + (st * std::sin(phi)) * bt + c * d);
    }
};

double hg_phase(double cos_theta, double g)
{
    double d = 1.0 + g * g - 2.0 * g * cos_theta;
    return (1.0 - g * g) / (4.0 * PI * d * std::sqrt(d));
}

/// HG lobe about the (unit) direction c folded into the hemisphere of wo: integrates to 1 over that hemisphere.
double folded_hg(V3 c, V3 wo, double g)
{
    return hg_phase(dot(c, wo), g) + hg_phase(dot(c, V3{wo.x, wo.y, -wo.z}), g);
}

double phi_fn(double x)
{
    return x < 1e-3 ? 1.0 - x * (0.5 - x / 6.0) : -std::expm1(-x) / x;
}

/// Attenuated depth integral G of layer k (0: top, 1: body) for wi (top) and wo, as in SpongeCakeBSDF::eval().
double layer_g(const std::array<Layer, 2>& l, int k, V3 wi, V3 wo)
{
    double li0 = l[0].lambda(wi), lo0 = l[0].lambda(wo), li1 = l[1].lambda(wi), lo1 = l[1].lambda(wo);
    if (wo.z > 0.0) {
        if (k == 0)
            return l[0].tau * phi_fn(l[0].tau * (li0 + lo0));
        return std::exp(-l[0].tau * (li0 + lo0)) * l[1].tau * phi_fn(l[1].tau * (li1 + lo1));
    }
    if (k == 0)
        return std::exp(-l[1].tau * lo1 - l[0].tau * std::min(li0, lo0)) * l[0].tau
            * phi_fn(l[0].tau * std::abs(li0 - lo0));
    return std::exp(-l[0].tau * li0 - l[1].tau * std::min(li1, lo1)) * l[1].tau
        * phi_fn(l[1].tau * std::abs(li1 - lo1));
}

/// White single-scattering albedos E[layer][R/T][mu] (azimuth averaged), their cosine-weighted averages and the
/// cosine-weighted average of the unscattered transmittance. E = integral of f |cos o| over wo, sampled from the
/// layer's phase function D(h) / (4 sigma(wi)), whose weight is lambda(wi) G (stratified 16 x 16 per direction).
void single_scattering(const std::array<Layer, 2>& l, float* e, float* e_avg, float* t_avg)
{
    int n_phi = (l[0].fiber || l[1].fiber) ? 16 : 1;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < TABLE_MU; ++j) {
            double sum[2] = {0.0, 0.0};
            if (l[k].tau > 0.0) {
                double mu = (j + 0.5) / TABLE_MU, st = std::sqrt(1.0 - mu * mu);
                for (int p = 0; p < n_phi; ++p) {
                    double phi = (p + 0.5) / n_phi * 2.0 * PI;
                    V3 wi{st * std::cos(phi), st * std::sin(phi), mu};
                    for (int a = 0; a < 16; ++a) {
                        for (int b = 0; b < 16; ++b) {
                            V3 wo = l[k].sample_direction(wi, (a + 0.5) / 16.0, (b + 0.5) / 16.0);
                            if (std::abs(wo.z) < 1e-6)
                                continue;
                            sum[wo.z > 0.0 ? 0 : 1] += l[k].lambda(wi) * layer_g(l, k, wi, wo);
                        }
                    }
                }
            }
            for (int rt = 0; rt < 2; ++rt)
                e[(k * 2 + rt) * TABLE_MU + j] = float(sum[rt] / (n_phi * 256));
        }
    }
    for (int i = 0; i < 4; ++i) {
        double avg = 0.0;
        for (int j = 0; j < TABLE_MU; ++j)
            avg += 2.0 * (j + 0.5) / TABLE_MU * e[i * TABLE_MU + j];
        e_avg[i] = float(avg / TABLE_MU);
    }
    double t = 0.0;
    for (int j = 0; j < 64; ++j) {
        double mu = (j + 0.5) / 64.0, st = std::sqrt(1.0 - mu * mu);
        for (int p = 0; p < n_phi; ++p) {
            double phi = (p + 0.5) / n_phi * 2.0 * PI;
            V3 w{st * std::cos(phi), st * std::sin(phi), mu};
            t += 2.0 * mu * std::exp(-l[0].tau * l[0].lambda(w) - l[1].tau * l[1].lambda(w)) / (64.0 * n_phi);
        }
    }
    *t_avg = float(t);
}

uint32_t pcg_hash(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Counter-based random numbers in [0, 1): each photon has its own stream, so the walks of nearby parameters
/// share their random numbers (common random numbers for the finite differences).
struct PhotonRng {
    uint32_t photon;
    uint32_t counter = 0;
    double next() { return pcg_hash(pcg_hash(photon) + counter++) * (1.0 / 4294967296.0); }
};

struct WalkResult {
    double reflected_share;
    double event_share[2];
    double scale[2][TABLE_GAMMA];
    /// Share of the multiply scattered light leaving at the top per incident |cos| (16 bins).
    double share[TABLE_SHARE];
    /// Forward lobes of the multiply scattered light: beta_R, g_R, beta_T, g_T.
    double lobe[4];
    /// Cosine-weighted averages of m_R(mu) = share(mu) m(mu) and m_T(mu) = (1 - share(mu)) m(mu).
    double ms_avg[2];
};

/// Linear interpolation of a table over mu at the bin centers (j + 0.5) / n, as in spongecake_bsdf.slang.
double interp_mu(const double* table, int n, double mu)
{
    double x = std::clamp(mu * n - 0.5, 0.0, double(n - 1));
    int i = std::min(int(x), n - 2);
    double f = x - i;
    return table[i] * (1.0 - f) + table[i + 1] * f;
}

/// Energy m(mu) a white slab scatters more than once at the table directions: 1 - sum of E - t (azimuth averaged).
std::array<double, TABLE_MU> multiple_mu(const std::array<Layer, 2>& l, const float* e)
{
    int n_phi = (l[0].fiber || l[1].fiber) ? 16 : 1;
    std::array<double, TABLE_MU> m{};
    for (int j = 0; j < TABLE_MU; ++j) {
        double mu = (j + 0.5) / TABLE_MU, st = std::sqrt(1.0 - mu * mu), t = 0.0;
        for (int p = 0; p < n_phi; ++p) {
            double phi = (p + 0.5) / n_phi * 2.0 * PI;
            V3 w{st * std::cos(phi), st * std::sin(phi), mu};
            t += std::exp(-l[0].tau * l[0].lambda(w) - l[1].tau * l[1].lambda(w)) / n_phi;
        }
        double sum = 0.0;
        for (int i = 0; i < 4; ++i)
            sum += e[i * TABLE_MU + j];
        m[j] = std::max(1.0 - sum - t, 0.0);
    }
    return m;
}

/// A multiply scattered photon of the walk.
struct MsPhoton {
    V3 wi;
    V3 wo;
    double weight;
};

/// Weighted maximum-likelihood fit of q(wo) = beta F_g(wo; c) + (1 - beta) K(wo) to the photons: F_g the HG lobe of
/// anisotropy g about c (the travel direction -wi for transmission, its mirror image for reflection) folded into the
/// exit hemisphere, K the diffuse (Kulla-Conty-shaped) lobe of the material. EM for beta, golden section for g.
void fit_lobe(const std::vector<MsPhoton>& photons, bool transmission, const std::vector<double>& k, double* lobe)
{
    lobe[0] = 0.0;
    lobe[1] = 0.5;
    if (photons.size() < 32)
        return;
    std::vector<V3> c(photons.size());
    for (size_t i = 0; i < photons.size(); ++i) {
        V3 wi = photons[i].wi;
        c[i] = transmission ? -1.0 * wi : V3{-wi.x, -wi.y, wi.z};
    }
    std::vector<double> f(photons.size());
    auto likelihood = [&](double g, double& beta)
    {
        for (size_t i = 0; i < photons.size(); ++i)
            f[i] = folded_hg(c[i], photons[i].wo, g);
        beta = 0.5;
        for (int it = 0; it < 12; ++it) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < photons.size(); ++i) {
                double q = beta * f[i] + (1.0 - beta) * k[i];
                num += photons[i].weight * beta * f[i] / std::max(q, 1e-30);
                den += photons[i].weight;
            }
            beta = num / den;
        }
        double ll = 0.0;
        for (size_t i = 0; i < photons.size(); ++i)
            ll += photons[i].weight * std::log(std::max(beta * f[i] + (1.0 - beta) * k[i], 1e-30));
        return ll;
    };
    // Golden section over the whole range (a fixed bracket: the fit is a continuous function of the photons, so
    // the finite differences of the tables with common random numbers stay smooth).
    constexpr double R = 0.6180339887498949;
    double a = 0.0, b = 0.98;
    double x1 = b - R * (b - a), x2 = a + R * (b - a), beta1, beta2;
    double l1 = likelihood(x1, beta1), l2 = likelihood(x2, beta2);
    for (int it = 0; it < 24; ++it) {
        if (l1 > l2) {
            b = x2;
            x2 = x1;
            l2 = l1;
            x1 = b - R * (b - a);
            l1 = likelihood(x1, beta1);
        } else {
            a = x1;
            x1 = x2;
            l1 = l2;
            x2 = a + R * (b - a);
            l2 = likelihood(x2, beta2);
        }
    }
    double g = 0.5 * (a + b), beta;
    likelihood(g, beta);
    lobe[0] = std::clamp(beta, 0.0, 1.0);
    lobe[1] = g;
}

/// White random walk through the slab (layer k occupies depth [k, k + 1] with density tau_k), incidence stratified
/// in |cos| (uniform; cosine-weighted statistics weight the photons by 2 mu): the share w_R of the multiply scattered
/// light (scattering order n >= 2) leaving at the top, overall and per incident |cos|, E[gamma^n | n >= 2, side] on
/// the albedo grid, the layers' shares of the multiple scattering events and the forward lobes fitted to the exit
/// directions. e: the single-scattering tables of the slab.
WalkResult random_walk(const std::array<Layer, 2>& l, const float* e)
{
    constexpr int PHOTONS = 16384;
    constexpr int MAX_ORDER = 2000;
    constexpr size_t MAX_FIT = 4096;
    std::vector<int> orders[2];
    std::vector<double> order_weights[2];
    std::vector<MsPhoton> exits[2];
    double events[2] = {0.0, 0.0}, side_weight[2] = {0.0, 0.0};
    int bin_count[TABLE_SHARE][2] = {};
    for (int i = 0; i < PHOTONS; ++i) {
        PhotonRng rng{uint32_t(i)};
        double mu = (i + rng.next()) / PHOTONS, phi = 2.0 * PI * rng.next(), st = std::sqrt(1.0 - mu * mu);
        V3 wi{st * std::cos(phi), st * std::sin(phi), mu};
        V3 d = -1.0 * wi;
        double z = 0.0;
        int order = 0, side = -1;
        double photon_events[2] = {0.0, 0.0};
        for (int flight = 0; side < 0 && order < MAX_ORDER && flight < 4 * MAX_ORDER; ++flight) {
            double s = -std::log(1.0 - rng.next());
            double dz = -d.z;
            int scatter = -1;
            for (int step = 0; step < 8 && scatter < 0 && side < 0; ++step) {
                int k = std::clamp(int(std::floor(z + (dz > 0.0 ? 1e-9 : -1e-9))), 0, 1);
                double ext = l[k].tau * l[k].sigma(d);
                double to_boundary = dz > 0.0 ? (k + 1 - z) / dz : (dz < 0.0 ? (k - z) / dz : 1e300);
                if (s < ext * to_boundary) {
                    z += s / ext * dz;
                    scatter = k;
                } else {
                    s -= ext * to_boundary;
                    z += to_boundary * dz;
                    if (z <= 1e-9 && dz < 0.0)
                        side = 0;
                    else if (z >= 2.0 - 1e-9 && dz > 0.0)
                        side = 1;
                    else
                        z += (dz > 0.0 ? 2e-9 : -2e-9);
                }
            }
            if (scatter < 0)
                continue;
            double v1 = rng.next(), v2 = rng.next();
            d = l[scatter].sample_direction(-1.0 * d, v1, v2);
            photon_events[scatter] += 1.0;
            order++;
        }
        if (side < 0)
            side = 0;
        if (order >= 2) {
            double w = 2.0 * mu;
            orders[side].push_back(order);
            order_weights[side].push_back(w);
            exits[side].push_back({wi, d, w});
            events[0] += w * photon_events[0];
            events[1] += w * photon_events[1];
            side_weight[side] += w;
            bin_count[std::min(i * TABLE_SHARE / PHOTONS, TABLE_SHARE - 1)][side]++;
        }
    }
    WalkResult result{};
    double ms = side_weight[0] + side_weight[1];
    result.reflected_share = ms > 0.0 ? side_weight[0] / ms : 0.5;
    double total = events[0] + events[1];
    result.event_share[0] = total > 0.0 ? events[0] / total : 0.5;
    result.event_share[1] = total > 0.0 ? events[1] / total : 0.5;
    for (int rt = 0; rt < 2; ++rt) {
        for (int g = 0; g < TABLE_GAMMA; ++g) {
            double u = double(g) / (TABLE_GAMMA - 1), gamma = 1.0 - (1.0 - u) * (1.0 - u);
            double sum = 0.0;
            for (size_t j = 0; j < orders[rt].size(); ++j)
                sum += order_weights[rt][j] * std::pow(gamma, orders[rt][j]);
            result.scale[rt][g] = side_weight[rt] > 0.0 ? sum / side_weight[rt] : 0.0;
        }
    }
    // w_R(mu): weighted least-squares quadratic in mu through the bins' shares (the bins' counts are noisy, and the
    // finite differences of the tables need a smooth function of the parameters).
    double ata[3][3] = {}, atb[3] = {};
    for (int b = 0; b < TABLE_SHARE; ++b) {
        int n = bin_count[b][0] + bin_count[b][1];
        if (n == 0)
            continue;
        double mu = (b + 0.5) / TABLE_SHARE, basis[3] = {1.0, mu, mu * mu}, y = double(bin_count[b][0]) / n;
        for (int r = 0; r < 3; ++r) {
            atb[r] += n * basis[r] * y;
            for (int c = 0; c < 3; ++c)
                ata[r][c] += n * basis[r] * basis[c];
        }
    }
    double coef[3] = {result.reflected_share, 0.0, 0.0};
    double det = ata[0][0] * (ata[1][1] * ata[2][2] - ata[1][2] * ata[2][1])
        - ata[0][1] * (ata[1][0] * ata[2][2] - ata[1][2] * ata[2][0])
        + ata[0][2] * (ata[1][0] * ata[2][1] - ata[1][1] * ata[2][0]);
    if (std::abs(det) > 1e-9 * std::pow(ata[0][0], 3)) {
        for (int k = 0; k < 3; ++k) {
            double m[3][3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    m[r][c] = c == k ? atb[r] : ata[r][c];
            coef[k] = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                       + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
                / det;
        }
    }
    for (int b = 0; b < TABLE_SHARE; ++b) {
        double mu = (b + 0.5) / TABLE_SHARE;
        result.share[b] = std::clamp(coef[0] + coef[1] * mu + coef[2] * mu * mu, 0.0, 1.0);
    }

    // The diffuse lobes K_side(wo) = m_side(mu_o) mu_o / (pi A_side), m_side as evaluated by the BSDF.
    std::array<double, TABLE_MU> m = multiple_mu(l, e);
    for (int rt = 0; rt < 2; ++rt) {
        double avg = 0.0;
        for (int j = 0; j < TABLE_MU; ++j) {
            double mu = (j + 0.5) / TABLE_MU, w_r = interp_mu(result.share, TABLE_SHARE, mu);
            avg += 2.0 * mu * (rt == 0 ? w_r : 1.0 - w_r) * m[j] / TABLE_MU;
        }
        result.ms_avg[rt] = avg;
    }
    for (int rt = 0; rt < 2; ++rt) {
        size_t stride = std::max<size_t>(1, exits[rt].size() / MAX_FIT);
        std::vector<MsPhoton> fit;
        std::vector<double> k;
        for (size_t j = 0; j < exits[rt].size(); j += stride) {
            const MsPhoton& photon = exits[rt][j];
            double mu = std::abs(photon.wo.z), w_r = interp_mu(result.share, TABLE_SHARE, mu);
            double m_side = (rt == 0 ? w_r : 1.0 - w_r) * interp_mu(m.data(), TABLE_MU, mu);
            fit.push_back(photon);
            k.push_back(result.ms_avg[rt] > 0.0 ? m_side * mu / (PI * result.ms_avg[rt]) : mu / PI);
        }
        fit_lobe(fit, rt == 1, k, &result.lobe[2 * rt]);
    }
    return result;
}

/// The walk's table entries: w_R and the layers' shares (3), the multiple scattering scales (2 x 32), the reflected
/// share per |cos| (16), the forward lobes (4) and the multiple scattering averages (2).
constexpr int WALK_ENTRIES = 89;
std::array<double, WALK_ENTRIES> walk_entries(const WalkResult& w)
{
    std::array<double, WALK_ENTRIES> v{};
    v[0] = w.reflected_share;
    v[1] = w.event_share[0];
    v[2] = w.event_share[1];
    for (int i = 0; i < 2 * TABLE_GAMMA; ++i)
        v[3 + i] = w.scale[i / TABLE_GAMMA][i % TABLE_GAMMA];
    for (int i = 0; i < TABLE_SHARE; ++i)
        v[67 + i] = w.share[i];
    for (int i = 0; i < 4; ++i)
        v[83 + i] = w.lobe[i];
    v[87] = w.ms_avg[0];
    v[88] = w.ms_avg[1];
    return v;
}

/// All tables for (top thickness, top roughness, body thickness, body roughness) and the layer types (top fiber,
/// top HG, body fiber, body HG), with their derivatives by central differences.
std::vector<float> compute_tables(std::array<double, 4> params, std::array<bool, 4> types)
{
    std::vector<float> tab(TAB_SIZE, 0.f);
    auto layers = [&](const std::array<double, 4>& p)
    {
        return std::array<Layer, 2>{Layer{p[0], p[1], types[0], types[1]}, Layer{p[2], p[3], types[2], types[3]}};
    };
    std::array<Layer, 2> l = layers(params);
    single_scattering(l, &tab[TAB_E], &tab[TAB_EBAR], &tab[TAB_TBAR]);
    for (int p = 0; p < 4; ++p) {
        double v = params[p], lo, hi;
        if (p % 2 == 0) {
            double h = 1e-3 * std::max(v, 0.1);
            lo = std::max(v - h, 0.0);
            hi = v + h;
        } else {
            lo = std::max(v - 2e-3, 0.01);
            hi = std::min(v + 2e-3, 1.0);
        }
        std::array<double, 4> a = params, b = params;
        a[p] = lo;
        b[p] = hi;
        float ea[128], eb[128], ea_avg[4], eb_avg[4], ta, tb;
        std::array<Layer, 2> la = layers(a), lb = layers(b);
        single_scattering(la, ea, ea_avg, &ta);
        single_scattering(lb, eb, eb_avg, &tb);
        float inv = float(1.0 / (hi - lo));
        for (int i = 0; i < 128; ++i)
            tab[TAB_DE + p * 128 + i] = (eb[i] - ea[i]) * inv;
        for (int i = 0; i < 4; ++i)
            tab[TAB_DEBAR + p * 4 + i] = (eb_avg[i] - ea_avg[i]) * inv;
        tab[TAB_TBAR + 1 + p] = (tb - ta) * inv;
    }

    // Values (p < 0) or derivatives along parameter p of the walk's entries.
    auto store = [&](const std::array<double, WALK_ENTRIES>& v, int p)
    {
        int wr = p < 0 ? TAB_WR : TAB_DWR + 3 * p;
        int ms = p < 0 ? TAB_MS : TAB_DMS + 2 * TABLE_GAMMA * p;
        int share = p < 0 ? TAB_SHARE : TAB_DSHARE + TABLE_SHARE * p;
        int lobe = p < 0 ? TAB_LOBE : TAB_DLOBE + 4 * p;
        int avg = p < 0 ? TAB_MSAVG : TAB_DMSAVG + 2 * p;
        for (int i = 0; i < 3; ++i)
            tab[wr + i] = float(v[i]);
        for (int i = 0; i < 2 * TABLE_GAMMA; ++i)
            tab[ms + i] = float(v[3 + i]);
        for (int i = 0; i < TABLE_SHARE; ++i)
            tab[share + i] = float(v[67 + i]);
        for (int i = 0; i < 4; ++i)
            tab[lobe + i] = float(v[83 + i]);
        tab[avg] = float(v[87]);
        tab[avg + 1] = float(v[88]);
    };
    store(walk_entries(random_walk(l, &tab[TAB_E])), -1);
    // The walk's outputs vary less smoothly (common random numbers; the noise of the differences falls as
    // 1 / sqrt(photons x step)): wide steps.
    for (int p = 0; p < 4; ++p) {
        double v = params[p], lo, hi;
        if (p % 2 == 0) {
            double h = 0.15 * std::max(v, 0.3);
            lo = std::max(v - h, 0.0);
            hi = v + h;
        } else {
            // Relative: HG lobes (g = 1 - roughness) and glossy flakes are strongly nonlinear at low roughness.
            double h = std::clamp(0.3 * v, 0.01, 0.08);
            lo = std::max(v - h, 0.01);
            hi = std::min(v + h, 1.0);
        }
        std::array<double, 4> a = params, b = params;
        a[p] = lo;
        b[p] = hi;
        std::array<Layer, 2> la = layers(a), lb = layers(b);
        float ea[128], eb[128], avg[4], t;
        single_scattering(la, ea, avg, &t);
        single_scattering(lb, eb, avg, &t);
        std::array<double, WALK_ENTRIES> va = walk_entries(random_walk(la, ea));
        std::array<double, WALK_ENTRIES> vb = walk_entries(random_walk(lb, eb));
        for (int i = 0; i < WALK_ENTRIES; ++i)
            vb[i] = (vb[i] - va[i]) / (hi - lo);
        store(vb, p);
    }
    return tab;
}

} // namespace

SpongeCakeMaterial::SpongeCakeMaterial()
{
    set_slang_type_name("SpongeCakeMaterial");
}

SpongeCakeMaterial::~SpongeCakeMaterial() { }

void SpongeCakeMaterial::on_load_resources()
{
    auto load = [this](const ref<sgl::Texture>& texture, const std::filesystem::path& path, bool srgb) -> TextureHandle
    {
        if (texture)
            return m_scene->texture_manager()->register_texture({.texture = texture});
        if (!path.empty())
            return m_scene->texture_manager()->load_texture({.path = path, .srgb = srgb, .load_deferred = true});
        return {};
    };
    m_albedo_texture_handle = load(m_albedo_texture, m_albedo_texture_path, true);
    m_top_albedo_texture_handle = load(m_top_albedo_texture, m_top_albedo_texture_path, true);
    m_base_color_texture_handle = load(m_base_color_texture, m_base_color_texture_path, true);
    m_normal_texture_handle = load(m_normal_texture, m_normal_texture_path, false);
}

void SpongeCakeMaterial::update(SceneUpdateContext& ctx)
{
    float top_roughness = std::clamp(m_top_roughness, 0.01f, 1.f), roughness = std::clamp(m_roughness, 0.01f, 1.f);
    float top_thickness = std::max(m_top_thickness, 0.f), thickness = std::max(m_thickness, 0.f);
    std::array<float, 6> key{
        top_thickness,
        top_roughness,
        m_top_hg ? 2.f : (m_top_fiber ? 1.f : 0.f),
        thickness,
        roughness,
        m_hg ? 2.f : (m_fiber ? 1.f : 0.f),
    };
    if (key == m_table_key && m_tables_buffer)
        return;
    m_table_key = key;
    m_tables = compute_tables(
        {top_thickness, top_roughness, thickness, roughness},
        {m_top_fiber && !m_top_hg, m_top_hg, m_fiber && !m_hg, m_hg}
    );
    if (!m_tables_buffer) {
        m_tables_buffer = ctx.device()->create_buffer({
            .size = m_tables.size() * sizeof(float),
            .usage = sgl::BufferUsage::shader_resource,
            .label = fmt::format("{}::tables", m_name),
        });
    }
    ctx.command_encoder()->upload_buffer_data(m_tables_buffer, 0, m_tables.size() * sizeof(float), m_tables.data());
}

template<typename CursorT>
void SpongeCakeMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["tables"] = to_handle(m_tables_buffer);
    cursor["albedo_texture"] = m_albedo_texture_handle;
    cursor["top_albedo_texture"] = m_top_albedo_texture_handle;
    cursor["base_color_texture"] = m_base_color_texture_handle;
    cursor["normal_texture"] = m_normal_texture_handle;
    cursor["albedo"] = m_albedo;
    cursor["thickness"] = std::max(m_thickness, 0.f);
    cursor["top_albedo"] = m_top_albedo;
    cursor["top_thickness"] = std::max(m_top_thickness, 0.f);
    cursor["base_color"] = m_base_color;
    cursor["base_weight"] = std::clamp(m_base_weight, 0.f, 1.f);
    cursor["roughness"] = std::clamp(m_roughness, 0.01f, 1.f);
    cursor["top_roughness"] = std::clamp(m_top_roughness, 0.01f, 1.f);
    cursor["normal_texture_scale"] = m_normal_texture_scale;
    // Bits 3 / 4: HG body / top; bit 5: the tables have the forward multiple scattering lobes.
    cursor["layer_flags"] = (m_fiber && !m_hg ? 1u : 0u) | (m_top_fiber && !m_top_hg ? 2u : 0u)
        | (m_delta_transmission ? 4u : 0u) | (m_hg ? 8u : 0u) | (m_top_hg ? 16u : 0u) | 32u;
}

template void SpongeCakeMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void SpongeCakeMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(SpongeCakeMaterial);

} // namespace falcor
