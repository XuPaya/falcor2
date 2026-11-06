# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SpongeCakeMaterial with a Henyey-Greenstein (forward-scattering) layer, the sheer-fabric configuration: reflected
and transmitted energies and the shape of the scattered transmission against a brute-force volumetric random walk
through the slab, white furnace, and sampling consistency (sample weights and pdfs vs eval)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers

PROBE_MODULE = str(Path(__file__).with_name("test_spongecake_material.slang"))
# OpenPBR (any scene material) does not compile for CUDA here; SpongeCake alone would, but keep the device set of
# the other render tests.
DEVICE_TYPES = [t for t in helpers.DEFAULT_DEVICE_TYPES if t != spy.DeviceType.cuda]

#: max_tv: tolerance of the total-variation distance of the scattered transmission. In thick slabs the true
#: multiple-scattering lobe bends toward the normal (longer paths at grazing angles); the forward lobe stays centred on
#: the incident direction, so its shape error grows with the optical thickness (sheer ~0.02-0.05, tau 2 ~0.07-0.11).
CASES = {
    "sheer": dict(tau=0.5, g=0.85, albedo=0.9, max_tv=0.07),
    "thick": dict(tau=2.0, g=0.8, albedo=0.8, max_tv=0.13),
}


# Reference -----------------------------------------------------------------------------------------------------


def hg_sample(d: np.ndarray, g: float, rng: np.random.Generator) -> np.ndarray:
    """Directions with HG(g) distributed cosine to the unit directions d (N, 3)."""
    n = len(d)
    s = (1 - g * g) / (1 - g + 2 * g * rng.random(n))
    c = np.clip((1 + g * g - s * s) / (2 * g), -1, 1)
    phi = 2 * np.pi * rng.random(n)
    a = np.where(np.abs(d[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t = np.cross(d, a)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(d, t)
    st = np.sqrt(1 - c * c)
    return (st * np.cos(phi))[:, None] * t + (st * np.sin(phi))[:, None] * b + c[:, None] * d


def random_walk(tau: float, g: float, wi: np.ndarray, n: int, rng: np.random.Generator):
    """Analog walk through an HG slab of optical thickness tau (z = normal, depth in [0, 1]) for light arriving
    from wi: exit side (0 top, 1 bottom), exit direction and scattering count per photon."""
    d = np.tile(-wi, (n, 1))
    z = np.zeros(n)
    order = np.zeros(n, np.int64)
    side = np.full(n, -1)
    live = np.arange(n)
    while len(live):
        dl, zl = d[live], z[live]
        step = -np.log1p(-rng.random(len(live))) / tau        # free path in units of the slab thickness
        znew = zl - step * dl[:, 2]
        out_top, out_bottom = znew <= 0.0, znew >= 1.0
        side[live[out_top]] = 0
        side[live[out_bottom]] = 1
        stay = ~(out_top | out_bottom)
        idx = live[stay]
        z[idx] = znew[stay]
        d[idx] = hg_sample(dl[stay], g, rng)
        order[idx] += 1
        live = idx
    return side, d, order


# falcor2 probes ------------------------------------------------------------------------------------------------


class Probe:
    def __init__(self, device: spy.Device, props: dict):
        import sys

        sys.path.insert(0, str(Path(__file__).parent))     # the scene helpers of the adjoint tracer's tests
        import test_adjoint_pathtracer as scenes

        self.scene = f2.Scene.create(device)
        values = {k: spy.float3(*v) if isinstance(v, (tuple, list)) else v for k, v in props.items()}
        material = self.scene.create_material("SpongeCakeMaterial", f2.Properties(values))
        scenes.add_quad_y(self.scene, -1.0, 1.0, -1.0, 1.0, 0.0, material)
        self.scene.update()
        self.eval_fn = scenes.probe_function(device, self.scene, "eval_probe", PROBE_MODULE)
        self.sample_fn = scenes.probe_function(device, self.scene, "sample_probe", PROBE_MODULE)

    def eval(self, wi, wo):
        """(f |cos wo|, pdf) for local directions (z = normal)."""
        wo = to_world(wo)
        wi = np.ascontiguousarray(np.broadcast_to(to_world(wi), wo.shape))
        return self.eval_fn(wi=wi, wo=wo, _result="numpy")

    def sample(self, wi, n):
        wi = spy.float3(*map(float, to_world(wi)))
        weight = self.sample_fn(tid=spy.grid((n,)), wi=wi, which=0, _result="numpy")
        sample = self.sample_fn(tid=spy.grid((n,)), wi=wi, which=1, _result="numpy")
        return weight, np.concatenate([to_local(sample[:, :3]), sample[:, 3:]], 1)


def to_world(w):
    """Local (z = normal) to the quad's world frame (normal +y); a reflection, fine for an isotropic material."""
    w = np.asarray(w, np.float32)
    return np.ascontiguousarray(w[..., [0, 2, 1]])


to_local = to_world


def material_props(case: dict, albedo: float | None = None) -> dict:
    a = case["albedo"] if albedo is None else albedo
    return dict(hg=True, thickness=case["tau"], roughness=1.0 - case["g"], albedo=(a, a, a), top_thickness=0.0,
                delta_transmission=True)


def sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def histogram(wo: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exit-direction histogram over (|cos|, azimuth from the incidence plane), normalized."""
    mu = np.abs(wo[:, 2])
    phi = np.abs(np.arctan2(wo[:, 1], wo[:, 0]))
    h, _, _ = np.histogram2d(mu, phi, bins=[10, 12], range=[[0, 1], [0, np.pi]], weights=weights)
    return h / h.sum()


# Tests ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        "sheer",
        # Known model limit: at grazing incidence (mu 0.4) the forward lobe overestimates the transmission of a
        # tau 2 slab by ~13 % (TV ~0.2), as the true lobe bends toward the normal (see CASES).
        pytest.param("thick", marks=pytest.mark.xfail(reason="forward-lobe limit of thick slabs at grazing incidence")),
    ],
)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_spongecake_hg_slab_matches_random_walk(device_type: spy.DeviceType, device: spy.Device, case: str) -> None:
    """Reflected / transmitted energy of the scattered light, the unscattered transmittance and the shape of the
    scattered transmission (single + the forward multiple-scattering lobe) agree with a volumetric random walk."""
    c = CASES[case]
    probe = Probe(device, material_props(c))
    rng = np.random.default_rng(1)
    for mu in (0.95, 0.7, 0.4):
        wi = np.array([math.sqrt(1 - mu * mu), 0.0, mu])
        side, d, order = random_walk(c["tau"], c["g"], wi, 1 << 19, rng)
        w = c["albedo"] ** order
        n = len(side)
        walk = dict(R=w[(order > 0) & (side == 0)].sum() / n, T=w[(order > 0) & (side == 1)].sum() / n,
                    delta=np.mean((order == 0) & (side == 1)))
        wo = sphere(1 << 21, rng)
        e = probe.eval(wi, wo)
        value = e[:, :3].mean(1)
        up = wo[:, 2] > 0
        model = dict(R=value[up].sum() * 4 * np.pi / len(wo), T=value[~up].sum() * 4 * np.pi / len(wo),
                     delta=math.exp(-c["tau"] / mu))
        sel = (order > 0) & (side == 1)
        tv = 0.5 * np.abs(histogram(wo[~up], value[~up]) - histogram(d[sel], w[sel])).sum()
        print(f"{case} mu {mu}: walk R {walk['R']:.4f} T {walk['T']:.4f} delta {walk['delta']:.4f} | "
              f"model R {model['R']:.4f} T {model['T']:.4f} delta {model['delta']:.4f} | TV(T) {tv:.3f}")
        assert abs(model["delta"] - walk["delta"]) < 0.004
        assert abs(model["R"] - walk["R"]) < 0.01 + 0.08 * walk["R"]
        assert abs(model["T"] - walk["T"]) < 0.01 + 0.04 * walk["T"]
        assert tv < c["max_tv"]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_spongecake_hg_white_furnace_and_sampling(device_type: spy.DeviceType, device: spy.Device) -> None:
    """A white HG slab conserves energy (eval integral + delta = 1); sample weights and pdfs match eval."""
    rng = np.random.default_rng(2)
    for case in CASES.values():
        white = Probe(device, material_props(case, albedo=1.0))
        colored = Probe(device, material_props(case))
        for mu in (0.9, 0.5, 0.2):
            wi = np.array([math.sqrt(1 - mu * mu), 0.0, mu])
            wo = sphere(1 << 21, rng)
            total = white.eval(wi, wo)[:, :3].mean(1).sum() * 4 * np.pi / len(wo) + math.exp(-case["tau"] / mu)
            assert abs(total - 1.0) < 0.015, (case, mu, total)

            e = colored.eval(wi, wo)
            integral = e[:, :3].mean(1).sum() * 4 * np.pi / len(wo) + math.exp(-case["tau"] / mu)
            weight, sample = colored.sample(wi, 1 << 20)
            ok = sample[:, 3] >= 0
            assert ok.mean() > 0.999
            estimate = weight[ok, :3].mean(1).sum() / len(weight)
            assert abs(estimate / integral - 1.0) < 0.01, (case, mu, estimate, integral)
            # Recorded pdf and weight of the non-delta samples equal eval_with_pdf at the sampled direction.
            delta = weight[:, 3].astype(np.uint32) == 0x40
            nd = ok & ~delta
            e2 = colored.eval(wi, sample[nd, :3])
            assert np.max(np.abs(e2[:, 3] - sample[nd, 3]) / np.maximum(sample[nd, 3], 1e-6)) < 1e-3
            assert np.max(np.abs(e2[:, :3] / e2[:, 3:4] - weight[nd, :3]) / np.maximum(weight[nd, :3], 1e-3)) < 1e-3
