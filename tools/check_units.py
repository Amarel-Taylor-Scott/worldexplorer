#!/usr/bin/env python3
"""Per-BRANCH golden gate for the registry refactors. The e2e decision gate
only exercises the branches its tiny smoke happens to hit; this drives EVERY
skill kind (fit+predict), EVERY ranker family and EVERY transform once on a
fixed synthetic fixture and hashes the outputs. A refactor must reproduce the
goldens byte-for-byte (same machine, same env).

Usage:
  python tools/check_units.py --save  <engine.py> <goldens.json>
  python tools/check_units.py --check <engine.py> <goldens.json>
"""
import sys, importlib.util, json, hashlib
from pathlib import Path

import numpy as np


def load_engine(path: str):
    spec = importlib.util.spec_from_file_location("eng_units", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["eng_units"] = m
    spec.loader.exec_module(m)
    return m


def fixture(m):
    """Deterministic synthetic world: anon X-block + market-ish columns, mild
    signal, time-ordered segments. Small enough that every skill fits fast."""
    rng = np.random.default_rng(20260611)
    n, d_anon = 2400, 30
    Xa = rng.normal(size=(n, d_anon)).astype(np.float32)
    drift = np.linspace(0, 1.5, n, dtype=np.float32)[:, None]
    Xa[:, :6] += drift                                    # time-ordered structure
    market = rng.normal(size=(n, 6)).astype(np.float32)
    X = np.ascontiguousarray(np.hstack([market, Xa]).astype(np.float32))
    cols = [f"mkt_{j}" for j in range(6)] + [f"X{j+1}" for j in range(d_anon)]
    beta = np.zeros(X.shape[1], np.float32)
    beta[6:14] = np.linspace(0.6, 0.1, 8, dtype=np.float32)
    y = (X @ beta + 0.8 * rng.normal(size=n)).astype(np.float32)
    seg = (np.arange(n) * 8 // n).astype(np.int32)
    return X, y, seg, cols


def hash_arr(a) -> str:
    a = np.ascontiguousarray(np.asarray(a, np.float32))
    return f"{a.shape}|{hashlib.sha1(a.tobytes()).hexdigest()[:16]}|mean={float(np.nanmean(a)):.6g}"


def collect(m) -> dict:
    X, y, seg, cols = fixture(m)
    c = m.CFG
    c.SEED = 42
    c.MLP_MAX_ITER = 2; c.MLP_MAX_ROWS = 2000; c.STABSEL_BOOT = 5
    c.GBDT_ESTIMATORS = min(getattr(c, "GBDT_ESTIMATORS", 60), 60)
    # fit the target-free organs so terrain/weather/pressure/invariant/irm/
    # compass/watershed/router exercise their REAL branch, not the fallback
    m.ATLAS = m.TerrainAtlas(c.TERRAIN_CLUSTERS, 42).fit(X, cols, c.TERRAIN_FIT_ROWS)
    m.GAUGE = m.WeatherGauge(c.WEATHER_STATES).fit(X, cols)
    m.PRESSURE = m.PressureGauge(c.WEATHER_STATES).fit(X, cols)
    # seed the stigmergy channels so mycelium/periphery/red-pheromone paths bite
    m.MYCELIUM.clear(); m.MYCELIUM.update({6: 3.0, 7: 2.0, 8: 1.0, 9: 0.5})
    m.RED_MYCELIUM.clear(); m.RED_MYCELIUM.update({20: 2.0, 21: 1.0})
    m.TRAPS.clear(); m.TRAPS.update({22})
    hold = np.arange(0, len(X), 3)

    out = {"skills": {}, "rankers": {}, "transforms": {}}
    for skill in sorted(m.SKILL_REGISTRY):
        spec = m.ViewportSpec(name="top24_identity", family="top", k=24,
                              transform="identity", proj_dim=16)
        st = m.fit_skill(skill, spec, X, y, seg, cols,
                         np.random.default_rng(7), c, 123)
        p = m.predict_skill(st, X[hold])
        out["skills"][skill] = hash_arr(p)
    for fam in m.FAMILIES:
        m._RANK_CACHE.clear()
        spec = m.ViewportSpec(name=f"{fam}24_identity", family=fam, k=24,
                              transform="identity", proj_dim=16)
        ranked = m._ranked_for(spec, X, y, seg, cols)
        out["rankers"][fam] = ranked[:40]
    for tf in m.ALL_TRANSFORMS:
        m._RANK_CACHE.clear()
        spec = m.ViewportSpec(name=f"top16_{tf}", family="top", k=16,
                              transform=tf, proj_dim=8)
        idx, tfn = m.build_viewport(spec, X, y, seg, cols, np.random.default_rng(11))
        Zh = tfn(X[hold][:, idx])
        out["transforms"][tf] = {"idx": list(map(int, idx)), "Z": hash_arr(Zh)}
    return out


def main() -> int:
    mode, eng, base = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    got = collect(load_engine(eng))
    if mode == "--save":
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_text(json.dumps(got, indent=2))
        n = sum(len(v) for v in got.values())
        print(f"goldens saved -> {base} ({n} branch fingerprints)")
        return 0
    ref = json.loads(base.read_text())
    bad = []
    for sect in ("skills", "rankers", "transforms"):
        for key in sorted(set(got[sect]) | set(ref.get(sect, {}))):
            if got[sect].get(key) != ref.get(sect, {}).get(key):
                bad.append(f"{sect}.{key}: {got[sect].get(key)!r} != {ref.get(sect, {}).get(key)!r}")
    for b_ in bad:
        print("MISMATCH", b_)
    n = sum(len(v) for v in got.values())
    print(f"{'ALL ' + str(n) + ' BRANCHES EQUIVALENT' if not bad else str(len(bad)) + ' MISMATCHES'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
