"""Balanced-benchmark calibration (CLAUDE.md D41, D43).

Tunes ONLY the productivity table A and each country's consumption shares, each within +/-30% of the
CLAUDE.md §4.3 values (shares re-normalised to sum to 1 and kept inside the band). Objective, at the
settled state after burn-in (D44, food floor off while searching):
  - every country's net exports within +/-3% of its GDP           (D41)
  - no ENERGY shortage                                             (D41)
  - the burn-in settles                                            (D44)
  - roster characters kept: DORNE is the largest net exporter of ENERGY, BRONTIA of GOODS, CERES of
    FOOD, AURELIA of SERVICES, EVERMERE of TECH
  - soft extras that help the flow test: settled prices in [0.6, 1.6], unemployment below 8%.
Search: a (1+lambda) evolution strategy on log-multipliers with the 1/5 success rule.
Then D43: food_floor_i = 0.7 x settled FOOD demand per person of country i.

    uv run python scripts/calibrate_balance.py            # search and print the result
    uv run python scripts/calibrate_balance.py --write    # also write config/countries.yaml
Then regenerate the fixture: uv run python -m world.engine.burn_in
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world.config import COUNTRIES, SECTORS, Config, load_config  # noqa: E402
from world.engine.burn_in import run_burn_in, settled_streak  # noqa: E402
from world.rng import Stream, make_rng  # noqa: E402

# CLAUDE.md §4.3 reference values (the +/-30% band is around these).
A_REF = np.array(
    [
        [0.6, 2.5, 0.6, 0.4, 0.8],  # DORNE
        [0.9, 0.4, 2.0, 0.9, 0.9],  # BRONTIA
        [2.2, 0.5, 0.6, 0.4, 0.8],  # CERES
        [1.0, 0.9, 1.1, 0.6, 0.7],  # FALKEN
        [0.5, 0.3, 0.9, 1.3, 2.2],  # AURELIA
        [0.5, 0.5, 1.0, 2.4, 1.4],  # EVERMERE
    ]
)
SHARES_DEFAULT = [0.22, 0.13, 0.25, 0.12, 0.28]
SHARES_AURELIA = [0.21, 0.12, 0.20, 0.11, 0.36]
S_REF = np.array([SHARES_AURELIA if c == "AURELIA" else SHARES_DEFAULT for c in COUNTRIES])
BAND = 0.30
CHARACTER = {
    "DORNE": "ENERGY",
    "BRONTIA": "GOODS",
    "CERES": "FOOD",
    "AURELIA": "SERVICES",
    "EVERMERE": "TECH",
}
NX_TOL = 0.03
FLOOR_SHARE = 0.7  # D43
N_A = A_REF.size


def project_shares(raw: np.ndarray) -> np.ndarray:
    """Shares within [0.7, 1.3] x reference and summing to 1 (clip / re-normalise until stable)."""
    lo, hi = S_REF * (1 - BAND), S_REF * (1 + BAND)
    s = raw / raw.sum(axis=1, keepdims=True)
    for _ in range(50):
        s = np.clip(s, lo, hi)
        s = s / s.sum(axis=1, keepdims=True)
        if np.all(s >= lo - 1e-12) and np.all(s <= hi + 1e-12):
            break
    return s


def decode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.exp(np.clip(x, np.log(1 - BAND), np.log(1 + BAND)))
    A = A_REF * m[:N_A].reshape(A_REF.shape)
    shares = project_shares(S_REF * m[N_A:].reshape(S_REF.shape))
    return A, shares


def make_cfg(base: Config, A: np.ndarray, shares: np.ndarray, floors: np.ndarray | None = None) -> Config:
    countries = []
    for i, c in enumerate(base.countries.countries):
        upd = {
            "productivity": {g: float(A[i, k]) for k, g in enumerate(SECTORS)},
            "consumption_shares": {g: float(shares[i, k]) for k, g in enumerate(SECTORS)},
            "food_floor": 0.0 if floors is None else float(floors[i]),
        }
        countries.append(c.model_copy(update=upd))
    return base.model_copy(update={"countries": base.countries.model_copy(update={"countries": countries})})


def measure(cfg: Config) -> dict:
    s, hist = run_burn_in(cfg, 0)
    b = cfg.world.burn_in
    last = hist[-3:]
    nx = np.zeros(6)
    net_by_good = np.zeros((6, 5))
    for k, cur in enumerate(last):
        prev = hist[len(hist) - 3 + k - 1]
        val = cur.trade * prev.price[None, :, :]  # traded at the price set before the step
        net_by_good += val.sum(axis=0) - val.sum(axis=1)
    net_by_good /= len(last)
    nx = net_by_good.sum(axis=1) / np.maximum(s.gdp, 1e-9)
    energy_short = s.shortage[:, 1] / np.maximum(s.demand[:, 1], 1e-9)
    return {
        "settled": settled_streak(hist, b.settle_tol) >= b.settle_streak,
        "turns": len(hist) - 1,
        "nx": nx,
        "net_by_good": net_by_good,
        "energy_short": energy_short,
        "price": (float(s.price.min()), float(s.price.max())),
        "u": s.unemployment,
        "food_pp": s.demand[:, 0] / s.population,
        "gdp": s.gdp,
    }


def characters_kept(net_by_good: np.ndarray) -> list[str]:
    bad = []
    for c, good in CHARACTER.items():
        i, g = COUNTRIES.index(c), SECTORS.index(good)
        if net_by_good[i, g] <= 0 or np.argmax(net_by_good[:, g]) != i:
            bad.append(f"{c}/{good}")
    return bad


def objective(x: np.ndarray) -> float:
    try:
        m = measure(make_cfg(load_config(), *decode(x)))
    except Exception:  # an invariant failure is just a very bad point
        return 1e6
    j = 0.0 if m["settled"] else 20.0
    j += 200 * np.sum(np.maximum(np.abs(m["nx"]) - NX_TOL * 0.8, 0.0)) + 5 * np.sum(m["nx"] ** 2)
    j += 50 * np.sum(m["energy_short"])
    j += 10 * len(characters_kept(m["net_by_good"]))
    lo, hi = m["price"]
    j += 5 * (max(0.6 - lo, 0) + max(hi - 1.6, 0))
    j += 10 * np.sum(np.maximum(m["u"] - 0.08, 0))
    return float(j)


def search(generations: int, lam: int, workers: int) -> np.ndarray:
    rng = make_rng(0, 0, Stream.BOT)  # calibration only; the engine never sees this stream
    dim = N_A + S_REF.size
    best = np.zeros(dim)
    best_j = objective(best)
    sigma = 0.08
    with ProcessPoolExecutor(workers) as ex:
        for gen in range(generations):
            cands = [best + sigma * rng.standard_normal(dim) for _ in range(lam)]
            scores = list(ex.map(objective, cands))
            k = int(np.argmin(scores))
            wins = sum(sc < best_j for sc in scores)
            if scores[k] < best_j:
                best, best_j = cands[k], scores[k]
            sigma *= 1.2 if wins / lam > 0.2 else 0.9
            sigma = float(np.clip(sigma, 0.03, 0.25))
            print(f"gen {gen:3d}  best J {best_j:9.4f}  sigma {sigma:.3f}", flush=True)
            if best_j < 1e-6:
                break
    return best


def write_yaml(A: np.ndarray, shares: np.ndarray, floors: np.ndarray) -> None:
    path = ROOT / "config" / "countries.yaml"
    text = path.read_text()
    blocks = re.split(r"(?=\n  - name: )", text)
    out = [blocks[0]]
    for blk in blocks[1:]:
        name = re.search(r"- name: (\w+)", blk).group(1)  # type: ignore[union-attr]
        i = COUNTRIES.index(name)
        fmt = lambda v: "{" + ", ".join(f"{g}: {v[k]:.5f}" for k, g in enumerate(SECTORS)) + "}"  # noqa: E731
        sh = np.round(shares[i], 5)
        sh[np.argmax(sh)] += round(1.0 - float(sh.sum()), 5)  # exact sum 1 after rounding
        blk = re.sub(r"productivity: \{[^}]*\}", f"productivity: {fmt(A[i])}", blk)
        blk = re.sub(r"\n    consumption_shares: \{[^}]*\}[^\n]*", "", blk)
        blk = re.sub(
            r"(\n    productivity: [^\n]*)",
            rf"\1  # D41 calibrated (docs/calibration.md)"
            rf"\n    consumption_shares: {fmt(sh)}  # D41 calibrated",
            blk,
        )
        blk = re.sub(r"food_floor: [0-9.e-]+", f"food_floor: {floors[i]:.5f}", blk)
        out.append(blk)
    path.write_text("".join(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--lam", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    base = load_config()
    print("reference (CLAUDE.md §4.3) objective:", objective(np.zeros(N_A + S_REF.size)))
    x = search(args.generations, args.lam, args.workers)
    A, shares = decode(x)
    m = measure(make_cfg(base, A, shares))
    floors = FLOOR_SHARE * m["food_pp"]
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print("settled:", m["settled"], "after", m["turns"], "turns")
    print("net exports % GDP:", 100 * m["nx"])
    print("energy shortage:", m["energy_short"], " prices:", m["price"], " u:", m["u"])
    print("characters broken:", characters_kept(m["net_by_good"]) or "none")
    print("A / reference:\n", A / A_REF)
    print("shares / reference:\n", shares / S_REF)
    print("A:\n", A, "\nshares:\n", shares)
    print("D43 food floors (0.7 x settled FOOD demand per person):", floors)
    if args.write:
        write_yaml(A, shares, floors)
        print("wrote config/countries.yaml")


if __name__ == "__main__":
    main()
