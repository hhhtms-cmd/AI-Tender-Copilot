"""
Huber FTL Tender Copilot - core algorithm (v2, rebuilt from scratch)
====================================================================
Scenario: Full-truckload, mostly 1 order = 1 truck (no consolidation).
Goal:     For each shipment, recommend the tender sequence that gives the
          LOWEST EXPECTED TOTAL COST (rate + risk of rejection -> spot),
          then allocate a whole batch without overloading any carrier.

Layers
  1. Data hygiene        - drop OTM placeholder rates, flag per-lane anomalies
  2. Acceptance model    - learned from tender history, with fallback
                           (lane+carrier -> carrier -> global)
  3. Expected cost       - USD, not a score
  4. Tender sequence     - best order of carriers within lead-time limit
  5. Batch allocation    - daily carrier capacity, hardest loads first

STATUS: Parameters marked [CALIBRATE] are placeholders. They will be replaced
with values learned from Huber tender history once the data arrives.
Real client data must stay inside the IBM environment - never commit to a
public repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    placeholder_rates: tuple = (999999.99, 99999.99, 0.0)   # OTM dummy values
    anomaly_iqr_k: float = 1.5            # per-lane outlier rule (Tukey)
    min_lane_obs_for_anomaly: int = 5
    prior_strength: float = 8.0           # [CALIBRATE] pseudo-tenders for shrinkage
    global_accept_prior: float = 0.75     # [CALIBRATE] used until history loaded
    spot_premium: float = 1.45            # [CALIBRATE] spot cost / contract rate
    hours_per_tender_round: float = 4.0   # [CALIBRATE] OTM tender response window
    reject_round_cost: float = 120.0      # [CALIBRATE] cost of each rejection:
                                          #   planner rework + lost lead time + service risk
    spot_escalation_per_round: float = 0.08  # [CALIBRATE] spot gets pricier as pickup nears
    max_sequence_len: int = 4
    max_candidates: int = 6               # carriers considered per shipment


# ---------------------------------------------------------------------------
# Layer 1 - Data hygiene
# ---------------------------------------------------------------------------
def clean_rates(rates: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """rates columns: lane, carrier, rate. Adds 'valid' and 'issue'."""
    df = rates.copy()
    df["valid"], df["issue"] = True, ""

    is_placeholder = df["rate"].round(2).isin(cfg.placeholder_rates) | (df["rate"] <= 0)
    df.loc[is_placeholder, ["valid", "issue"]] = [False, "placeholder/zero rate"]

    for lane, g in df[df["valid"]].groupby("lane"):
        if len(g) < cfg.min_lane_obs_for_anomaly:
            continue
        q1, q3 = g["rate"].quantile([0.25, 0.75])
        lo, hi = q1 - cfg.anomaly_iqr_k * (q3 - q1), q3 + cfg.anomaly_iqr_k * (q3 - q1)
        out = g.index[(g["rate"] < lo) | (g["rate"] > hi)]
        df.loc[out, "issue"] = f"outlier vs lane range ({lo:,.0f}-{hi:,.0f})"
    return df


# ---------------------------------------------------------------------------
# Layer 2 - Acceptance model (hierarchical shrinkage)
# ---------------------------------------------------------------------------
@dataclass
class AcceptanceModel:
    cfg: Config
    global_rate: float = 0.75
    carrier_stats: dict = field(default_factory=dict)   # carrier -> (acc, n)
    lane_stats: dict = field(default_factory=dict)      # (lane, carrier) -> (acc, n)

    def fit(self, history: pd.DataFrame) -> "AcceptanceModel":
        """history columns: lane, carrier, accepted (0/1). One row per tender attempt."""
        if history.empty:
            self.global_rate = self.cfg.global_accept_prior
            return self
        self.global_rate = history["accepted"].mean()
        for c, g in history.groupby("carrier"):
            self.carrier_stats[c] = (g["accepted"].sum(), len(g))
        for k, g in history.groupby(["lane", "carrier"]):
            self.lane_stats[k] = (g["accepted"].sum(), len(g))
        return self

    def predict(self, lane: str, carrier: str) -> tuple[float, str]:
        k = self.cfg.prior_strength
        # carrier level, shrunk toward global
        acc_c, n_c = self.carrier_stats.get(carrier, (0, 0))
        p_carrier = (acc_c + k * self.global_rate) / (n_c + k)
        # lane level, shrunk toward carrier level
        acc_l, n_l = self.lane_stats.get((lane, carrier), (0, 0))
        p = (acc_l + k * p_carrier) / (n_l + k)
        basis = (f"{n_l} tenders on this lane, {n_c} overall" if n_c
                 else "no history - using portfolio average")
        return float(np.clip(p, 0.02, 0.99)), basis


# ---------------------------------------------------------------------------
# Layer 3 + 4 - Expected cost and optimal tender sequence
# ---------------------------------------------------------------------------
def sequence_expected_cost(seq, spot_cost, round_cost: float = 0.0,
                           escalation: float = 0.0) -> float:
    """
    seq: list of (rate, p_accept). Tender in order; if all reject -> spot.
    Each rejection costs `round_cost` and makes spot more expensive by
    `escalation` (fewer trucks left as pickup approaches).
    Without these two terms, 'cheapest first' is always optimal - which is
    exactly why the rejection cost must be calibrated from real data.
    """
    cost, reach = 0.0, 1.0
    for rate, p in seq:
        cost += reach * p * rate
        reach *= (1 - p)
        cost += reach * round_cost
    final_spot = spot_cost * (1 + escalation * len(seq))
    return cost + reach * final_spot


def best_tender_sequence(shipment: dict, offers: pd.DataFrame,
                         model: AcceptanceModel, cfg: Config,
                         spot_cost: Optional[float] = None) -> dict:
    """
    shipment: {id, lane, lead_time_hours}
    offers:   valid contract rates for this lane (carrier, rate)
    """
    lane = shipment["lane"]
    cands = []
    for _, r in offers.iterrows():
        p, basis = model.predict(lane, r["carrier"])
        cands.append({"carrier": r["carrier"], "rate": r["rate"], "p": p, "basis": basis})

    if spot_cost is None:
        ref = min((c["rate"] for c in cands), default=None)
        spot_cost = ref * cfg.spot_premium if ref else np.nan

    if not cands:
        return {"shipment": shipment["id"], "sequence": [], "expected_cost": spot_cost,
                "spot_risk": 1.0, "note": "No contract rate on lane -> contract gap"}

    # rounds allowed by lead time
    rounds = int(shipment.get("lead_time_hours", 48) // cfg.hours_per_tender_round)
    max_len = max(1, min(cfg.max_sequence_len, rounds, len(cands)))

    # prune to most promising candidates (cheapest single-tender expected cost)
    cands.sort(key=lambda c: sequence_expected_cost([(c["rate"], c["p"])], spot_cost,
                                                         cfg.reject_round_cost, cfg.spot_escalation_per_round))
    cands = cands[:cfg.max_candidates]

    best, best_cost = None, np.inf
    for L in range(1, max_len + 1):
        for perm in permutations(cands, L):
            ec = sequence_expected_cost([(c["rate"], c["p"]) for c in perm], spot_cost,
                                        cfg.reject_round_cost, cfg.spot_escalation_per_round)
            if ec < best_cost - 1e-9:
                best, best_cost = perm, ec

    spot_risk = float(np.prod([1 - c["p"] for c in best]))
    cheapest = min(cands, key=lambda c: c["rate"])
    note = ""
    if best[0]["carrier"] != cheapest["carrier"]:
        note = (f"Cheapest rate ({cheapest['carrier']} ${cheapest['rate']:,.0f}) not first: "
                f"acceptance only {cheapest['p']:.0%}, higher expected cost.")
    return {
        "shipment": shipment["id"],
        "sequence": [{"carrier": c["carrier"], "rate": c["rate"],
                      "p_accept": round(c["p"], 2), "basis": c["basis"]} for c in best],
        "expected_cost": round(best_cost, 2),
        "spot_risk": round(spot_risk, 3),
        "spot_cost_assumed": round(spot_cost, 2),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Layer 5 - Batch allocation with daily carrier capacity
# ---------------------------------------------------------------------------
def allocate_batch(shipments: list, rates: pd.DataFrame, model: AcceptanceModel,
                   cfg: Config, capacity: dict) -> pd.DataFrame:
    """
    Regret-first greedy: shipments whose 'second-best' option is much worse
    get their first choice first. Capacity counts first-tender loads per carrier.
    """
    valid = rates[rates["valid"]]
    remaining = dict(capacity)
    options = {}
    for s in shipments:
        offers = valid[valid["lane"] == s["lane"]]
        per_first = []
        for c in offers["carrier"].unique():
            sub = offers[offers["carrier"] != c]
            first = offers[offers["carrier"] == c]
            # force carrier c first, optimise the rest
            r_first = best_tender_sequence(s, first, model, cfg)
            r_rest = best_tender_sequence({**s, "lead_time_hours": max(
                0, s.get("lead_time_hours", 48) - cfg.hours_per_tender_round)},
                sub, model, cfg, spot_cost=r_first["spot_cost_assumed"]) if len(sub) else None
            p0 = r_first["sequence"][0]["p_accept"]
            rate0 = r_first["sequence"][0]["rate"]
            tail = r_rest["expected_cost"] if r_rest else r_first["spot_cost_assumed"]
            ec = p0 * rate0 + (1 - p0) * (cfg.reject_round_cost + tail)
            seq = r_first["sequence"] + (r_rest["sequence"] if r_rest else [])
            per_first.append((ec, c, seq))
        per_first.sort(key=lambda x: x[0])
        options[s["id"]] = per_first

    def regret(sid):
        o = options[sid]
        return (o[1][0] - o[0][0]) if len(o) > 1 else np.inf

    rows = []
    for sid in sorted(options, key=regret, reverse=True):
        chosen = None
        for ec, c, seq in options[sid]:
            if remaining.get(c, np.inf) > 0:
                chosen = (ec, c, seq)
                remaining[c] = remaining.get(c, np.inf) - 1
                break
        if chosen is None:
            rows.append({"shipment": sid, "first_carrier": None,
                         "sequence": "", "expected_cost": np.nan,
                         "note": "No capacity / no contract -> spot or new rate needed"})
            continue
        ec, c, seq = chosen
        best_unconstrained = options[sid][0][1]
        rows.append({
            "shipment": sid, "first_carrier": c,
            "sequence": " -> ".join(x["carrier"] for x in seq),
            "expected_cost": round(ec, 2),
            "note": "" if c == best_unconstrained else
                    f"Moved from {best_unconstrained} (capacity full)",
        })
    return pd.DataFrame(rows).sort_values("shipment").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data adapters - fill in once Huber tender history arrives
# ---------------------------------------------------------------------------
TENDER_HISTORY_SCHEMA = {
    "shipment_id": "OTM shipment GID",
    "lane": "origin -> destination (e.g. 'CRAMER_GA->DALLAS_TX')",
    "carrier": "servprov GID tendered",
    "tender_seq": "1, 2, 3 ... order of tender attempt",
    "tender_time": "timestamp tender sent",
    "response": "ACCEPTED / DECLINED / EXPIRED",
    "pickup_date": "planned pickup",
    "planned_cost": "OTM planned cost",
    "actual_cost": "final settled cost",
    "changed_by": "SYSTEM / planner user id",
}


def load_tender_history(path: str) -> pd.DataFrame:
    """Map Huber export to model input. Column names to confirm with Rafael."""
    df = pd.read_csv(path)
    missing = [c for c in ("lane", "carrier", "response") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected: {list(TENDER_HISTORY_SCHEMA)}")
    df["accepted"] = (df["response"].str.upper() == "ACCEPTED").astype(int)
    return df


# ---------------------------------------------------------------------------
# Demo with SYNTHETIC data only
# ---------------------------------------------------------------------------
def _demo():
    cfg = Config()
    rng = np.random.default_rng(7)
    lane = "PLANT_A->CITY_X"
    true_p = {"A": 0.92, "B": 0.55, "C": 0.85}
    hist = pd.DataFrame([{"lane": lane, "carrier": c, "accepted": int(rng.random() < p)}
                         for c, p in true_p.items() for _ in range(30)])
    model = AcceptanceModel(cfg).fit(hist)

    rates = clean_rates(pd.DataFrame([
        {"lane": lane, "carrier": "A", "rate": 1050},
        {"lane": lane, "carrier": "B", "rate": 980},
        {"lane": lane, "carrier": "C", "rate": 1020},
        {"lane": lane, "carrier": "GENERIC", "rate": 999999.99},
    ]), cfg)
    print("== Layer 1: data hygiene ==")
    print(rates.to_string(index=False), "\n")

    s = {"id": "SH001", "lane": lane, "lead_time_hours": 24}
    r = best_tender_sequence(s, rates[rates["valid"]], model, cfg)
    print("== Layers 2-4: single shipment ==")
    for i, x in enumerate(r["sequence"], 1):
        print(f"  {i}. {x['carrier']}  ${x['rate']:,.0f}  p={x['p_accept']:.0%}  ({x['basis']})")
    print(f"  Expected cost ${r['expected_cost']:,.2f} | spot risk {r['spot_risk']:.1%}")
    print(f"  Note: {r['note']}\n")

    first = r["sequence"][0]["carrier"]
    print(f"== Layer 5: batch of 10, carrier {first} capacity = 4 ==")
    batch = [{"id": f"SH{i:03d}", "lane": lane, "lead_time_hours": 24} for i in range(1, 11)]
    out = allocate_batch(batch, rates, model, cfg, capacity={"A": 10, "B": 10, "C": 10, first: 4})
    print(out.to_string(index=False))


if __name__ == "__main__":
    _demo()
