"""
Huber FTL Tender Copilot - v3 (calibrated on OTM tender history)
================================================================
Changes vs v2
  * Layer 2 (acceptance model) is now LEARNED from the OTM Tender Activity
    Report instead of hand-set coefficients. Hierarchical shrinkage:
        lane+carrier  ->  carrier+origin mill  ->  carrier  ->  global
    Shrinkage strength K was chosen on an out-of-time test (K=3 best).
  * Layer 5 (batch allocation) uses an EMPIRICAL daily capacity per carrier
    (90th percentile of accepted loads per pickup day) as a soft cap.
  * Built-in out-of-time backtest (train Jul-Sep, test Oct-Nov).

Still needed  ([CALIBRATE] = not available in the current data)
  * Rate per carrier per lane        -> without it, the sequence minimizes
                                        expected tender attempts, not dollars.
  * Cost of one rejection / spot     -> provisional values below.
  * Tender sent / response timestamps-> needed for wait time per round and
                                        true last-minute analysis.

Data handling: run only inside the approved IBM environment. Do not publish
customer data or customer-identifying outputs.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
K_SHRINK = 3                      # calibrated on out-of-time test
TRAIN_END = "2026-10-01"          # time split for backtest
CAPACITY_QUANTILE = 0.90          # empirical daily capacity
P_FLOOR, P_CEIL = 0.01, 0.99
# [CALIBRATE] provisional cost assumptions (USD). Replace with values derived
# from planned-vs-actual cost data once linked by shipment ID.
REJECTION_COST = 200.0            # cost of one failed round (time, re-plan, price creep)
SPOT_PREMIUM_PCT = 0.55           # spot vs planned rate, provisional median
MAX_ROUNDS_DEFAULT = 4            # [CALIBRATE] depends on tender response window


# ------------------------------------------------------------- 1. load/clean
def load_tender_history(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet2", header=3, dtype=str).dropna(how="all")
    df = df[df.TENDER_TRANSACTION_NO.notna()].copy()
    df["tno"] = pd.to_numeric(df.TENDER_TRANSACTION_NO)
    df["carrier"] = df.SERVICE_PROVIDER_ID.str.replace("HEW.", "", regex=False)
    df["origin"] = df.SOURCE_LOCATION_NAME
    state = df.DESTINATION_LOCATION_NAME.str[-2:]
    df["lane"] = df.origin + ">" + df.DESTINATION_CITY + "," + state
    df["pickup"] = pd.to_datetime(df.PICKUP_DATE, utc=True)
    df["accepted"] = (df.TENDER_ACCEPTANCE_STATUS == "ACCEPTED").astype(int)
    df["expired"] = (df.TENDER_ACCEPTANCE_STATUS == "EXPIRED").astype(int)
    df = df.sort_values(["SHIPMENT_GID", "tno"])
    df["attempt"] = df.groupby("SHIPMENT_GID").cumcount() + 1
    return df


# ---------------------------------------------------- 2. acceptance model
class AcceptanceModel:
    """Hierarchical Bayesian-style shrinkage of acceptance rates."""

    def __init__(self, k: float = K_SHRINK):
        self.k = k

    def fit(self, d: pd.DataFrame) -> "AcceptanceModel":
        k = self.k
        self.g = d.accepted.mean()
        c = d.groupby("carrier").accepted.agg(["sum", "count"])
        self.p_c = (c["sum"] + k * self.g) / (c["count"] + k)
        co = d.groupby(["carrier", "origin"]).accepted.agg(["sum", "count"]).reset_index()
        co["prior"] = co.carrier.map(self.p_c).fillna(self.g)
        co["p"] = (co["sum"] + k * co.prior) / (co["count"] + k)
        self.p_co = co.set_index(["carrier", "origin"]).p
        cl = d.groupby(["carrier", "lane", "origin"]).accepted.agg(["sum", "count"]).reset_index()
        cl["prior"] = [self.p_co.get((a, o), self.p_c.get(a, self.g)) for a, o in zip(cl.carrier, cl.origin)]
        cl["p"] = (cl["sum"] + k * cl.prior) / (cl["count"] + k)
        self.p_cl = cl.set_index(["carrier", "lane"]).p
        self.n_cl = cl.set_index(["carrier", "lane"])["count"]
        return self

    def predict_one(self, carrier: str, lane: str, origin: str) -> tuple[float, str]:
        if (carrier, lane) in self.p_cl.index:
            p, basis = self.p_cl[(carrier, lane)], f"lane history ({int(self.n_cl[(carrier, lane)])} tenders)"
        elif (carrier, origin) in self.p_co.index:
            p, basis = self.p_co[(carrier, origin)], "carrier history at this mill"
        elif carrier in self.p_c.index:
            p, basis = self.p_c[carrier], "carrier overall history"
        else:
            p, basis = self.g, "network average (no history)"
        return float(np.clip(p, P_FLOOR, P_CEIL)), basis

    def predict(self, d: pd.DataFrame) -> pd.Series:
        return pd.Series([self.predict_one(c, l, o)[0] for c, l, o in zip(d.carrier, d.lane, d.origin)], index=d.index)


# ------------------------------------------- 3+4. expected cost & sequence
def expected_sequence_cost(seq: list[dict], max_rounds: int, spot_cost: float | None) -> dict:
    """seq: [{'carrier','p','rate'(optional)}]. Returns expected attempts and USD."""
    reach, exp_attempts, exp_usd = 1.0, 0.0, 0.0
    for i, c in enumerate(seq[:max_rounds]):
        exp_attempts += reach
        if c.get("rate") is not None:
            exp_usd += reach * c["p"] * c["rate"]
        exp_usd += reach * (1 - c["p"]) * REJECTION_COST
        reach *= 1 - c["p"]
    p_spot = reach
    if spot_cost is not None:
        exp_usd += p_spot * spot_cost
    return {"expected_attempts": round(exp_attempts, 2), "p_spot": round(p_spot, 3),
            "expected_cost_usd": round(exp_usd, 0) if spot_cost is not None else None}


def recommend_sequence(model: AcceptanceModel, lane: str, origin: str, candidates: list[str],
                       rates: dict[str, float] | None = None, max_rounds: int = MAX_ROUNDS_DEFAULT) -> dict:
    rows = []
    for c in candidates:
        p, basis = model.predict_one(c, lane, origin)
        rows.append({"carrier": c, "p": p, "basis": basis, "rate": (rates or {}).get(c)})
    if rates:  # dollar objective: rank by rate + risk-adjusted cost of failure
        key = lambda r: r["rate"] + (1 - r["p"]) / max(r["p"], 0.01) * REJECTION_COST
        ordered = sorted(rows, key=key)
        ref = min(r["rate"] for r in rows if r["rate"] is not None)
        spot = ref * (1 + SPOT_PREMIUM_PCT)
    else:      # no rates yet: minimize expected attempts
        ordered = sorted(rows, key=lambda r: -r["p"])
        spot = None
    return {"sequence": ordered, **expected_sequence_cost(ordered, max_rounds, spot)}


# ---------------------------------------------------- 5. batch allocation
def empirical_capacity(d: pd.DataFrame, q: float = CAPACITY_QUANTILE) -> pd.Series:
    acc = d[d.accepted == 1].assign(day=lambda x: x.pickup.dt.date)
    return acc.groupby(["carrier", "day"]).size().groupby("carrier").quantile(q).round()


def allocate_batch(model: AcceptanceModel, shipments: pd.DataFrame, candidates: dict[str, list[str]],
                   capacity: pd.Series) -> pd.DataFrame:
    """Greedy: shipments with the largest gap between best and 2nd-best carrier go first."""
    plans = []
    for _, s in shipments.iterrows():
        ps = sorted(((model.predict_one(c, s.lane, s.origin)[0], c) for c in candidates[s.lane]), reverse=True)
        gap = ps[0][0] - (ps[1][0] if len(ps) > 1 else 0)
        plans.append((gap, s.SHIPMENT_GID, ps))
    used, out = {}, []
    for gap, gid, ps in sorted(plans, reverse=True):
        for p, c in ps:
            if used.get(c, 0) < capacity.get(c, 1):
                used[c] = used.get(c, 0) + 1
                out.append({"shipment": gid, "first_tender": c, "p_accept": round(p, 3)})
                break
        else:
            out.append({"shipment": gid, "first_tender": ps[0][1], "p_accept": round(ps[0][0], 3), "note": "capacity exceeded"})
    return pd.DataFrame(out)


# --------------------------------------------------------------- backtest
def backtest(df: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    tr, te = df[df.pickup < TRAIN_END], df[df.pickup >= TRAIN_END].copy()
    m = AcceptanceModel().fit(tr)
    te["p"] = m.predict(te)
    base = te.carrier.map(m.p_c).fillna(m.g).clip(P_FLOOR, P_CEIL)
    rows = []
    for gid, x in te.groupby("SHIPMENT_GID"):
        car = x.groupby("carrier").agg(p=("p", "first"), acc=("accepted", "max"))
        if car.acc.max() == 0:
            continue
        order = car.sort_values("p", ascending=False).acc.values
        rows.append((len(x), int(np.argmax(order == 1)) + 1, len(car)))
    r = pd.DataFrame(rows, columns=["actual", "model", "ncar"])
    mt = r[r.actual > 1]
    return {
        "train_tenders": len(tr), "test_tenders": len(te),
        "auc_model": round(roc_auc_score(te.accepted, te.p), 3),
        "auc_carrier_only": round(roc_auc_score(te.accepted, base), 3),
        "brier_model": round(brier_score_loss(te.accepted, te.p), 3),
        "brier_network_avg": round(brier_score_loss(te.accepted, np.full(len(te), m.g)), 3),
        "multi_tender_shipments": len(mt),
        "avg_attempts_actual": round(mt.actual.mean(), 2),
        "avg_attempts_random_order": round(((mt.ncar + 1) / 2).mean(), 2),
        "avg_attempts_model_order": round(mt.model.mean(), 2),
        "note": "Re-ranks only carriers actually tendered; optimistic upper bound.",
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "HEW.TENDER_ACTIVITY_REPORT-NOV 27.xlsx"
    data = load_tender_history(path)
    print("Backtest:", backtest(data))
    model = AcceptanceModel().fit(data)
    cap = empirical_capacity(data)
    lane, origin = "CRY>HAZLETON,PA", "CRY"
    cands = data[data.lane == lane].carrier.unique().tolist()
    rec = recommend_sequence(model, lane, origin, cands)
    print(f"\nExample lane {lane}: expected attempts {rec['expected_attempts']}, P(spot) {rec['p_spot']}")
    for s in rec["sequence"][:4]:
        print(f"  {s['carrier']:6s} p={s['p']:.2f}  ({s['basis']})")
