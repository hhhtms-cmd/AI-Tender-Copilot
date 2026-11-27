# AI Tender — Huber FTL Tender Copilot (v3)

## v3 — acceptance model learned from tender history

`huber_tender_copilot_v3.py` supersedes the v2 acceptance layer:

- Layer 2 is **learned from OTM tender history** (accepted / rejected /
  expired per tender attempt) instead of hand-set coefficients.
  Hierarchical shrinkage: lane+carrier → carrier+origin mill → carrier
  → global, with shrinkage strength chosen on an out-of-time test.
- Layer 5 uses an **empirical daily capacity** per carrier (90th
  percentile of accepted loads per pickup day) as a soft cap.
- Built-in **out-of-time backtest**: train on earlier months, test on
  later months; reports AUC/Brier and average tender attempts under
  actual vs. model ordering.
- Still `[CALIBRATE]`: per-lane carrier rates, rejection/spot costs in
  USD, and tender sent/response timestamps for wait time per round.
  Without rates, the sequence minimizes expected tender attempts.

Run inside the approved environment only; do not publish customer data
or customer-identifying outputs. The v2 expected-cost engine
(`algorithm.py`) and the demo UI (`index.html`) remain below.

A from-scratch tendering algorithm for Huber Woods full-truckload (FTL),
mostly 1 order = 1 truck. For each shipment it recommends the tender
sequence with the **lowest expected total cost** (rate + rejection risk
→ spot), then allocates a whole batch without overloading any carrier.

> ⚠️ Synthetic data only in this repo. Real Huber client data must stay
> inside the IBM environment — never commit it here.

## Five layers

| # | Layer | What it does | Fill in when real data arrives |
|---|---|---|---|
| 1 | Data hygiene | Drops OTM placeholder rates (e.g. 999,999.99); flags per-lane rate outliers (Tukey IQR) | Rate table |
| 2 | Acceptance model | Lane+carrier history, shrunk toward carrier level, then portfolio average; every prediction states its basis (e.g. "30 tenders on this lane") | Tender history: who was tendered, accept/decline |
| 3 | Expected cost | In USD, not a score: rate × acceptance + rejection cost + final spot cost | Rejection cost, spot premium |
| 4 | Tender sequence | Cheapest expected-cost order; fewer lead-time hours → fewer tender rounds | OTM response window per round |
| 5 | Batch allocation | Per-carrier daily capacity caps; shipments with the largest regret (worst second-best option) get their first choice first | Carrier capacity + allocation commitments |

## Key finding (synthetic test, verified)

A $1,050 / high acceptance · B $980 / low acceptance · C $1,020 / medium.
The optimal sequence depends on **what one rejection actually costs**:

| Rejection cost | Lead time | Optimal sequence | Expected cost |
|---:|---:|---|---:|
| $0 | 24h | B → C → A | $1,000.31 |
| $120 | 24h | B → C → A | $1,052.77 |
| $200 | 24h | A → C → B | $1,069.94 |
| $300 | 24h | A → C → B | $1,079.24 |
| any | 4h | A only | — |

Plainly: with plenty of time and cheap rejections, trying the cheapest
carrier first wins. When a rejection is expensive — planner rework, lost
lead time, spot getting pricier near pickup — or time is tight, tender
the most reliable carrier first. "Cheapest ≠ cheapest overall" is only
true under those conditions.

**The rejection cost at Huber is the single most important number to
calibrate from real data. Do not guess it.**

## Calibration — parameters marked `[CALIBRATE]`

In `Config` (algorithm.py): `prior_strength`, `global_accept_prior`,
`spot_premium`, `hours_per_tender_round`, `reject_round_cost`,
`spot_escalation_per_round`. Replace with values learned from Huber
tender history once Rafael's export arrives.

## When the data arrives

1. `load_tender_history(path)` — required fields are listed in
   `TENDER_HISTORY_SCHEMA` (shipment_id, lane, carrier, tender_seq,
   tender_time, response, pickup_date, planned_cost, actual_cost,
   changed_by). Column names to confirm with Rafael.
2. Estimate from the data: rejection cost, spot premium, and how much
   spot escalates as pickup approaches.
3. Backtest: compare actual outcomes vs. what the algorithm would have
   tendered.

## Run

```bash
pip install -r requirements.txt
python3 algorithm.py
```

Demo uses synthetic data only (seed 7): Layer 1 drops the 999,999.99
placeholder; Layers 2–4 rank B → C → A at the default $120 rejection
cost; Layer 5 moves overflow loads from B to C when B's capacity (4)
fills.
