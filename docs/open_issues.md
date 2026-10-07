# Open issues (engine, Phases 2–3)

Last updated 2026-10-06. Earlier open items were only in the Phase 3 chat summary; this file now
keeps them in the repo.

## Resolved by the owner's decisions (CLAUDE.md §17)

| Item | Decision |
|---|---|
| Energy importers run dry (firm inputs come from stock, nothing refilled it) | D30: firms' planned energy inputs are part of ENERGY demand (stock-building) |
| Tax base: GDP vs household income (fiscal open question A) | D31: taxes on wages + private profits; state profits untaxed |
| Debt premium becomes infinite when GDP hits 0 | D32: floored 4-turn GDP average in every debt ratio; premium cap 0.20/yr |
| Firm losses, what H is, timing of policies and income | D33 |
| Who pays for imports bought for the government or for the energy stock | D34 (assistant, keeps the ledger closed) |
| Household cash could end a turn negative (landed costs above home prices) | D35 (assistant, keeps `H ≥ 0`) |

After these changes every §6.14 invariant holds: 14 turns × 20 seeds, with and without shocks.

## Update after D52–D53 (2026-10-07, Phase 4 calibration pass 4)

- **S2 resolved (D52):** the blocker was the D37 payout key (bond-market surplus pro rata to household
  cash sent ~30% of world interest to DORNE). Paid by population now; settled net exports are −2.2% to
  +3.1% of GDP and every roster country is the top exporter of its good (`tests/test_roster.py`).
- **S1 partly addressed (D53):** w_ref = 0.08 removes FALKEN's permanent welfare drag. The flow minimum
  is 41.9 (floor 40) with k_m = 7. Still open: k_m stays far from the spec (1) because status-quo play
  has real background noise (firm exits cause short world food/energy shortages; mild deflation costs
  2–5 points a turn through the inflation term; a debt-heavy country can default). A dead band on the
  inflation term would be an equation change (needs owner approval).
- **New, S5: the aggressor mostly topples itself.** In `tests/test_shock_bite.py` a country that
  sanctions everyone and maxes military spending loses its own leader in almost every seed; victims
  fall rarely (CERES most often). Sanctioning everyone is close to self-imposed autarky. Worth watching
  when LLM agents play: the engine punishes broad sanctions hard and targeted ones lightly.

## Status after D45–D51 (2026-10-07): the flow CHECK passes

All Phase 2 / Phase 3 checks pass with no xfail (20 seeds): GDP 0.75–1.02 × start, prices
0.58–1.43 × settled, unemployment ≤ 15.3%, stability 40.3–73.0, money growth 0%, bond market and
treasuries bounded, the burn-in settles (29 turns), test_energy, the food-floor tests, and the
14-turn × 20-seed shock run. Numbers and every parameter change: `docs/calibration.md`, pass 3.

Still open (none of these fail a test):
- **S1, thin stability margin:** the minimum is 40.3 against a floor of 40, with k_m = 6 (spec 1).
  The real cause is the stability formula: |π − 2%| costs points every turn even near target, and
  FALKEN's welfare share sits below w_ref permanently. Next: a dead band on the inflation term, or a
  per-country w_ref equal to the starting welfare share; either would let k_m go back toward 1.
- **S2, D41 trade balance not met:** settled net exports are −15.8% of GDP (DORNE) to +8.4%
  (BRONTIA), and AURELIA is not the largest SERVICES exporter. Next: weight the net-export term
  more in `scripts/calibrate_balance.py`, or allow θ (home bias) to vary by good.
- **S3, D50 and D51 are built but off:** turning them on (PASS 2) made the burn-in stop settling.
- **S4, burn-in sensitivity:** before D45 the settled state depended chaotically on tiny parameter
  changes. With D45 and D47 the search now evaluates exactly what it writes, and the result is
  reproducible, but it was not re-tested for robustness to small A changes.

## Known limitation (D48, not fixed): government borrowing creates money

When a treasury runs out of cash, the government borrows from the `bond_market` account, which may
go negative: it is the money issuer, so each borrowed credit is new money. Interest the governments
pay later goes back to the bond market and the D37 sweep pays any surplus to households. So a
government that keeps running deficits keeps adding money to the economy. Under status-quo play this
stays small; `tests/test_money_supply.py` checks that world money (household cash + treasuries) grows
by less than 1% per turn. Agents who run large deficits on purpose could make it grow faster; the
experiments should report money growth per run.

## Update after D40–D44 and calibration pass 2 (2026-10-06)

Fixed: money loop (bond market ≈ 0, treasuries bounded), unemployment (≤ 8%), GDP stays in the band
(0.94–1.33 × start, seed 1), D43 floor (does not bind at the settled state, binds in a harvest failure).
Still open (numbers in `docs/calibration.md`, pass 2):
- **P1 (burn-in does not converge):** after 40 turns GDP still moves > 1% per turn. Shortages of food
  and energy come back in waves, and the settled state depends chaotically on tiny parameter changes
  (rounding A in the 5th digit moves net exports by 5 points). Likely cause: §6.4 imports only fill
  gaps, so who supplies what follows price paths, not productivity.
- **P2 (prices):** relative prices at the settled state span 0.33–2.0.
- **P3 (stability):** recurring 5–30% food/energy shortages × k_f = 3 / k_e = 1.5 cost up to 31
  stability points a turn; stability sinks to 0–20 even with full employment and stable GDP.
- **P4 (government-interest loop):** with inflation, the Taylor rate raises government interest, which
  is borrowed (new money) and swept to households (D37). Removing savings interest (D42) did not remove
  this path; it is quiet only while prices are stable.

## Update after D36–D39 and calibration pass 1 (2026-10-06)

F1–F4 below were answered by D36 (food floor), D37 (money loop), D38 (burn-in) and D39 (energy
spoilage), plus one yaml calibration pass. Results and the current per-turn table are in
`docs/calibration.md`. Still open:
- **O1 (D36):** 70% of *world* food output per person (f_min = 0.0272) binds in 4 of 6 countries at the
  settled state. 70% of the lowest country's food demand per person would be 0.0144.
- **O2:** the 8-turn burn-in does not settle: energy importers still have 20–28% ENERGY shortages, and
  BRONTIA's ENERGY sector has zero revenue. Stability then falls to 0–30 in turns 1–3.
- **O3 (§6.4 trade structure):** imports only fill gaps and exports never compete with home goods, so
  energy importers lose money every turn until they shrink to ~50% of their starting GDP.
- **O4 (§6.8):** savings interest is new money from the bond market, and higher inflation raises `r`,
  so the loop can run away. The price cap 0.05 hides it under status quo; shocks or agents may not.

The sections below are the history before D36–D39 (old parameters).

## Still open: the economy does not settle (before D36–D39) (needs owner decisions, Phase 4 calibration)

The engine is internally consistent (money, goods and the income identity all close to 1e-9), but
with the current parameters the status-quo economy collapses in the first turns and never settles.
`tests/test_flow.py` and `tests/test_energy.py::test_no_country_starved_of_energy` fail because of
this. Their bands were not loosened.

### F1. The minimum food need is about 4× world food output
`f_min · pop` = 0.15 × 1133M people = **170 FOOD units/quarter**; world FOOD output at the start is
**43** (DORNE 3.1, BRONTIA 9.3, CERES 19.4, FALKEN 7.4, AURELIA 1.8, EVERMERE 2.0). The floor
always binds, so households put their whole budget into food (BRONTIA turn 1: FOOD demand 42.8,
GOODS 1.5, TECH 0, SERVICES 0). Every non-food sector loses its customers, and unemployment reaches
60–90% in turn 2. CERES (the food exporter) then sees runaway food prices: CPI 4,850 by turn 60.

### F2. Money drains into the bond market and idle treasuries
Even with the food floor made non-binding (diagnostic run, `f_min = 0.01`, not committed), world GDP
falls 211 → 37 over 37 turns. Household cash falls 142 → 40 while the `bond_market` balance grows
3 → 95 and treasuries hold ~95 that is never spent. Government interest goes to the bond market
faster than savings interest comes back, and a treasury surplus is never recycled. Money is
conserved, but it leaves the circular flow.

### F3. Stability falls from 70 to 0 in turn 1
Turn-1 inputs (f_min = 0.01 run): annualised CPI inflation −11% to −37% (`k_π` term up to −31),
and energy shortage fractions of 27–69% for importers (`k_e` term up to −100). Part of this is the
rough pre-burn-in start (§6.15), which has a full quarter of stock on top of output, so sales are
far below output in turn 1.

### F4. Energy: importers short, DORNE's energy sector dies
Firm energy fill falls to 0.44 (CERES t2) and 0.64 (DORNE t14). DORNE never exports while its own
firms are short (D30 works). But its energy price falls (0.31 by t9) and its revenue with it, so
output goes 10.3 → 0 by t14. Even when it works, 5% ENERGY spoilage means a refilled stock covers
only ~95% of next turn's plan.

### Proposals (not applied; each needs an owner decision)
1. F1: calibrate `f_min` against per-capita food output (for example, 50% of starting food
   consumption per head, ≈ 0.02), or scale productivity/population so the floor is reachable.
2. F2: recycle the bond market's net interest income to households (as bond-holder interest) and/or
   let a treasury above a buffer repay debt or lower borrowing. Both are §6.7/§6.8 equation changes.
3. F3: implement the §6.15 burn-in (Phase 4) and start stability at its configured value after burn-in.
4. F4: plan firm energy as `E_d / (1 − δ_ENERGY)` so spoilage does not ration inputs every turn.

## Per-turn numbers, seed 1, status quo, no shocks (current parameters)

Prices: P_FOOD, P_ENERGY, CPI. "Firm energy fill" = energy used / planned energy demand.

| t | DORNE | BRONTIA | CERES | FALKEN | AURELIA | EVERMERE |
|---|---|---|---|---|---|---|
| 0 GDP | 19.5 | 49.2 | 29.8 | 25.6 | 25.3 | 23.7 |
| 0 P_FOOD | 1 | 1 | 1 | 1 | 1 | 1 |
| 0 P_ENERGY | 1 | 1 | 1 | 1 | 1 | 1 |
| 0 CPI | 1 | 1 | 1 | 1 | 1 | 1 |
| 0 ENERGY stock | 7.78 | 9.83 | 4.72 | 3.2 | 3.05 | 3.2 |
| 1 GDP | 23 | 54.1 | 41.6 | 26.3 | 35.8 | 30.7 |
| 1 P_FOOD | 1.2 | 1.17 | 1.2 | 1.05 | 1.2 | 1.2 |
| 1 P_ENERGY | 1.12 | 1.13 | 1.1 | 1.2 | 1.13 | 1.1 |
| 1 CPI | 0.93 | 0.923 | 0.927 | 0.906 | 0.936 | 0.927 |
| 1 ENERGY stock | 2.12 | 6.69 | 3.38 | 2.67 | 2.05 | 2.3 |
| 1 firm energy fill | 1 | 0.984 | 0.991 | 1 | 0.996 | 0.991 |
| 1 unemployment | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 stability | 15 | 0 | 3.59 | 0 | 0 | 0 |
| 2 GDP | 19.6 | 15.7 | 54.8 | 17.8 | 9.05 | 5.92 |
| 2 P_FOOD | 1.33 | 1.19 | 1.27 | 1.05 | 1.36 | 1.33 |
| 2 P_ENERGY | 0.896 | 1.16 | 1.14 | 1.22 | 1.13 | 1.32 |
| 2 CPI | 0.87 | 0.872 | 0.986 | 0.851 | 0.86 | 0.881 |
| 2 ENERGY stock | 11.4 | 2.83 | 6.44 | 1.65 | 1.31 | 1.04 |
| 2 firm energy fill | 1 | 1 | 0.443 | 1 | 1 | 1 |
| 2 unemployment | 0.22 | 0.593 | 0 | 0.292 | 0.651 | 0.697 |
| 2 stability | 0 | 15 | 15 | 15 | 0 | 15 |
| 3 GDP | 11.7 | 16.2 | 61.4 | 17.9 | 1.17 | 2.61 |
| 3 P_FOOD | 1.33 | 1.19 | 1.25 | 1.05 | 1.36 | 1.33 |
| 3 P_ENERGY | 0.75 | 1.16 | 1.14 | 1.24 | 1.35 | 1.59 |
| 3 CPI | 0.803 | 0.825 | 0.951 | 0.806 | 0.845 | 0.868 |
| 3 ENERGY stock | 10.6 | 2.73 | 8.09 | 1.65 | 0.418 | 0.535 |
| 3 firm energy fill | 1 | 0.984 | 0.757 | 0.952 | 1 | 1 |
| 3 unemployment | 0.533 | 0.544 | 0 | 0.193 | 0.898 | 0.795 |
| 3 stability | 15 | 0 | 23.9 | 0 | 15 | 0 |
| 4 GDP | 13.7 | 14.1 | 64 | 18.1 | 0.929 | 2.22 |
| 4 P_FOOD | 1.33 | 1.27 | 1.11 | 1.1 | 1.36 | 1.33 |
| 4 P_ENERGY | 0.653 | 1.16 | 1.14 | 1.28 | 1.62 | 1.9 |
| 4 CPI | 0.777 | 0.805 | 0.972 | 0.786 | 0.853 | 0.871 |
| 4 ENERGY stock | 9.9 | 2.17 | 8.17 | 1.48 | 0.215 | 0.272 |
| 4 firm energy fill | 1 | 1 | 0.94 | 1 | 1 | 1 |
| 4 unemployment | 0.358 | 0.602 | 0 | 0.167 | 0.913 | 0.838 |
| 4 stability | 15 | 0 | 24.5 | 0 | 15 | 0 |
| 5 GDP | 14.7 | 16.3 | 54.3 | 21.6 | 1.1 | 2.91 |
| 5 P_FOOD | 1.33 | 1.33 | 0.943 | 1.12 | 1.36 | 1.33 |
| 5 P_ENERGY | 0.535 | 1.16 | 1.14 | 1.2 | 1.62 | 1.97 |
| 5 CPI | 0.749 | 0.788 | 0.997 | 0.753 | 0.826 | 0.849 |
| 5 ENERGY stock | 13.8 | 2.33 | 6.28 | 2.22 | 0.26 | 0.315 |
| 5 firm energy fill | 1 | 0.886 | 1 | 0.877 | 0.787 | 0.82 |
| 5 unemployment | 0.108 | 0.544 | 0 | 0.0111 | 0.898 | 0.795 |
| 5 stability | 18.4 | 15 | 26 | 15 | 0 | 0 |
| 6 GDP | 14.1 | 19.9 | 41.3 | 21.6 | 0.708 | 2.26 |
| 6 P_FOOD | 1.33 | 1.24 | 0.775 | 1.09 | 1.36 | 1.33 |
| 6 P_ENERGY | 0.439 | 1.16 | 1.14 | 1.16 | 1.95 | 1.78 |
| 6 CPI | 0.75 | 0.744 | 1.03 | 0.728 | 0.853 | 0.802 |
| 6 ENERGY stock | 14.5 | 2.29 | 5.42 | 2.18 | 0.11 | 0.437 |
| 6 firm energy fill | 1 | 0.965 | 1 | 1 | 1 | 1 |
| 6 unemployment | 0.17 | 0.491 | 0.0849 | 0 | 0.948 | 0.846 |
| 6 stability | 27.9 | 0 | 21.8 | 1.94 | 0 | 0 |
| 7 GDP | 15.8 | 19.6 | 27.3 | 20.9 | 0.894 | 2.11 |
| 7 P_FOOD | 1.33 | 1.11 | 0.638 | 1.04 | 1.36 | 1.33 |
| 7 P_ENERGY | 0.36 | 1.16 | 1.14 | 1.1 | 2.11 | 1.52 |
| 7 CPI | 0.733 | 0.697 | 1.09 | 0.704 | 0.881 | 0.759 |
| 7 ENERGY stock | 15.2 | 2.34 | 4.05 | 2.43 | 0.114 | 0.625 |
| 7 firm energy fill | 1 | 0.932 | 1 | 1 | 0.912 | 1 |
| 7 unemployment | 0 | 0.424 | 0.281 | 0 | 0.926 | 0.838 |
| 7 stability | 24.4 | 0 | 15 | 0 | 0 | 0 |
| 8 GDP | 16 | 17.6 | 19.1 | 20 | 0.906 | 1.91 |
| 8 P_FOOD | 1.31 | 1.01 | 0.532 | 0.973 | 1.36 | 1.33 |
| 8 P_ENERGY | 0.319 | 1.16 | 1.14 | 1.01 | 1.95 | 1.32 |
| 8 CPI | 0.72 | 0.658 | 1.18 | 0.672 | 0.875 | 0.741 |
| 8 ENERGY stock | 11.9 | 2.38 | 3.33 | 2.63 | 0.139 | 0.649 |
| 8 firm energy fill | 1 | 0.931 | 1 | 1 | 1 | 1 |
| 8 unemployment | 0 | 0.357 | 0.335 | 0.0383 | 0.934 | 0.823 |
| 8 stability | 23.6 | 15 | 15 | 0 | 0 | 15 |
| 9 GDP | 16 | 18.2 | 15.3 | 18.7 | 0.865 | 2.1 |
| 9 P_FOOD | 1.28 | 0.916 | 0.45 | 0.882 | 1.36 | 1.33 |
| 9 P_ENERGY | 0.311 | 1.16 | 1.14 | 0.893 | 1.68 | 1.09 |
| 9 CPI | 0.71 | 0.626 | 1.29 | 0.633 | 0.861 | 0.723 |
| 9 ENERGY stock | 8.27 | 2.59 | 2.72 | 2.99 | 0.194 | 0.775 |
| 9 firm energy fill | 1 | 0.874 | 1 | 1 | 1 | 1 |
| 9 unemployment | 0 | 0.243 | 0.397 | 0.125 | 0.935 | 0.824 |
| 9 stability | 24.1 | 15 | 15 | 15 | 0 | 15 |
| 10 GDP | 15.7 | 17.2 | 11.3 | 15.4 | 0.822 | 2.24 |
| 10 P_FOOD | 1.24 | 0.839 | 0.392 | 0.789 | 1.36 | 1.33 |
| 10 P_ENERGY | 0.313 | 1.24 | 1.21 | 0.897 | 2.01 | 1.31 |
| 10 CPI | 0.7 | 0.61 | 1.44 | 0.61 | 0.924 | 0.767 |
| 10 ENERGY stock | 7.49 | 2.1 | 1.9 | 1.72 | 0.0659 | 0.266 |
| 10 firm energy fill | 1 | 0.94 | 1 | 1 | 1 | 1 |
| 10 unemployment | 0 | 0.135 | 0.44 | 0.183 | 0.939 | 0.809 |
| 10 stability | 21.3 | 15 | 15 | 15 | 0 | 15 |
| 11 GDP | 15.1 | 16.9 | 9.07 | 14.8 | 1.12 | 3.29 |
| 11 P_FOOD | 1.17 | 0.772 | 0.357 | 0.711 | 1.36 | 1.33 |
| 11 P_ENERGY | 0.331 | 1.38 | 1.34 | 0.974 | 2.42 | 1.57 |
| 11 CPI | 0.687 | 0.606 | 1.63 | 0.6 | 1 | 0.821 |
| 11 ENERGY stock | 6.14 | 1.72 | 1.31 | 1.55 | 0.0665 | 0.287 |
| 11 firm energy fill | 0.971 | 0.845 | 1 | 1 | 0.941 | 0.881 |
| 11 unemployment | 0 | 0.0958 | 0.487 | 0.158 | 0.911 | 0.705 |
| 11 stability | 0 | 15 | 15 | 1.48 | 15 | 15 |
| 12 GDP | 12.1 | 14.3 | 7.96 | 13.8 | 1.36 | 3.88 |
| 12 P_FOOD | 1.1 | 0.728 | 0.346 | 0.647 | 1.36 | 1.33 |
| 12 P_ENERGY | 0.358 | 1.49 | 1.43 | 1.05 | 2.9 | 1.89 |
| 12 CPI | 0.68 | 0.605 | 1.86 | 0.595 | 1.06 | 0.86 |
| 12 ENERGY stock | 4.91 | 1.59 | 1.14 | 1.36 | 0.0596 | 0.287 |
| 12 firm energy fill | 0.935 | 0.816 | 0.888 | 1 | 1 | 0.951 |
| 12 unemployment | 0.103 | 0.0943 | 0.502 | 0.129 | 0.893 | 0.612 |
| 12 stability | 15 | 0 | 15 | 15 | 15 | 0 |
| 13 GDP | 10.4 | 14.1 | 7.61 | 12.4 | 1.9 | 4.89 |
| 13 P_FOOD | 1.05 | 0.713 | 0.346 | 0.631 | 1.36 | 1.33 |
| 13 P_ENERGY | 0.43 | 1.57 | 1.5 | 1.2 | 3.48 | 2.27 |
| 13 CPI | 0.685 | 0.611 | 2.14 | 0.612 | 1.17 | 0.909 |
| 13 ENERGY stock | 2.68 | 1.49 | 1.02 | 1.2 | 0.0594 | 0.287 |
| 13 firm energy fill | 0.928 | 0.866 | 0.922 | 1 | 0.953 | 0.948 |
| 13 unemployment | 0.188 | 0.104 | 0.507 | 0.103 | 0.856 | 0.479 |
| 13 stability | 0 | 15 | 15 | 15 | 0 | 0 |
| 14 GDP | 8.71 | 14.5 | 7.13 | 14.1 | 2.76 | 6.87 |
| 14 P_FOOD | 1.05 | 0.747 | 0.359 | 0.676 | 1.39 | 1.34 |
| 14 P_ENERGY | 0.46 | 1.6 | 1.51 | 1.27 | 4.18 | 2.37 |
| 14 CPI | 0.716 | 0.625 | 2.51 | 0.646 | 1.31 | 0.947 |
| 14 ENERGY stock | 3.25 | 1.56 | 0.9 | 1.13 | 0.0666 | 0.302 |
| 14 firm energy fill | 0.638 | 0.842 | 1 | 1 | 0.847 | 0.904 |
| 14 unemployment | 0.146 | 0.0328 | 0.549 | 0 | 0.778 | 0.26 |
| 14 stability | 15 | 4.63 | 0 | 15 | 15 | 15 |
