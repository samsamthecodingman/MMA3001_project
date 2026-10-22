# Overnight build — progress & handoff

Working notes for the autonomous build session (kept in the repo so work can
resume after interruptions). Plan: see "Execution order" below.

## Status
- [x] Repo scaffold, physics model (forward Euler/RK4/exact/lfilter, inverse, LS calibration), resampling, synthetic generator — 17 tests pass
- [x] Pipeline + `scripts/run_experiment.py --synthetic` (baseline, physics, linear/tree/forest/SVR/MLP, hybrids)
- [x] `scripts/sensitivity_synthetic.py` (convergence, Euler stability, dt/scheme/noise sweeps, target alignment)
- [x] `scripts/profile_performance.py` (scaling, FLOPs, arithmetic intensity, cProfile)
- [x] Real data downloaded from GitHub release `data-v1` → `data/raw/` (not committed)
- [x] Ingest real data, audit (3/5 env sensors frozen, 76 % duplicate rows), data-driven sensor/space association
- [x] Real-data experiment: blocked leave-one-week-out CV (primary + 2 controls) + chronological regime-shift test + figures
- [x] Centre occupancy target on the CO2 sample time (alignment finding)
- [x] Real-data sensitivity: scheme × grid, g-invariance, hyper-parameters, learning curve
- [x] Docs: README, LICENSE, Makefile, requirements, docstrings, pdoc HTML (docs/), pytest-html report
- [x] Report → reports/report.pdf (body ends p.10); DECISIONS.md; AI_USE.md (draft — student must complete [YOU] parts); VIVA_PREP.md
- [ ] Notebook walkthrough (in progress)
- [ ] Independent code review → fixes
- [ ] Clean `make all` reproducibility run
- [ ] Presentation slides (nice to have)

## Data facts (from inspection)
- Env CSV: one row per sensor report; columns id, sensorid, devicetype, jsondata (list of {values:[{time,value}], variable:{name,unit}}), status, ... Variables include CO2 (ppm), Humidity (HR%), Temperature (°C), Pressure (mb), Formaldehyde, LVOC, PM, Battery.
- Occ CSV: ~9M event rows; deviceid, floorspaceid, occupancystatus (CurrentlyOccupied/RecentlyOccupied/NotOccupied), headcount, collecteddate (+11 tz).
- Sensor sheet OCC tab flagged "SpaceFloorIDs invalid" → matching occupancy spaces to env sensors is a key risk.

## Key synthetic findings so far
- RK4 order 4.14, Euler 0.97 (theory 4, 1). Euler unstable for h > 2τ as predicted.
- Target alignment: interval-mean occupancy labelled at interval start leads CO2 by dt/2 → V overestimated ~+40% at 15 min; centring the target cuts this to ~+9%.
- lfilter exact solver ~40× faster than Python loop, ~280× faster than RK4 loop; inverse estimator ~5–10 ns/sample (memory-bound, AI ≈ 0.5 FLOP/B); MLP ≈ 7.3 kFLOP/sample (AI ≈ 37).
