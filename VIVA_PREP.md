# Presentation and interview preparation

The presentation is 5 minutes plus about 3 minutes of questions, in Week 14, with no AI allowed.
Learn the *reasons* rather than the wording. Every number below is in
`reports/results/`, and every decision is in `DECISIONS.md`.

---

## A. Five-minute presentation plan (6 slides plus a live repository demo)

| Time | Slide | Say |
|---|---|---|
| 0:00–0:40 | **1. Problem** | HVAC ventilates to schedule, not to actual people. Counters are rare, CO2 sensors are common. Question: can existing CO2 sensors estimate occupancy every 15 min? Input: CO2/T/RH. Output: persons per 15 min. |
| 0:40–1:30 | **2. Data reality** (`real_audit_timeline.png`, `real_association.png`) | 76 % of env rows were duplicates. 3 of 5 sensors were frozen. The location sheet doesn't link sensors to counters. I inferred the pairing from data, using correlation after removing the weekly pattern (r = 0.65), and kept weaker pairs as controls. |
| 1:30–2:20 | **3. Method** | Single-zone CO2 balance `V dC/dt = 1e6 g N − Q(C − C_out)`. Inverting it gives N from C and dC/dt (finite differences). It is linear in the unknowns, so calibration is one least-squares solve. Compared against a schedule baseline, linear, tree, forest, SVR and MLP, plus hybrids. |
| 2:20–3:00 | **4. Verification** (`fig_convergence.png`, the test report) | Observed orders: Euler 0.97, RK4 4.14. Euler unstable above 2τ, as theory predicts. Calibration recovers known V and Q on synthetic data. 27 pytest tests. Show the test report in the repo. |
| 3:00–4:00 | **5. Validation results** (`real_cv_bars.png`, `real_cv_week.png`) | Blocked leave-one-week-out CV. Physics R² 0.47 vs forest 0.51 vs baseline −0.05. MAE improves by 0.6–0.8 people. Weak at detecting *vacancy* (balanced accuracy ≤ 0.65). Controls show a much smaller gain. The 3-parameter physics model needs about 4 days of labels; the forest needs about 14. Every model under-predicts peaks above ~10 people. |
| 4:00–4:40 | **6. Limitations, lessons, AI** | One inferred pair, 25 days. Evaluation design mattered more than model choice (the exam-period test window). Centring the target fixed a +40 % V bias. AI wrote most of the code; I verified it with tests and synthetic truth. Errors it made: unit bug, split bug, alignment. |
| 4:40–5:00 | **Repo demo** | README → `docs/` (pdoc) → `reports/test_report.html` → `make all`. |

---

## B. The brief's sample questions, with answers

**"What physical or engineering system did your numerical model represent, and what assumptions were required to make the problem tractable?"**
A ventilated zone as one well-mixed control volume, under a CO2 mass balance: occupants generate CO2 at g = 5.2e-6 m³/s each, and ventilation exchanges air at Q with supply air at C_out. The assumptions are:
- perfect mixing;
- Q and C_out constant over the calibration window;
- constant per-person generation;
- no other CO2 sources or sinks;
- the sensor represents the zone.

Perfect mixing and representativeness are the weakest assumptions here: the sensor is mounted high in a large workshop, which is why peaks are under-predicted.

**"Which parameters or boundary conditions had the greatest influence on the behaviour of your solution, and how did you verify they were realistic?"**
- **Q and C_out dominate.** The estimate is mostly the steady-state term Q(C − C_out). Q was identified consistently at ≈ 0.25 m³/s (≈ 250 L/s) across all derivative schemes and grids. C_out ≈ 430 ppm is consistent with outdoor air.
- **V is poorly identified.** It ranged 23–545 m³ depending on the scheme, so I don't interpret it physically.
- **g has no effect on the predictions.** They are exactly invariant to g, which only rescales V and Q.
- **Realism checks.** Synthetic rooms with known parameters were recovered within 5 % for Q. The real ACH of about 2 per hour is plausible for a workshop.

**"How did you ensure numerical stability in your implementation?"**
- Explicit Euler's amplification factor is |1 − Δt/τ|. It is stable only for Δt < 2τ (and non-oscillatory for Δt < τ). I demonstrated both limits numerically.
- The production solver is the *exact* solution for piecewise-constant occupancy, `C_{k+1} = aC_k + (1 − a)C_ss`, with `a = exp(−Δt/τ)` between 0 and 1. It is unconditionally stable.
- The inverse problem is differentiation, which amplifies noise. I used Savitzky–Golay smoothing and checked sensitivity to the scheme.
- Calibration is a well-posed linear least-squares problem, solved with `lstsq` (SVD-based), and it rejects non-physical results.

**"Describe any convergence criteria you used. Why are they appropriate?"**
- **Solver.** I verified the observed order of accuracy against the analytical solution as the step halves (Euler → 1, RK4 → 4). That is the correct criterion for a time integrator, and it showed the code implements the method correctly.
- **MLP.** Early stopping: training stops after 20 epochs without validation improvement (validation fraction 15 %). This guards against overfitting rather than demanding loss convergence.
- **Least squares.** No iteration is needed, because it is a closed-form solve.

**"What are the main sources of numerical error in your solution, and how did you quantify or control them?"**

| Error source | How it was quantified or controlled |
|---|---|
| Interpolation of irregular CO2 | Hold-out test: linear MAE 7.9 ppm, best of 4 methods |
| Finite-difference derivative (truncation vs noise amplification) | Scheme × grid sweep; smoothing |
| Sensor noise biasing the calibration (errors-in-variables) | Synthetic noise sweep: 0 → 50 ppm took MAE from 0.10 to 0.60 people and moved the V error from +23 % to −46 % |
| Time misalignment between interval targets and instantaneous CO2 | Found synthetically and fixed by centring |
| Model-form error (single zone) | The dominant error on real data; seen as under-predicted peaks |

**"How did you verify that your numerical method was implemented correctly (e.g. comparison with analytical solutions, manufactured solutions, benchmark problems)?"**
- Analytical solution for constant occupancy, with convergence orders.
- Manufactured data: simulate with known V, Q, C_out and N(t), then check that the inverse recovers N (±0.05) and calibration recovers the parameters (within 5 %).
- Vectorised implementations checked against loop references (to 1e-9 / 1e-10).
- A hand-calculated aggregation case.

All of these are automated in pytest.

**"Which part of your solution is most vulnerable to numerical failure, and how would you mitigate that risk?"**
The derivative dC/dt. Differentiating noisy, interpolated, 10-min data amplifies noise, and gaps break it. Mitigations already in place:
- smoothing (Savitzky–Golay);
- per-run filtering around gaps;
- gap masking (no estimates more than 45 min from data);
- rejecting non-physical calibrations.

Next step: a Kalman filter or output-error fit that estimates N without explicit differentiation.

---

## C. Likely follow-up questions

- **Why not random K-fold?** Neighbouring 15-min samples are strongly correlated, so random splits put near-duplicates in train and test and inflate scores. Blocked weeks with 1-day buffers avoid that.
- **Why were the chronological R² values negative?** The test week was in the exam period: mean occupancy 0.8 people, std 0.65. R² is relative to that tiny variance, so it is meaningless there (baseline R² −23). MAE was fine (forest 0.65), and the schedule baseline failed (2.42), which is the point.
- **How do you know the sensor and counter are in the same space?** I don't know it physically; it is inferred. The residual correlation (after removing the weekly pattern) is 0.65, versus at most 0.41 for every other pair. Controls show much smaller gains. It should be confirmed on site, and that is a stated limitation.
- **Why does the MLP not beat linear regression?** There are 2 600 samples and about 20 tabular features, and the relationship is close to linear in CO2 level. The extra capacity isn't usable and adds variance.
- **What is arithmetic intensity, and why does it matter here?** It is FLOPs per byte moved.
  - Physics estimator: ≈ 0.5–0.9 FLOP/B. Machine balance is ≈ 22 FLOP/B (188 GFLOP/s dgemm ÷ 8.4 GB/s), so it is memory-bound. The roofline bound is ≈ 2 ns/sample; measured ≈ 8.3 ns because NumPy makes temporaries.
  - MLP: ≈ 37 FLOP/B, so it is compute-bound when batched, and needs ~500× more FLOP per estimate.
  - Theoretical peak: 4 cores × 2.1 GHz × (512/64) × 4 = 269 GFLOP/s.
- **What optimisation did you do?** The exact solution was rewritten as an IIR filter via `scipy.signal.lfilter`: 40× faster than the Python loop, about 260× faster than RK4. The headcount aggregation was rewritten as a cumulative-integral evaluation, taking 2 M events in 0.1–0.4 s. Profiling showed run time is dominated by SVR and forest fitting, not the numerics.
- **What would you do with more time?** Confirm co-location and add rooms (leave-one-room-out), add a two-zone or delay model, and use Kalman-filter estimation with uncertainty bounds.
- **Why balanced accuracy, not F1?** 77 % of intervals are occupied, so always saying "occupied" already gives F1 = 0.87. Balanced accuracy averages the hit rates for occupied and vacant intervals, so any constant guess scores 0.5.
- **Isn't choosing the best-correlated pair and then evaluating on it biased?** Yes, slightly optimistic. I chose 1 of 10 pairs using the same labels. With only 25 days I couldn't spare a separate selection period. The controls show that the gain is specific to that pair, but the absolute R² should be read as an upper estimate.
- **How did you use AI, and how do you know it's right?** See `AI_USE.md` §4–5. Know 2–3 of the AI's errors and how they were caught.

---

## D. Things to be able to derive on a whiteboard

1. Steady state: `C_ss = C_out + 1e6 g N / Q`. Example: 10 people, Q = 0.125 m³/s gives +416 ppm.
2. The exact discrete update and why `0 < a < 1` makes it stable.
3. Euler amplification factor `1 − Δt/τ`, and the stability limit `Δt < 2τ`.
4. Calibration as linear least squares: `N = a·dC/dt + b·C + c`, so `V = 1e6 g a`, `Q = 1e6 g b`, `C_out = −c/b`. This shows why predictions don't depend on g.
5. Central-difference truncation error O(Δt²) vs backward O(Δt).
