# MMA3001 Project Primer: Virtual Occupancy Sensing (read this first)

This is a learning guide to the project "Virtual Occupancy Sensing from Building CO₂ Data: a Physics-Based Mass Balance versus Machine-Learning Regression" by Samuel Roberts (MMA3001 Numerical Methods and Machine Learning, Monash University, 2026). It explains the project from top to bottom for someone who did not build it. The other sources in this notebook hold the detail: the report, the decision log, the interview preparation notes, the AI-use statement, the source code and the result tables.

The code and most drafts were written by an AI assistant (Claude, through Claude Code) overnight, with Samuel setting the goal, choosing the dataset and supplying the data. Samuel now needs to understand every part well enough to present it in 5 minutes and answer 3 minutes of questions.

---

## 1. The one-paragraph story

Buildings waste energy ventilating empty rooms. If you knew how many people were in each room, you could ventilate only as much as needed; this is called demand-controlled ventilation. People-counters are expensive and rare, but cheap CO₂ sensors are common. People breathe out CO₂, so a room's CO₂ level should reveal how many people are in it. The project asks whether the CO₂ sensors already installed in Monash buildings can act as a "virtual occupancy sensor". It tests a simple physics model, a CO₂ mass balance with only three unknown numbers, against five machine-learning models and a timetable-style baseline. Each is scored against real people-counter data. The answer is a qualified yes. The physics model gets R² ≈ 0.47 and an average error of about 1.6 people. The best machine-learning model, a random forest, gets R² ≈ 0.51 and 1.5 people. The timetable baseline explains nothing (R² ≈ 0). The physics model also needs far less training data.

## 2. The data, and why the audit changed everything

The teaching team supplied two exports from Monash Smart Infrastructure (MSI):

- **Environmental sensors:** CO₂, temperature and humidity; 179,447 rows; 5 sensors.
- **Occupancy counters:** people counts; 9.09 million event rows; 5 spaces.

The plan was to match each sensor to the room its counter is in, using the location sheet. The audit found three problems.

1. **76 % of environmental rows were duplicates.** Three of the five sensors were "frozen": they re-sent the same single reading tens of thousands of times. Only two sensors had live data.
2. **The location sheet didn't link any working sensor to any counted space.** Its occupancy tab is even labelled as having invalid IDs.
3. **The counters were reliable but went silent often**, with thousands of gaps longer than 30 minutes.

**The fix: infer the pairing from the data itself.** For every sensor–space combination, correlate CO₂ with headcount, after first removing each series' average weekly pattern (its "time-of-week mean"). That removal step is the clever part. Without it, any CO₂ sensor and any counter would correlate simply because both are high during office hours. With it, only rooms whose CO₂ rises and falls with *their own* unusual occupancy correlate.

- **Primary pair:** one combination stood out at r = 0.65: sensor `…0869` (room G.38, Building 60) with counter space `3e5a14df`, over about 25 days of overlapping labelled data.
- **Control pairs:** two weaker combinations (r = 0.41 and 0.33) are kept as a negative check. If a model does almost as well on a sensor that is *not* in the room, the "room-level" claim would be false.

## 3. Turning raw data into a clean table (pre-processing)

Every model works on a uniform 15-minute grid. Two different numerical ideas are needed.

- **CO₂, temperature and humidity are continuous signals, so they are interpolated.** Four methods were compared by hiding 10–50 % of real readings and predicting them: linear, PCHIP, cubic spline and nearest-neighbour. Linear won (about 8 ppm error). The cubic spline was worst, because it overshoots on noisy data. Gaps longer than 45 minutes are left empty rather than invented.
- **Headcount is a step signal reported at events, so it is integrated, not interpolated.** The count stays constant until the next event (a "zero-order hold"). The 15-minute value is the time-average of that step function, which is an integral divided by the duration. The first version looped over events in Python. The final version uses a vectorised cumulative sum, handling about 2 million events in under half a second. A counter silent for more than 30 minutes is treated as offline, not as "still the same count".
- **Timing alignment (a subtle bug that was found and fixed).** A CO₂ reading is a snapshot at time t. The headcount value is an average over an interval. If you label the average over [t, t+15 min) as "time t", occupancy appears to lead CO₂ by 7.5 minutes. On synthetic data, this made the calibrated room volume 31–52 % too large. The fix is to centre the interval on t, using [t−7.5 min, t+7.5 min).

## 4. The physics model (the heart of the project)

Treat the room as one well-mixed box of air:

**V · dC/dt = 10⁶ · g · N − Q · (C − C_out)**

In words: the rate of change of CO₂ in the room equals what the people breathe out, minus what the ventilation flushes away.

| Symbol | Meaning |
|---|---|
| C | CO₂ concentration in the room (ppm) |
| V | effective room volume (m³) |
| N | number of people (what we want) |
| g | CO₂ breathed out per person, 5.2 × 10⁻⁶ m³/s for a sitting adult |
| Q | outdoor-air ventilation flow rate (m³/s) |
| C_out | CO₂ of the incoming supply air (ppm) |

The 10⁶ converts a volume fraction into ppm. The time constant τ = V/Q (typically 20–60 min) sets how quickly the room responds.

The project uses this one equation three ways.

1. **Forward (simulation):** given N, compute C over time. It is solved three ways: explicit Euler (simple, first-order accurate), RK4 (fourth-order accurate) and the exact solution, which is valid when N is constant over each step. The exact solution turns out to be a first-order IIR digital filter, so `scipy.signal.lfilter` runs it in compiled code. That is 40× faster than a Python loop of the same formula and about 260× faster than the RK4 loop.
2. **Inverse (the estimator):** rearrange to get people from CO₂:
   **N = (V · dC/dt + Q · (C − C_out)) / (10⁶ · g)**.
   This needs the slope dC/dt, which is approximated by finite differences: backward, central, or Savitzky–Golay. Savitzky–Golay fits a small quadratic to 5 neighbouring points and differentiates it, which is less noisy.
3. **Calibration (finding V, Q and C_out):** expand the inverse equation to get N = a · dC/dt + b · C + c. This is *linear* in the unknowns a, b and c, so one ordinary least-squares solve gives them, with no iteration and a unique answer, in 2 ms. Then V = 10⁶ g a, Q = 10⁶ g b and C_out = −c/b.

**Neat consequence:** the predicted N does not depend on the assumed value of g. Changing g just rescales a and b, so V and Q change but N does not. This was checked numerically to machine precision.

**What the calibration found:** Q ≈ 0.25 m³/s (about 250 L/s) is stable across every setting tried, so it is a credible ventilation estimate. V is *not* stable (anywhere from 23 to 545 m³), so it should not be read as the physical room volume. The reason is that the steady-state term dominates and the slope term is poorly determined with 10-minute sampling.

## 5. The machine-learning models

All models come from scikit-learn and use only current and past information ("near-causal" features), so they could run live. The features are:

- CO₂ and its change, at lags 0–3 (now, 15, 30 and 45 min ago);
- CO₂ above a rolling 24-hour baseline;
- temperature and humidity, and their changes;
- hour of day, encoded as sine and cosine so 23:00 is close to 00:00;
- a weekend flag.

| Model | One-line idea |
|---|---|
| Time-of-week baseline | Predict the average count for that hour of that weekday; no sensor at all. It sets the bar. |
| Ridge regression | Linear model with a penalty that stops large coefficients. |
| Decision tree | A series of yes/no splits on features. |
| Random forest | 200 trees on random subsets, averaged; robust and usually strong on tabular data. |
| SVR | Support-vector regression with an RBF kernel. |
| MLP (64-32) | A small neural network with two hidden layers, ReLU, Adam and early stopping. |
| Hybrids | The forest and MLP with the physics estimate added as an extra feature. |

## 6. How the models were tested (validation), and why it matters most

**Verification asks: does the code solve the equations correctly?** The project generated fake rooms where the true answer is known.

- **Order of accuracy:** halving the step size should cut the error by 2× for Euler and 16× for RK4. Measured orders were 0.97 and 4.14, against theory 1 and 4.
- **Stability:** Euler applied to this equation multiplies the error by (1 − Δt/τ) each step. It oscillates once Δt > τ and blows up once Δt > 2τ. Both were reproduced numerically. This is one reason production code uses the exact solution.
- **Calibration:** on synthetic rooms, calibration recovered V and Q within about 5 % (4–17 % for V in the end-to-end test).
- **Tests:** 27 automated pytest tests cover all of this, and an HTML test report is in the repo.

**Validation asks: does the model match reality?** Choosing how to split the data was the most important decision.

- **Random K-fold is wrong for time series.** Neighbouring 15-minute samples are nearly identical, so a random split puts near-copies of the test data in training. That inflates the scores.
- **The main protocol is blocked leave-one-week-out cross-validation.** Each week is predicted by a model trained on all the other weeks, with a 1-day gap on either side so lagged features cannot leak. Predictions from all folds are pooled and scored once: 7 folds and 2,414 intervals.
- **A chronological 60/20/20 split was also run.** Its test week happened to fall in the June exam period, when the room was nearly empty, so R² there is meaningless. It is reported as a "regime shift" scenario instead. The timetable baseline failed badly (MAE 2.42) because it still expected teaching-week crowds. The CO₂ models adapted (forest MAE 0.65, physics 0.85). This is the practical argument for sensors over timetables.

**Metrics:**

- MAE: the average error in people, the number an operator cares about.
- RMSE.
- R²: the fraction of variance explained; 0 means no better than predicting the mean.
- Bias.
- Balanced accuracy for occupied versus vacant.

F1 was dropped as evidence of skill. 77 % of intervals are occupied, so always guessing "occupied" already scores F1 = 0.87. Balanced accuracy gives 0.5 to any constant guess.

## 7. The results in one table (blocked CV, primary pair)

| Model | MAE (people) | R² | Balanced acc. |
|---|---|---|---|
| Timetable baseline | 2.35 | −0.05 | 0.52 |
| Physics (3 parameters) | 1.60 | 0.47 | 0.56 |
| Ridge (linear) | 1.63 | 0.45 | 0.65 |
| SVR | 1.68 | 0.40 | 0.61 |
| MLP | 1.71 | 0.37 | 0.57 |
| Random forest | 1.50 | 0.51 | 0.55 |
| Forest + physics feature | 1.50 | 0.51 | 0.56 |

How to read it:

1. **The sensor carries real information.** Every CO₂ model beats the timetable by 0.6–0.8 people. On the control pairs the gain is much smaller (forest R² 0.27/0.40 against baseline 0.15/0.26).
2. **Three physics parameters nearly match a random forest.** The physics model has the smallest bias, fits about 300× faster, and reaches R² 0.45 with just 4 days of labels. The forest needs about 14 days.
3. **The neural network did not help.** With about 2,400 tabular samples, an MLP is no better than linear regression.
4. **Every model under-predicts peaks above about 10 people.** The room is large, so 15 people raise CO₂ by only 100–150 ppm.
5. **Vacancy detection is weak.** Balanced accuracy is at most 0.65, because CO₂ lingers for tens of minutes after people leave.

## 8. Computational performance (the "numerical methods" angle)

**Machine:** 4 vCPUs at 2.1 GHz with AVX-512.

- Theoretical peak is about 269 GFLOP/s; a measured matrix multiply reached 188 GFLOP/s.
- Memory bandwidth is about 8.4 GB/s.
- So the machine can do about 22 floating-point operations per byte it reads (its "machine balance").

**Physics estimator:** about 0.5–0.9 operations per byte, far below machine balance. It is **memory-bound**: speed is limited by moving data, not by arithmetic.

**MLP inference:** about 37 operations per byte, so it is **compute-bound**. It needs about 500× more FLOPs per estimate.

**Profiling:** most of the experiment's run time is spent fitting SVR and forests inside cross-validation, not in the numerical methods. At building scale, every method is cheap. The real deciding factors are accuracy, how much labelled data is needed, and interpretability.

## 9. The bugs and mistakes found along the way (great interview material)

| Problem | Symptom | Fix |
|---|---|---|
| Timestamp unit bug (pandas 3 stores datetimes at a different resolution) | Times were off by 1000× | A `unix_seconds` helper that is robust to resolution |
| Interval labelled at its start, not its centre | Room volume biased by about +40 % | Centre the target interval |
| Splitting across a long unlabelled stretch | Every model had negative R² | Trim to the labelled overlap before splitting |
| Chronological test week fell in exams | R² meaningless | Blocked CV as the main protocol |
| F1 score | Looked impressive but a constant guess scores 0.87 | Use balanced accuracy |
| Savitzky–Golay over gaps | NaNs crashed calibration | Filter each contiguous run separately |
| IIR filter initial state wrong | Simulation started wrong | Correct `zi=[c0]` |
| Daylight-saving transition | Crash building the time grid | Build the grid in UTC |

An independent code review (a second AI agent) also checked the work, and its findings were fixed or disclosed.

## 10. How the code is organised

All code is in `src/occupancy/`; the scripts in `scripts/` run the experiments.

| File | What it does |
|---|---|
| `io.py` | Loads raw JSON/CSV, flags invalid readings, aggregates headcount onto the grid (vectorised integral) |
| `resample.py` | Interpolates the environmental data onto the uniform grid; hold-out comparison of interpolation methods |
| `timeutil.py` | Converts timestamps to seconds, robust to the pandas datetime resolution |
| `physics.py` | The mass balance: `RoomParams`, Euler/RK4/exact simulation, the fast `lfilter` version, finite-difference derivatives, the inverse estimator, least-squares calibration |
| `features.py` | Builds the ML features (lags, differences, baseline, time encodings) |
| `models.py` | Timetable baseline, physics estimator wrapper, ML model factory |
| `evaluate.py` | Metrics, chronological split, blocked weekly folds |
| `pipeline.py` | Puts it together: prepare a room, fit and predict, score on a common mask, run CV |
| `synthetic.py` | Generates fake rooms with known truth for verification |

| Script | What it does |
|---|---|
| `ingest.py` | Audit and sensor–space association |
| `run_experiment.py` | Main results |
| `sensitivity_synthetic.py` / `sensitivity_real.py` | Sensitivity studies |
| `profile_performance.py` | FLOPs, timing, roofline |
| `make_figures.py` | All figures |

**Running it:** `make all` reproduces everything from the raw data. That covers the tests, the HTML documentation (made with pdoc from the docstrings) and the report PDF.

**Repository files:**

- `README.md`: overview and results.
- `DECISIONS.md`: every design decision, D1–D21, with reasons.
- `VIVA_PREP.md`: presentation plan and question answers.
- `AI_USE.md`: AI-use statement.
- `reports/report.pdf`: the report, 10 pages of body.
- `notebooks/walkthrough.ipynb`: a guided notebook.

## 11. Limitations to own in the interview

1. **The evidence base is narrow.** There is one statistically inferred sensor–room pair over 25 days. The pairing was chosen as the best of 10 using the same labels the models are scored on, so the R² is slightly optimistic.
2. **The single well-mixed zone is an approximation.** A sensor high in a big room sees diluted, delayed CO₂.
3. **There are no uncertainty estimates.** A real ventilation controller should have them.
4. **The ground truth (the counter) has its own unknown error,** and night-time labels are sparse.

Next steps, in priority order:

1. Confirm the pairing on site.
2. Get more rooms, to allow leave-one-room-out testing.
3. Try a two-zone model.
4. Build a Kalman-filter estimator with uncertainty bounds.

## 12. Things you must be able to explain without notes

1. Derive the mass balance and the inverse formula for N.
2. Why calibration is a *linear* least-squares problem, and how V, Q and C_out come out of a, b and c.
3. Why the predicted N does not depend on g.
4. Euler's stability limit, Δt < 2τ, from the amplification factor |1 − Δt/τ|.
5. Why random K-fold leaks on time series, and what blocked CV with buffers does.
6. Why the headcount is integrated but CO₂ is interpolated.
7. Why linear interpolation beat the cubic spline.
8. Why the time-of-week mean is removed before correlating sensors with spaces.
9. Memory-bound versus compute-bound, and what arithmetic intensity and machine balance mean.
10. Why an MLP didn't beat linear regression here.
11. What you personally checked, changed or rejected in the AI's work.

## Glossary

| Term | Meaning |
|---|---|
| ACH | Air changes per hour, Q/V × 3600 |
| Arithmetic intensity | FLOPs per byte of memory traffic |
| Balanced accuracy | Average of the true-positive rate and true-negative rate |
| DCV | Demand-controlled ventilation |
| Errors-in-variables | Least squares underestimates coefficients when an input (here the noisy derivative) is itself measured with error |
| IIR filter | Infinite-impulse-response filter; a recurrence y_k = a·y_{k−1} + b·x_k |
| PCHIP | Piecewise cubic Hermite interpolation; shape-preserving, no overshoot |
| Roofline model | Performance bound = min(peak FLOP/s, bandwidth × intensity) |
| Savitzky–Golay | Smoothing and differentiation by local polynomial fit |
| Zero-order hold | Keep the last value until the next event |
