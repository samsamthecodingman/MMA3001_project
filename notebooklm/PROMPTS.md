# NotebookLM prompts for the MMA3001 project

Upload sources 00–04 from this folder (and `01_report.pdf`). This file is not a source; copy the prompts into NotebookLM.

## Video Overview (Studio → Video Overview → Customise)

Choose the **Explainer** format and paste:

> Explain this engineering project to the student who will present it but did not write the code. Follow the order of the primer: the problem (demand-controlled ventilation), the data audit and why the sensor–room pairing had to be inferred, pre-processing (interpolation versus integration), the CO₂ mass-balance equation used forwards, inversely and for least-squares calibration, the machine-learning comparison, the validation design (why blocked cross-validation instead of random K-fold), the results table, computational performance (memory-bound versus compute-bound), and the limitations. Show the key equations on screen and explain each symbol. Use the real numbers from the sources. Finish with the five questions an examiner is most likely to ask, with short answers.

For a second, shorter video, choose **Brief** and paste:

> A 3-minute recap of only the results and what they mean: physics versus random forest versus baseline, the control pairs, the regime-shift test, and the limitations.

## Audio Overview (Deep Dive)

> Two hosts walk through this project as if preparing the student for an oral exam. Spend the most time on the physics model (derivation, inverse, linear calibration, why N doesn't depend on g, Euler stability) and on validation design. Have one host play a sceptical examiner who challenges weak points: the single inferred sensor pair, the modest R² of about 0.5, peak under-prediction, and the heavy AI use. The other host answers using evidence from the sources.

The **Debate** format also works with: "Was a 3-parameter physics model a better choice than the random forest for this problem?"

## Chat prompts (paste into the chat panel)

**Study guide:**

> Create a study guide organised by the 12 "must explain" items in the primer, with a worked derivation for each mathematical one.

**Mock interview:**

> Act as my MMA3001 examiner. Ask me one question at a time about this project, mixing the brief's sample questions (from VIVA_PREP) with harder follow-ups. Wait for my answer, grade it out of 5, and tell me what a full-marks answer would include. Do not reveal the answer before I try.

**Code walkthrough:**

> Walk me through the call path of `run_room_cv` in pipeline.py: which functions it calls, in which files, and what each step does to the data.

**Explain a decision:**

> From DECISIONS.md, explain decision D__ (replace with the number): the alternatives considered and why this one was chosen.

**Find weaknesses:**

> List the ten weakest points an examiner could attack, and the best honest answer to each, citing the sources.

Use the Studio buttons for **Flashcards** (focus: equations, numbers and definitions) and **Quiz** (focus: validation and numerical methods). A **Mind Map** of the code structure is also useful.
