# Design decisions

Each entry records an ambiguity in the spec, the choice made, and why.

## Data

1. **Default marks scheme: Assignment 10, MST-1 15, MST-2 15, End Term 60 (internal 40 + external 60 = 100).**
   A common Indian university split. Every maximum is editable in the sidebar, and the app
   adjusts it automatically when the data clearly uses a different scheme (see 3).

2. **Absent ("AB", "Absent", "-") is a flag plus a blank value, never a stored zero.**
   Averages of a component skip absentees. When computing internal/total marks an absent
   component counts as 0 marks earned, because that is how a result sheet is computed.
   Being absent in the end-term exam is always a Fail.

3. **Max marks are inferred when more than 2% of a column exceeds the configured maximum.**
   Header hints like `MST-1 (24)` win. Otherwise, the 99.5th percentile is rounded up to a
   "nice" maximum (24, 25, 30, ...). A single typo (51 out of 15) does not trigger this; it
   is flagged as out-of-range instead.

4. **Out-of-range values are blanked and flagged, not clipped.**
   The true value is unknown (51 could be 15 or 5), so guessing would silently corrupt results.

5. **Pass/Fail is recomputed from scores with the sidebar pass mark (default 40%).**
   The sheet's own result column is kept as `result_reported`. The pass-mark slider must
   change results live, which a stored result column cannot do.

6. **Score % uses Total / max total; falls back to internals, then end term.**
   This lets the same app work mid-semester (only MSTs entered) and after finals.

7. **Single sheet → single marks scheme.** Subjects with different schemes need separate
   uploads. Supporting per-subject schemes was judged not worth the UI complexity.

8. **Blank section → "Unassigned"; no subject column → subject "General"; no roll numbers →
   IDs generated from names.** The app should keep working on partial sheets and say so in
   the Data Quality tab rather than refusing the file.

9. **Sample data: 180 students, 5 subjects, seed 42, ~10.6% fail rate.** Section B and
   Discrete Maths are deliberately weaker so the dashboard has a story to tell.

## Interface

10. **Streamlit, single codebase.** An autonomous build can test it end to end (AppTest + headless
    browser); a separate React frontend would double the surface and the failure points.

11. **Tabs use `on_change="rerun"` with a key.** Only the open tab executes (faster), and the app can
    switch tabs programmatically: selecting a student in a table opens the Student Profile tab.

12. **Drill-down is driven by chart clicks *and* selectboxes.** Clicks are the fast path; the selectboxes
    above the student table show the current drill-down and work on touch screens and with keyboards.

13. **Colours follow a validated, colour-blind-checked palette.** Scores are blue, fail rates orange,
    pass/fail and risk use reserved status colours that always come with a text label or icon.
    Section colours follow the section (A is always blue), even when filters hide others.

14. **MST change tolerance = 10 percentage points** (1.5 marks out of 15). Smaller tolerances treat
    single-mark noise as a trend.

## AI analyst

15. **Any OpenAI-compatible endpoint, through the OpenAI SDK.** One code path covers OpenAI, Gemini,
    Groq, OpenRouter and local Ollama. Default models (checked against provider docs, October 2026):
    `gpt-5.4-mini`, `gemini-3.8-flash`, `openai/gpt-oss-120b`, `qwen3:8b` (Ollama). `temperature` is never
    sent because reasoning models reject it.

16. **The agent analyses the data in view (sidebar filters applied).** "What you see is what the AI sees";
    the scope is stated in the prompt and above the chat. IDs come from the *unfiltered* data so they stay
    stable when filters change.

17. **Pseudonymise, don't just drop names.** Students become S001, S002, … (sorted by roll number). Full
    names and roll numbers of 5+ characters typed into a question are masked too; shorter roll numbers ("12")
    are left alone because they collide with ordinary numbers and identify no one. Answers, tables and chart
    labels are unmasked only for display.

18. **"Max 3 consecutive error retries" = first attempt + 3 fixes.** The agent gives up after the 4th
    consecutive failed tool call. Step limit is 6 model calls; on the last one the model is told to answer.

19. **Plain-text replies count as an answer.** Some models answer without calling `final_answer`; rejecting
    that would waste steps. Empty replies get a nudge instead.

20. **Sandbox = AST allow-list + separate process.** Validation rejects imports, dunder access, eval-like
    calls, file/network I/O and getattr tricks; the worker then runs with restricted builtins, an empty
    environment (no API keys), CPU/memory/file-size limits and a wall-clock alarm. DuckDB runs with external
    access disabled. Results return as JSON, never pickles.

21. **Follow-ups keep the last 4 Q&A turns** (question, answer and the code used), not the full tool
    transcripts, to keep prompts small.

## At-risk students

22. **Risk unit = student; ML unit = student-subject.** The model predicts failing a subject from attendance,
    assignment, MST-1 and MST-2 (never end-term marks, which would leak the answer). A student's model
    probability is their highest across subjects they have not already failed.

23. **Rule points (calibrated on the sample).** Failed subject 30 (max 60), internals below pass 20 per
    subject (max 40), attendance below requirement 20 (30 if 15+ below), sharp MST-2 drop (20+ points) 10 per
    subject (max 20), passing by under 10 points in 2+ subjects 10, absent in an MST/assignment 10.
    High ≥ 50, Medium ≥ 25. One signal alone stays Low; a fail is Medium; a fail plus another signal is High.

24. **Model choice by mean F1, ties broken by ROC-AUC,** across stratified k-fold CV (k = min(5, fails)).
    Both candidates use `class_weight="balanced"`. Probabilities shown for training rows are out-of-fold.
    Model levels: ≥ 0.75 High, ≥ 0.5 Medium (balanced training inflates probabilities, so 0.5 is the decision
    boundary). Final level = the higher of the rule and model levels.

25. **Train automatically on final-results sheets; save only on request.** Saving writes
    `models/risk_model.joblib` (configurable with `GRADELENS_MODEL_DIR`) and is then used for mid-semester
    sheets. Models are never loaded from uploads, because unpickling an untrusted file can run code.

## Reports

26. **Reports cover the data in view** (sidebar filters), like every other tab. The scope is printed on the
    cover and in the file name.

27. **Charts in reports use matplotlib, fonts come from matplotlib.** Plotly's static export needs a browser
    (kaleido), which breaks in slim containers. DejaVu Sans (bundled with matplotlib) gives the PDF symbols
    such as ≥ and → with no system fonts installed.

28. **The PDF lists the top 40 at-risk students; the Excel sheet lists everyone.** The PDF is for reading,
    the workbook for working.

29. **"Include student names" off = shareable copy.** Names and roll numbers are removed from both files and
    the PDF's at-risk list becomes counts per section.

30. **The executive summary is template-written by default and editable.** "Write with AI" sends only
    aggregate facts (never names). An edited summary is kept when filters change; "Use automatic text"
    restores the generated one.

31. **CO attainment in reports** comes from the CO tab's latest result; for the sample data it is computed
    automatically with the default OBE thresholds.

32. **Reports are generated on demand** (a button), not on every rerun, and the download buttons disappear
    when inputs change, so a stale file is never downloaded by mistake.
