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
    such as ≥ and → with no system fonts installed. Indian scripts use bundled Noto fonts (see 52).

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

## Deployment

33. **Exact pins for direct dependencies only** (`requirements.txt`), verified by a clean, wheels-only
    install (`pip install --only-binary=:all:`, the same command as the Dockerfile) plus `pip check`. Dev tools
    live in `requirements-dev.txt`.

34. **Container runs as a non-root user (UID 1000, see 44)** with a HEALTHCHECK written in Python (curl is not in the
    slim image). Sample data is generated at build time; `/app/models` is a Compose volume so a saved at-risk
    model survives restarts.

35. **`.env` is optional and never overrides real environment variables.** The same variables work in a
    shell, `.env`, Docker Compose and Streamlit secrets.

36. **The run scripts reinstall only when `requirements.txt` changes** (they keep a copy of the last installed
    file in `.venv`), so the second start is instant.

## Round 2 (bug hunt) decisions

37. **Maxima follow the sheet.** With automatic maxima, internal and total maxima count only the components
    present (a sheet without an assignment column is out of 30 internal marks, not 40). Maxima typed in the
    sidebar always win.

38. **A whole class marked absent is a data issue.** When ≥ 80% of a subject's students (and at least 5) are
    "AB" for one test, GradeLens warns in Data Quality ("was the test held?") and does not count it as an
    individual risk signal. Scores are not changed: the teacher decides.

39. **Exports never contain live formulas.** Excel cells that start with "=" are stored as text; CSV cells
    starting with = + - @ get a leading apostrophe (plain numbers excepted). CSVs are written with a BOM so
    Excel shows names like "Zoë" correctly.

40. **Section labels keep their punctuation** ("AI&ML", "CSE-A"); only a "Section/Sec" prefix and stray edge
    punctuation are removed.

41. **Masking skips names that identify no one:** under 3 letters, or equal to a subject/section/semester value.
    Masking those would corrupt ordinary questions ("Section A").

42. **The agent forgives common model habits** (fenced code, imports of provided modules, code sent as text,
    alternative argument names) without loosening the sandbox: any other import or blocked call still fails.

43. **Optional features fail softly.** Model training or risk scoring errors fall back to rules (then "–"),
    report errors are shown as messages; the dashboard and every other tab keep working.

44. **Hosting: about 1 GB of RAM.** Free options are Streamlit Community Cloud and Hugging Face Spaces; the Render
    blueprint uses the 2 GB plan. The image listens on `$PORT` and runs as UID 1000.

## Round 3 (scale, browser and container) decisions

45. **Excel reports are streamed** (openpyxl write-only mode with `lxml`): memory stays flat however many students
    there are. The cost is no merged cells, so the executive summary is written one sentence per line.

46. **The risk model trains on at most 10,000 records**, a stratified sample when there are more. Four features
    stop improving long before that, and slider moves on big sheets stay interactive. Records outside the sample
    are scored by the trained model, which is a genuine out-of-sample prediction. The At-Risk tab says when a
    sample was used.

47. **Each cached step is keyed only on the settings it reads.** Cleaning depends on max marks, derived columns
    on max marks and the pass mark, risk on everything. Caches hold a few entries each and expire after an hour,
    so exploring sliders cannot grow memory without bound.

48. **Widgets are reset by assigning their defaults**, from one table in `ui/state.py`. Deleting a widget's key
    only resets the server's copy; the browser sends the old value back on the next click. Widgets with state
    take their initial value from session state, never from `default=`/`value=`.

49. **An interrupted AI question is marked, not hidden.** Any click reruns the app and stops the agent; the
    question stays with "Stopped before finishing" and an **Ask again** button, so nothing silently disappears.

50. **XSRF protection is turned off only on Hugging Face Spaces** (detected by `SPACE_ID`). Spaces embeds the app
    cross-site, where the XSRF cookie is blocked and every upload fails. A global switch would weaken every other
    deployment.

51. **The image creates its group explicitly and disables the file watcher.** Whether `useradd` makes a group is
    a distro default; without `watchdog` Streamlit would poll its own source files forever in production.

52. **Indian scripts in the PDF: bundled Noto fonts, per-script runs, HarfBuzz shaping.** Nine scripts (2.9 MB of
    OFL fonts) cover the languages of most Indian colleges. Text is split into runs so each font only draws
    characters it has, and ReportLab is made to shape each run with its own font (it shapes a whole word with
    the first font, which garbled punctuation next to Indian text). Chart labels get the glyphs but not shaping
    (matplotlib cannot shape). Urdu and other right-to-left scripts are out of scope.
