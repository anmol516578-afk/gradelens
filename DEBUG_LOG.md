# Debug log

Notable bugs found while building and testing, with root cause and fix.

| # | Phase | Symptom | Root cause | Fix |
|---|-------|---------|------------|-----|
| 1 | P1 | Synthetic fail rate was 22%, not the ~10% the spec asks for | Score intercept too low for the variance of the latent skill | Calibrated the intercept (57 → 66) against measured fail rates; now 10.6% |
| 2 | P1 | `MST (2)` would have been read as "max marks = 2" and the header stripped to `MST` | Max-mark hint regex accepted any number in brackets | Hints only count from 5 to 1000 marks, and are only stripped from the name when they count |
| 3 | P1 | `test_write_sample_files` failed on `co_mapping.csv` (82 bytes) | Test assumed every file > 100 bytes; the CO map is legitimately 6 short rows | Test now checks real content (row counts) instead of an arbitrary size |
| 4 | P2 | Every CO attainment call crashed with "truth value of a Series is ambiguous" | Operator precedence: `a \| b \| c.any()` applied `.any()` to the last term only | Build the full mask first, then call `.any()`; added a regression test for a blank / missing CO |
| 5 | P3 | 40% of student-subjects showed as "declined" from MST-1 to MST-2 | With MSTs out of 15, one mark = 6.7 percentage points, so a 5-point tolerance flagged every one-mark dip | Default tolerance raised to 10 points (1.5 marks); declines now match the genuine large drops in the data (216 vs 216) |
| 6 | P3 | Clicking a bar highlighted it in the browser but the drill-down never happened | A per-bar `marker.opacity` array combined with text labels stops Plotly from emitting the selection event (each alone is fine — found by bisecting in a headless browser) | Highlight the drilled bar with a per-bar colour array instead of opacity |
| 7 | P3 | Clicking a heatmap cell did nothing | Plotly heatmaps emit no selection events in Streamlit at all (confirmed with a minimal app) | An invisible square marker sits on every heatmap cell; it catches the click and carries the tooltip |
| 8 | P3 | Lambdas for chart callbacks would all have read the last chart's state | Python late binding: each lambda captured the loop variable `key`, reassigned for the next chart | Callbacks bound with `functools.partial` |
| 9 | P3 | A corrupt `.xlsx` upload crashed the whole app | The sidebar's sheet lookup ran before the main area's `try/except LoadError` | Sidebar catches `LoadError`; the main area shows the friendly message (AppTest regression test added) |
