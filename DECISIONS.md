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
