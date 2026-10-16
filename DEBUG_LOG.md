# Debug log

Notable bugs found while building and testing, with root cause and fix.

| # | Phase | Symptom | Root cause | Fix |
|---|-------|---------|------------|-----|
| 1 | P1 | Synthetic fail rate was 22%, not the ~10% the spec asks for | Score intercept too low for the variance of the latent skill | Calibrated the intercept (57 → 66) against measured fail rates; now 10.6% |
| 2 | P1 | `MST (2)` would have been read as "max marks = 2" and the header stripped to `MST` | Max-mark hint regex accepted any number in brackets | Hints only count from 5 to 1000 marks, and are only stripped from the name when they count |
| 3 | P1 | `test_write_sample_files` failed on `co_mapping.csv` (82 bytes) | Test assumed every file > 100 bytes; the CO map is legitimately 6 short rows | Test now checks real content (row counts) instead of an arbitrary size |
| 4 | P2 | Every CO attainment call crashed with "truth value of a Series is ambiguous" | Operator precedence: `a \| b \| c.any()` applied `.any()` to the last term only | Build the full mask first, then call `.any()`; added a regression test for a blank / missing CO |
