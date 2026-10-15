# DECISIONS.md

1. **Indexes.** `employees`: unique `emp_code` (identity, create race, lookups); `(department, emp_code)` (filtered, sorted list; trend headcount); `joined_on` (summary without a department). `attendance_logs`: unique `(emp_code, date)` (punch-in race, monthly, lookups); `(date desc, emp_code)` (list sort, leaderboard month range); `(status, date desc, emp_code)` (status-filtered list). Rejected: `(emp_code, punch_in)` for punch-out; per-employee records are few, so the sort is cheap.
2. **Punch-in race.** Both requests read the employee and build the same document, then `insert_one`. The unique `(emp_code, date)` index lets exactly one succeed; the other raises `DuplicateKeyError`, which becomes 409. There is no check-then-insert.
3. **Ties.** The leaderboard uses `$setWindowFields` with `$rank`, so ties share a rank and the next is skipped. `limit` filters `rank <= limit` after ranking, so both tied employees are returned, possibly exceeding `limit`.
4. **Headcount.** The summary starts from `employees` (`joined_on <= month end`), `$lookup`s logs with `$unwind` preserving empty arrays, groups per employee first, then per department, so someone with zero logs is still one row.
5. **100x data.** Pre-aggregate monthly per-employee totals (maintained on write) instead of scanning logs per request.
