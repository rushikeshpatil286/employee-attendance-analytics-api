# Employee Attendance & Analytics API

FastAPI + MongoDB implementation of the HROne assignment contract (`openapi.yaml`, rules R1-R10). All application code is in `app/main.py`.

## Run

Requires Python 3.11+ and MongoDB 6.0+ (developed against 7.0).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017"            # a local .env with the same names also works
export MONGO_DB="attendance_db"
python sample_seed.py                                   # optional: loads the nine sample documents
uvicorn app.main:app --port 8000
```

Indexes are created automatically at startup. Interactive docs: <http://localhost:8000/docs>. Readiness: `GET /health`.

## Tests

The tests need a reachable MongoDB and create (and drop) a throw-away database named `hrone_test_<random>`.

```bash
pip install -r requirements-dev.txt
MONGO_URI=mongodb://localhost:27017 pytest -q
```

## Status

All endpoints in `openapi.yaml` are implemented, including `GET /admin/explain/{endpoint}`.

Things worth knowing:

* Analytics read the stored `late_minutes` / `work_hours` / `overtime_minutes` / `half_day` values, as the contract requires.
* Rates and averages are rounded half-up using integer arithmetic (MongoDB's `$round` is banker's rounding).
* `PATCH` uses optimistic concurrency and retries up to 5 times; if it still loses the race it returns 409.
* Punch-out picks the most recent record with `punch_in <= punched_at`. If there is none but a later punch-in exists, it returns 422 (punch-out before punch-in); with no punch-in at all it returns 404.
* The department trend needs one extra small query (the department's employees) before the aggregation, to supply headcount per day.
* If MongoDB fails mid-request the API answers 503, not 500. The app does not start if MongoDB is unreachable at startup.
