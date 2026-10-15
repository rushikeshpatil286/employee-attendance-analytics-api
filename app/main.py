"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience; real environment variables win)

Contract: openapi.yaml (rules R1-R10). Stored shapes: DATA_MODEL.md.
Instants are stored as BSON datetimes (UTC) and exposed as epoch milliseconds. Calendar dates and
shift times are plain strings in IST. Analytics read the *stored* derived fields.
"""
import calendar
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated, Any, Literal, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field, StrictInt, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()  # does not override variables that are already set

# tz_aware=True: PyMongo returns aware UTC datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]


def create_indexes() -> None:
    """Idempotent: creating an index that already exists with the same spec is a no-op."""
    emp, logs = db.employees, db.attendance_logs
    # employees
    emp.create_index([("emp_code", ASCENDING)], unique=True, name="uq_emp_code")
    emp.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="dept_emp_code")
    emp.create_index([("joined_on", ASCENDING)], name="joined_on")
    # attendance_logs
    logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_emp_date")
    logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="date_desc_emp")
    logs.create_index(
        [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="status_date_emp"
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(PyMongoError)
async def mongo_error_handler(_: Request, exc: PyMongoError):
    # Database trouble is a 503, never a stack-trace 500.
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# --------------------------------------------------------------------------- #
# Constants and validated types
# --------------------------------------------------------------------------- #
UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MIN_MS, MAX_MS = 100_000_000_000, 4_102_444_800_000

PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]
NO_PUNCH = ("ABSENT", "LEAVE")
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]

GRACE_SECONDS = 10 * 60
OVERTIME_MIN_MINUTES = 30
HALF_DAY_BELOW = 4.5
MAX_SHIFT = timedelta(hours=24)


def _parse_date(value: str) -> str:
    # fromisoformat alone also accepts "20260105"; require the exact YYYY-MM-DD shape first.
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        raise ValueError("must be a date in YYYY-MM-DD format")
    date.fromisoformat(value)
    return value


def _parse_month(value: str) -> str:
    if int(value[:4]) < 1:
        raise ValueError("year must be >= 0001")
    return value


EpochMs = Annotated[StrictInt, Field(ge=MIN_MS, le=MAX_MS)]  # StrictInt rejects floats/strings/bools
DateStr = Annotated[str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"), AfterValidator(_parse_date)]
MonthStr = Annotated[str, Field(pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$"), AfterValidator(_parse_month)]
HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


def unprocessable(msg: str, loc: Optional[list] = None) -> HTTPException:
    """422 with FastAPI's validation body shape (detail is a list), for cross-field checks."""
    return HTTPException(422, detail=[{"loc": loc or ["body"], "msg": msg, "type": "value_error"}])


# --------------------------------------------------------------------------- #
# Time helpers (R1)
# --------------------------------------------------------------------------- #
def ms_to_dt(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(milliseconds=ms - ms % 1000)


def dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def to_ms(value: Any) -> Any:
    return dt_to_ms(value) if isinstance(value, datetime) else value


def now_dt() -> datetime:
    return ms_to_dt(dt_to_ms(datetime.now(UTC)))


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_overnight(emp: dict) -> bool:
    return _minutes(emp["shift_end"]) <= _minutes(emp["shift_start"])


def attendance_date(punch_in: datetime, emp: dict) -> str:
    """R1: IST calendar date of punch_in; an overnight-shift punch before shift_end belongs to the previous day."""
    ist = punch_in.astimezone(IST)
    day = ist.date()
    if is_overnight(emp) and ist.hour * 60 + ist.minute < _minutes(emp["shift_end"]):
        day -= timedelta(days=1)
    return day.isoformat()


def _at(day: str, hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return datetime.combine(date.fromisoformat(day), time(int(h), int(m)), tzinfo=IST)


def shift_start_at(day: str, emp: dict) -> datetime:
    return _at(day, emp["shift_start"])


def shift_end_at(day: str, emp: dict) -> datetime:
    end = _at(day, emp["shift_end"])
    return end + timedelta(days=1) if is_overnight(emp) else end


# --------------------------------------------------------------------------- #
# Derived fields (R2-R5)
# --------------------------------------------------------------------------- #
def compute_late_minutes(punch_in: datetime, day: str, emp: dict) -> int:
    """R2: late only if strictly more than 10:00 after shift_start; then whole minutes since shift_start."""
    seconds = int((punch_in - shift_start_at(day, emp)).total_seconds())
    return seconds // 60 if seconds > GRACE_SECONDS else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600 to 2 decimals, half-up, using integer arithmetic (no float rounding)."""
    seconds = int((punch_out - punch_in).total_seconds())
    return ((seconds * 100 + 1800) // 3600) / 100


def compute_overtime(punch_out: datetime, day: str, emp: dict) -> int:
    """R3: whole minutes after shift_end, only if >= 30."""
    minutes = int((punch_out - shift_end_at(day, emp)).total_seconds()) // 60
    return minutes if minutes >= OVERTIME_MIN_MINUTES else 0


def derive(status: str, punch_in: Optional[datetime], punch_out: Optional[datetime], day: str, emp: dict) -> dict:
    out = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if status in NO_PUNCH or punch_in is None:
        return out
    out["late_minutes"] = compute_late_minutes(punch_in, day, emp)
    if punch_out is not None:
        out["work_hours"] = compute_work_hours(punch_in, punch_out)
        out["overtime_minutes"] = compute_overtime(punch_out, day, emp)
        out["half_day"] = out["work_hours"] < HALF_DAY_BELOW
    return out


# --------------------------------------------------------------------------- #
# Document <-> API conversion
# --------------------------------------------------------------------------- #
def employee_out(e: dict) -> dict:
    return {
        "emp_code": e["emp_code"],
        "name": e["name"],
        "email": e["email"],
        "department": e["department"],
        "shift_start": e["shift_start"],
        "shift_end": e["shift_end"],
        "joined_on": e["joined_on"],
        "created_at": to_ms(e.get("created_at")),
    }


def record_out(r: dict) -> dict:
    """Missing legacy fields default as DATA_MODEL.md says: late/overtime 0, half_day false, history []."""
    return {
        "emp_code": r["emp_code"],
        "date": r["date"],
        "status": r["status"],
        "punch_in": to_ms(r.get("punch_in")),
        "punch_out": to_ms(r.get("punch_out")),
        "work_hours": r.get("work_hours"),
        "late_minutes": r.get("late_minutes") or 0,
        "overtime_minutes": r.get("overtime_minutes") or 0,
        "half_day": bool(r.get("half_day", False)),
        "history": [
            {
                "at": to_ms(h.get("at")),
                "by": h.get("by"),
                "reason": h.get("reason"),
                "changes": {
                    k: {"from": to_ms(v.get("from")), "to": to_ms(v.get("to"))}
                    for k, v in (h.get("changes") or {}).items()
                },
            }
            for h in (r.get("history") or [])
        ],
    }


def get_employee(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class EmployeeCreate(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=HHMM)
    shift_end: str = Field("18:30", pattern=HHMM)
    joined_on: DateStr

    @model_validator(mode="after")
    def shifts_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInRequest(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutRequest(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None


class RegularizeRequest(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# system / employees
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate):
    doc = body.model_dump()
    doc["created_at"] = now_dt()
    try:
        db.employees.insert_one(doc)  # unique index on emp_code decides races, not a prior find
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: Page = 1, page_size: PageSize = 20):
    q = {} if department is None else {"department": department}
    total = db.employees.count_documents(q)
    cur = db.employees.find(q).sort("emp_code", ASCENDING).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(e) for e in cur], "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# attendance
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest):
    emp = get_employee(body.emp_code)
    pin = ms_to_dt(body.punched_at) if body.punched_at is not None else now_dt()
    day = attendance_date(pin, emp)
    doc = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status,
        "punch_in": pin,
        "punch_out": None,
        **derive(body.status, pin, None, day, emp),
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date) index: exactly one racer wins
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest):
    emp = get_employee(body.emp_code)
    pout = ms_to_dt(body.punched_at) if body.punched_at is not None else now_dt()
    logs = db.attendance_logs

    # Most recent record whose punch_in is at or before punched_at (dates are ordered like punch-ins).
    rec = logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": pout}}, sort=[("date", DESCENDING)]
    )
    if rec is None:
        # Distinguish "punched out before the punch-in" (422) from "never punched in" (404).
        if logs.find_one({"emp_code": body.emp_code, "punch_in": {"$gt": pout}}, projection={"_id": 1}):
            raise unprocessable("punched_at must be after punch_in", ["body", "punched_at"])
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")

    pin = rec["punch_in"]
    if not (pin < pout <= pin + MAX_SHIFT):
        raise unprocessable(
            "punched_at must be after punch_in and within 24 hours of it", ["body", "punched_at"]
        )

    fields = derive(rec["status"], pin, pout, rec["date"], emp)
    fields["punch_out"] = pout
    # The filter re-checks "still open", so of two simultaneous punch-outs only one matches.
    updated = logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None}, {"$set": fields}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return record_out(updated)


def attendance_query(emp_code, date_from, date_to, status) -> dict:
    if date_from and date_to and date_from > date_to:
        raise unprocessable("date_from must not be after date_to", ["query", "date_from"])
    q: dict = {}
    if emp_code is not None:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status is not None:
        q["status"] = status
    return q


ATTENDANCE_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[Status] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    q = attendance_query(emp_code, date_from, date_to, status)
    logs = db.attendance_logs
    total = logs.count_documents(q) if q else logs.estimated_document_count()
    cur = logs.find(q).sort(ATTENDANCE_SORT).skip((page - 1) * page_size).limit(page_size)
    return {"items": [record_out(r) for r in cur], "total": total, "page": page, "page_size": page_size}


def build_correction(rec: dict, emp: dict, body: RegularizeRequest) -> tuple[dict, dict]:
    """Returns (fields to $set, changes for the history entry). Raises 422 on rule violations."""
    status = body.status or rec["status"]
    old_in, old_out = rec.get("punch_in"), rec.get("punch_out")

    if status in NO_PUNCH:
        if body.punch_in is not None or body.punch_out is not None:
            raise unprocessable(f"status {status} cannot carry punch times")
        new_in = new_out = None
    else:
        new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else old_in
        new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else old_out
        if new_in is None:
            raise unprocessable(f"status {status} requires a punch_in")
        if attendance_date(new_in, emp) != rec["date"]:
            raise unprocessable("punch_in must stay on the record's attendance date", ["body", "punch_in"])
        if new_out is not None and not (new_in < new_out <= new_in + MAX_SHIFT):
            raise unprocessable(
                "punch_out must be after punch_in and within 24 hours of it", ["body", "punch_out"]
            )

    new = {"status": status, "punch_in": new_in, "punch_out": new_out}
    new.update(derive(status, new_in, new_out, rec["date"], emp))
    old = {
        "status": rec["status"],
        "punch_in": old_in,
        "punch_out": old_out,
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": bool(rec.get("half_day", False)),
    }
    changes = {k: {"from": old[k], "to": v} for k, v in new.items() if old[k] != v}
    if not any(k in changes for k in ("status", "punch_in", "punch_out")):
        raise unprocessable("request changes nothing")
    return new, changes


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: Annotated[str, Path(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"), AfterValidator(_parse_date)], body: RegularizeRequest):
    emp = get_employee(emp_code)
    logs = db.attendance_logs
    # Optimistic concurrency: the update only applies if the record is still exactly what we read
    # (including history length). If someone else got there first we re-read and retry.
    for _ in range(5):
        rec = logs.find_one({"emp_code": emp_code, "date": date})
        if rec is None:
            raise HTTPException(404, "no attendance record for that date")
        fields, changes = build_correction(rec, emp, body)
        entry = {"at": now_dt(), "by": body.regularized_by, "reason": body.reason, "changes": changes}
        updated = logs.find_one_and_update(
            {
                "_id": rec["_id"],
                "status": rec["status"],
                "punch_in": rec.get("punch_in"),
                "punch_out": rec.get("punch_out"),
                "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, len(rec.get("history") or [])]},
            },
            {"$set": fields, "$push": {"history": entry}},
            return_document=ReturnDocument.AFTER,
        )
        if updated is not None:
            return record_out(updated)
    raise HTTPException(409, "record was modified concurrently, please retry")


# --------------------------------------------------------------------------- #
# analytics - shared pieces
# --------------------------------------------------------------------------- #
def month_range(month: str) -> tuple[str, str]:
    y, m = int(month[:4]), int(month[5:])
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def count_weekdays(start: date, end: date) -> int:
    n, d = 0, start
    while d <= end:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


def half_up_div(num: int, den: int) -> int:
    """round-half-up(num / den) with integers only (num, den >= 0, den > 0)."""
    return (2 * num + den) // (2 * den)


def log_exprs(prefix: str = "") -> dict:
    """Aggregation expressions over one attendance log. `prefix` is e.g. "logs." after a $lookup."""
    f = lambda name: f"${prefix}{name}"  # noqa: E731
    presence = {"$in": [f("status"), PRESENCE]}
    # $dayOfWeek: 1 = Sunday ... 7 = Saturday, so Mon-Fri is 2..6. The date string is parsed as UTC.
    dow = {"$dayOfWeek": {"$dateFromString": {"dateString": f("date"), "format": "%Y-%m-%d", "onError": None, "onNull": None}}}
    weekday = {"$in": [dow, [2, 3, 4, 5, 6]]}
    late = {"$ifNull": [f("late_minutes"), 0]}
    has_hours = {"$and": [presence, {"$isNumber": f("work_hours")}]}
    return {
        # present days in half-day units: full day = 2, half day = 1 (keeps arithmetic integer / exact)
        "present2": {"$cond": [{"$and": [presence, weekday]}, {"$cond": [{"$eq": [{"$ifNull": [f("half_day"), False]}, True]}, 1, 2]}, 0]},
        "late_count": {"$cond": [{"$gt": [late, 0]}, 1, 0]},
        "late_total": late,
        "leave": {"$cond": [{"$eq": [f("status"), "LEAVE"]}, 1, 0]},
        "on_duty": {"$cond": [{"$eq": [f("status"), "ON_DUTY"]}, 1, 0]},
        "overtime": {"$ifNull": [f("overtime_minutes"), 0]},
        # work hours in hundredths so the later average is exact
        "wh100": {"$cond": [has_hours, {"$round": [{"$multiply": [f("work_hours"), 100]}, 0]}, 0]},
        "wh_n": {"$cond": [has_hours, 1, 0]},
    }


# --------------------------------------------------------------------------- #
# analytics - pipeline builders (also used by /admin/explain so the plans are the real ones)
# --------------------------------------------------------------------------- #
def monthly_pipeline(emp_code: str, start: str, end: str) -> list:
    x = log_exprs()
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lte": end}}},
        {
            "$group": {
                "_id": None,
                "present2": {"$sum": x["present2"]},
                "leave_days": {"$sum": x["leave"]},
                "late_count": {"$sum": x["late_count"]},
                "total_late_minutes": {"$sum": x["late_total"]},
                "total_overtime_minutes": {"$sum": x["overtime"]},
            }
        },
    ]


def summary_pipeline(start: str, end: str, department: Optional[str]) -> list:
    match: dict = {"joined_on": {"$lte": end}}  # R9
    if department is not None:
        match["department"] = department
    x = log_exprs("logs.")
    sums = {
        "present2": {"$sum": x["present2"]},
        "wh100": {"$sum": x["wh100"]},
        "wh_n": {"$sum": x["wh_n"]},
        "late_count": {"$sum": x["late_count"]},
        "late_total": {"$sum": x["late_total"]},
        "leave": {"$sum": x["leave"]},
        "on_duty": {"$sum": x["on_duty"]},
    }
    again = {k: {"$sum": f"${k}"} for k in sums}
    return [
        {"$match": match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [{"$match": {"date": {"$gte": start, "$lte": end}}}],
                "as": "logs",
            }
        },
        # preserveNullAndEmptyArrays keeps employees with zero logs (they still count in headcount)
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        # one row per employee first, so that headcount is a plain count of employees, not of logs
        {"$group": {"_id": {"department": "$department", "emp_code": "$emp_code"}, **sums}},
        {"$group": {"_id": "$_id.department", "headcount": {"$sum": 1}, **again}},
        {"$sort": {"_id": 1}},
    ]


def leaderboard_pipeline(start: str, end: str, limit: int, department: Optional[str]) -> list:
    pipeline: list = [
        {"$match": {"date": {"$gte": start, "$lte": end}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # logs of unknown employees disappear here
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department only
    pipeline += [
        # $rank = standard competition ranking: ties share a rank, the next rank is skipped
        {"$setWindowFields": {"sortBy": {"total": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit applies after ranking
        {"$sort": {"total": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": "$total",
                "late_count": 1,
            }
        },
    ]
    return pipeline


def trend_pipeline(codes: list[str], joined_on: list[str], start: str, end: str) -> list:
    """Per-day aggregation, gap-filled with $densify, headcount and rates per row, 7-day window average.

    Rates are carried as integers in units of 1e-4 so that half-up rounding is exact.
    """
    x = log_exprs()
    first = datetime.combine(date.fromisoformat(start), time(), tzinfo=UTC)
    after_last = datetime.combine(date.fromisoformat(end), time(), tzinfo=UTC) + timedelta(days=1)
    return [
        {"$match": {"emp_code": {"$in": codes}, "date": {"$gte": start, "$lte": end}}},
        {"$group": {"_id": "$date", "present2": {"$sum": _present2_any_day()}, "late_count": {"$sum": x["late_count"]}}},
        {"$set": {"d": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}}}},
        {"$sort": {"d": 1}},
        # one row for every calendar day in [start, end]; new rows have no present2/late_count
        {"$densify": {"field": "d", "range": {"step": 1, "unit": "day", "bounds": [first, after_last]}}},
        {
            "$set": {
                "date": {"$dateToString": {"date": "$d", "format": "%Y-%m-%d"}},
                "present2": {"$ifNull": ["$present2", 0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
                "is_working_day": {"$in": [{"$dayOfWeek": "$d"}, [2, 3, 4, 5, 6]]},
            }
        },
        {
            "$set": {  # R9: employees with joined_on <= this day (ISO strings compare correctly)
                "headcount": {
                    "$size": {"$filter": {"input": {"$literal": joined_on}, "cond": {"$lte": ["$$this", "$date"]}}}
                }
            }
        },
        {
            "$set": {
                "rate_i": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        # floor(present/headcount * 10000 + 0.5), written with integers only
                        {
                            "$floor": {
                                "$divide": [
                                    {"$add": [{"$multiply": ["$present2", 10000]}, "$headcount"]},
                                    {"$multiply": [2, "$headcount"]},
                                ]
                            }
                        },
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"d": 1},
                "output": {
                    "win_sum": {"$sum": "$rate_i", "window": {"documents": [-6, 0]}},
                    "win_n": {
                        "$sum": {"$cond": [{"$eq": [{"$type": "$rate_i"}, "null"]}, 0, 1]},
                        "window": {"documents": [-6, 0]},
                    },
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": {"$divide": ["$present2", 2]},
                "late_count": 1,
                "attendance_rate": {
                    "$cond": [{"$eq": [{"$type": "$rate_i"}, "null"]}, None, {"$divide": ["$rate_i", 10000]}]
                },
                "moving_avg_7d": {
                    "$cond": [
                        {"$gt": ["$win_n", 0]},
                        {
                            "$divide": [
                                {
                                    "$floor": {
                                        "$divide": [
                                            {"$add": [{"$multiply": [2, "$win_sum"]}, "$win_n"]},
                                            {"$multiply": [2, "$win_n"]},
                                        ]
                                    }
                                },
                                10000,
                            ]
                        },
                        None,
                    ]
                },
            }
        },
        {"$sort": {"date": 1}},
    ]


def _present2_any_day() -> dict:
    """Trend present_count counts presence on any weekday or weekend (half day = 0.5), in half-day units."""
    presence = {"$in": ["$status", PRESENCE]}
    return {"$cond": [presence, {"$cond": [{"$eq": [{"$ifNull": ["$half_day", False]}, True]}, 1, 2]}, 0]}


def trend_inputs(department: str, start: str, end: str) -> tuple[list[str], list[str]]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if d1 < d0:
        raise unprocessable("'to' must not be before 'from'", ["query", "to"])
    if (d1 - d0).days + 1 > 92:
        raise unprocessable("range must not exceed 92 days", ["query", "to"])
    emps = list(db.employees.find({"department": department}, {"_id": 0, "emp_code": 1, "joined_on": 1}))
    return [e["emp_code"] for e in emps], [e["joined_on"] for e in emps]


# --------------------------------------------------------------------------- #
# analytics - endpoints
# --------------------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: MonthStr):
    emp = get_employee(emp_code)
    start, end = month_range(month)
    rows = list(db.attendance_logs.aggregate(monthly_pipeline(emp_code, start, end)))
    agg = rows[0] if rows else {}
    present2 = int(agg.get("present2", 0))

    # R7: working days count from joined_on for a mid-month joiner (none if they join after the month)
    first = max(date.fromisoformat(start), date.fromisoformat(emp["joined_on"]))
    last = date.fromisoformat(end)
    working_days = count_weekdays(first, last) if first <= last else 0
    pct = None
    if working_days > 0:
        # present/working*100 to 2 dp half-up, exact: (present2/2)/working*100*100 = 5000*present2/working
        pct = half_up_div(5000 * present2, working_days) / 100
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present2 / 2,
        "leave_days": int(agg.get("leave_days", 0)),
        "late_count": int(agg.get("late_count", 0)),
        "total_late_minutes": int(agg.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(agg.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: MonthStr, department: Optional[str] = None):
    start, end = month_range(month)
    items = []
    for r in db.employees.aggregate(summary_pipeline(start, end, department)):
        n = int(r["wh_n"])
        items.append(
            {
                "department": r["_id"],
                "headcount": int(r["headcount"]),
                "present_days": int(r["present2"]) / 2,
                "avg_work_hours": half_up_div(int(r["wh100"]), n) / 100 if n else None,
                "late_count": int(r["late_count"]),
                "total_late_minutes": int(r["late_total"]),
                "leave_count": int(r["leave"]),
                "on_duty_count": int(r["on_duty"]),
            }
        )
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: MonthStr,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    department: Optional[str] = None,
):
    start, end = month_range(month)
    items = list(db.attendance_logs.aggregate(leaderboard_pipeline(start, end, limit, department)))
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_: Annotated[DateStr, Query(alias="from")],
    to: DateStr,
):
    codes, joined_on = trend_inputs(department, from_, to)
    if not codes:
        raise HTTPException(404, "department not found")
    items = list(db.attendance_logs.aggregate(trend_pipeline(codes, joined_on, from_, to)))
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# admin: explain
# --------------------------------------------------------------------------- #
ExplainEndpoint = Literal[
    "attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"
]


def _require(**params) -> None:
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise HTTPException(
            422,
            detail=[
                {"loc": ["query", k], "msg": "Field required", "type": "missing"} for k in missing
            ],
        )


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: ExplainEndpoint,
    emp_code: Optional[str] = None,
    month: Optional[MonthStr] = None,
    department: Optional[str] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[Status] = None,
    from_: Annotated[Optional[DateStr], Query(alias="from")] = None,
    to: Optional[DateStr] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    # Build the same query the real endpoint runs, then wrap it in `explain`.
    if endpoint == "attendance_list":
        collection = "attendance_logs"
        command: dict = {
            "find": collection,
            "filter": attendance_query(emp_code, date_from, date_to, status),
            "sort": dict(ATTENDANCE_SORT),
            "skip": (page - 1) * page_size,
            "limit": page_size,
        }
    else:
        if endpoint == "employee_monthly":
            _require(emp_code=emp_code, month=month)
            collection, pipeline = "attendance_logs", monthly_pipeline(emp_code, *month_range(month))
        elif endpoint == "department_summary":
            _require(month=month)
            collection, pipeline = "employees", summary_pipeline(*month_range(month), department)
        elif endpoint == "late_leaderboard":
            _require(month=month)
            collection, pipeline = "attendance_logs", leaderboard_pipeline(*month_range(month), limit, department)
        else:
            _require(**{"department": department, "from": from_, "to": to})
            codes, joined_on = trend_inputs(department, from_, to)
            collection, pipeline = "attendance_logs", trend_pipeline(codes, joined_on, from_, to)
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}

    raw = db.command({"explain": command, "verbosity": "executionStats"})
    # BSON -> plain JSON (dates, ObjectIds, Int64 ...)
    explain = json.loads(json_util.dumps(raw, json_options=json_util.RELAXED_JSON_OPTIONS))
    return {"endpoint": endpoint, "collection": collection, "explain": explain}
