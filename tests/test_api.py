"""
Integration tests. They need a live MongoDB (MONGO_URI, default mongodb://localhost:27017) and use a
throw-away database that is dropped afterwards - never point MONGO_URI at data you care about.

    pytest -q
"""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

os.environ["MONGO_DB"] = f"hrone_test_{uuid.uuid4().hex[:8]}"

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def ist_ms(s: str, ms: int = 0) -> int:
    """'2026-07-06 09:40:01' (IST) -> epoch milliseconds."""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
    return (dt - EPOCH) // timedelta(milliseconds=1) + ms


def utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def client():
    try:
        main.client.admin.command("ping")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"MongoDB not reachable: {exc}")
    with TestClient(main.app) as c:
        yield c
    main.client.drop_database(main.db.name)


@pytest.fixture(autouse=True)
def clean(client):
    main.db.employees.delete_many({})
    main.db.attendance_logs.delete_many({})


def make_emp(client, code, dept="Eng", start="09:30", end="18:30", joined="2026-07-01"):
    r = client.post(
        "/employees",
        json={
            "emp_code": code, "name": f"Name {code}", "email": f"{code.lower()}@example.com",
            "department": dept, "shift_start": start, "shift_end": end, "joined_on": joined,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def insert_log(code, day, status="PRESENT", pin=None, pout=None, wh=None, late=0, ot=0, half=False):
    main.db.attendance_logs.insert_one({
        "emp_code": code, "date": day, "status": status, "punch_in": pin, "punch_out": pout,
        "work_hours": wh, "late_minutes": late, "overtime_minutes": ot, "half_day": half, "history": [],
    })


# ----------------------------------------------------------------- employees
def test_employee_create_validation_and_duplicate(client):
    e = make_emp(client, "EMP0001")
    assert set(e) == {"emp_code", "name", "email", "department", "shift_start", "shift_end", "joined_on", "created_at"}
    assert isinstance(e["created_at"], int) and e["created_at"] > 10**12
    assert client.post("/employees", json={**e, "email": "a@b.co"}).status_code == 409
    bad = {**e, "emp_code": "EMP0002"}
    assert client.post("/employees", json={**bad, "shift_end": bad["shift_start"]}).status_code == 422
    assert client.post("/employees", json={**bad, "emp_code": "X1"}).status_code == 422
    assert client.post("/employees", json={**bad, "joined_on": "2026-02-30"}).status_code == 422
    assert client.post("/employees", json={**bad, "email": "nope"}).status_code == 422


def test_concurrent_employee_create_one_winner(client):
    body = {"emp_code": "EMP0009", "name": "A", "email": "a@b.co", "department": "Eng", "joined_on": "2026-07-01"}
    with ThreadPoolExecutor(10) as ex:
        codes = list(ex.map(lambda _: client.post("/employees", json=body).status_code, range(10)))
    assert sorted(codes) == [201] + [409] * 9


def test_employee_list_filter_sort_pagination(client):
    for c, d in [("EMP0003", "Ops"), ("EMP0001", "Eng"), ("EMP0002", "Eng")]:
        make_emp(client, c, d)
    r = client.get("/employees", params={"department": "Eng", "page_size": 1, "page": 2}).json()
    assert r["total"] == 2 and [i["emp_code"] for i in r["items"]] == ["EMP0002"]
    assert [i["emp_code"] for i in client.get("/employees").json()["items"]] == ["EMP0001", "EMP0002", "EMP0003"]
    assert client.get("/employees", params={"page_size": 101}).status_code == 422
    assert client.get("/employees", params={"page": 0}).status_code == 422


# ---------------------------------------------------------- punch-in / out
@pytest.mark.parametrize(
    "when,ms,late",
    [("09:40:00", 0, 0), ("09:40:01", 0, 10), ("10:15:59", 0, 45), ("09:40:00", 900, 0), ("09:20:00", 0, 0)],
)
def test_late_rule_r2_and_truncation(client, when, ms, late):
    make_emp(client, "EMP0001")
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ist_ms(f"2026-07-06 {when}", ms)})
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["late_minutes"] == late and j["date"] == "2026-07-06"
    assert j["punch_in"] == ist_ms(f"2026-07-06 {when}")  # truncated to whole seconds
    assert (j["punch_out"], j["work_hours"], j["overtime_minutes"], j["half_day"], j["history"]) == (None, None, 0, False, [])


def test_punch_in_errors(client):
    make_emp(client, "EMP0001")
    t = ist_ms("2026-07-06 09:30:00")
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP7777", "punched_at": t}).status_code == 404
    for bad in (t // 1000, float(t), str(t), True, 10**14):
        assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": bad}).status_code == 422
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "status": "ABSENT"}).status_code == 422
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": t, "status": "WFH"}).status_code == 201
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": t + 60000}).status_code == 409


def test_concurrent_punch_in_one_winner(client):
    make_emp(client, "EMP0001")
    body = {"emp_code": "EMP0001", "punched_at": ist_ms("2026-07-06 09:30:00")}
    with ThreadPoolExecutor(12) as ex:
        codes = list(ex.map(lambda _: client.post("/attendance/punch-in", json=body).status_code, range(12)))
    assert sorted(codes) == [201] + [409] * 11
    assert main.db.attendance_logs.count_documents({}) == 1


@pytest.mark.parametrize(
    "out,wh,ot,half",
    [
        ("2026-07-06 18:59:00", 9.48, 0, False),   # 29 min after shift end: below the 30 min minimum
        ("2026-07-06 19:00:00", 9.5, 30, False),
        ("2026-07-06 13:59:00", 4.48, 0, True),    # 4h29m
        ("2026-07-06 14:00:00", 4.5, 0, False),    # exactly 4.50 is not a half day
    ],
)
def test_punch_out_rules(client, out, wh, ot, half):
    make_emp(client, "EMP0001")
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ist_ms("2026-07-06 09:30:00")})
    r = client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ist_ms(out)})
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["work_hours"], j["overtime_minutes"], j["half_day"]) == (wh, ot, half)
    assert j["history"] == []  # punch events never write history


def test_work_hours_round_half_up_and_rounded_half_day(client):
    make_emp(client, "EMP0001")
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ist_ms("2026-07-06 09:30:00")})
    # 8h00m18s = 8.005 h -> 8.01 under half-up (round() / banker's would give 8.0)
    r = client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ist_ms("2026-07-06 17:30:18")})
    assert r.json()["work_hours"] == 8.01
    make_emp(client, "EMP0002")
    client.post("/attendance/punch-in", json={"emp_code": "EMP0002", "punched_at": ist_ms("2026-07-06 09:30:00")})
    # 4h29m50s = 4.4972 h -> rounds to 4.50 -> NOT a half day (R5 uses the rounded value)
    r = client.post("/attendance/punch-out", json={"emp_code": "EMP0002", "punched_at": ist_ms("2026-07-06 13:59:50")})
    assert (r.json()["work_hours"], r.json()["half_day"]) == (4.5, False)


def test_punch_out_errors(client):
    make_emp(client, "EMP0001")
    t = ist_ms("2026-07-06 09:30:00")
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP7777"}).status_code == 404
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": t}).status_code == 404
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": t})
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": t}).status_code == 422
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": t - 3600000}).status_code == 422
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": t + 25 * 3600000}).status_code == 422
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": t + 3600000}).status_code == 200
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": t + 7200000}).status_code == 409


def test_concurrent_punch_out_one_winner(client):
    make_emp(client, "EMP0001")
    t = ist_ms("2026-07-06 09:30:00")
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": t})
    body = {"emp_code": "EMP0001", "punched_at": t + 9 * 3600000}
    with ThreadPoolExecutor(12) as ex:
        codes = list(ex.map(lambda _: client.post("/attendance/punch-out", json=body).status_code, range(12)))
    assert sorted(codes) == [200] + [409] * 11


def test_overnight_shift(client):
    make_emp(client, "EMP0005", "Ops", "22:00", "06:00")
    # 05:30 IST on the 7th is before shift_end -> belongs to the shift that started on the 6th
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ist_ms("2026-07-07 05:30:00")})
    assert r.json()["date"] == "2026-07-06"
    assert r.json()["late_minutes"] == 450  # 22:00 -> 05:30
    r = client.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ist_ms("2026-07-07 06:40:00")})
    assert r.status_code == 200
    assert r.json()["overtime_minutes"] == 40 and r.json()["work_hours"] == 1.17
    make_emp(client, "EMP0006", "Ops", "22:00", "06:00")
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0006", "punched_at": ist_ms("2026-07-06 21:55:00")})
    assert r.json()["date"] == "2026-07-06" and r.json()["late_minutes"] == 0
    r = client.post("/attendance/punch-out", json={"emp_code": "EMP0006", "punched_at": ist_ms("2026-07-07 06:40:00")})
    assert (r.json()["work_hours"], r.json()["overtime_minutes"]) == (8.75, 40)  # sample doc 7


# --------------------------------------------------------------- regularize
def test_regularize_flow_and_history(client):
    make_emp(client, "EMP0001")
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ist_ms("2026-07-07 10:05:00")})
    client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ist_ms("2026-07-07 18:30:00")})
    url = "/attendance/EMP0001/2026-07-07"
    body = {"punch_in": ist_ms("2026-07-07 09:28:00"), "reason": "biometric glitch", "regularized_by": "hr.admin", "work_hours": 99}
    r = client.patch(url, json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["late_minutes"], j["work_hours"]) == (0, 9.03)
    assert len(j["history"]) == 1
    ch = j["history"][0]["changes"]
    assert set(ch) == {"punch_in", "late_minutes", "work_hours"}  # only what really changed
    assert ch["punch_in"] == {"from": ist_ms("2026-07-07 10:05:00"), "to": ist_ms("2026-07-07 09:28:00")}
    assert ch["late_minutes"] == {"from": 35, "to": 0}

    r2 = client.patch(url, json={"status": "WFH", "reason": "was remote", "regularized_by": "hr.two"})
    assert r2.status_code == 200 and len(r2.json()["history"]) == 2
    assert r2.json()["history"][0] == j["history"][0]  # earlier entry untouched
    # same request again changes nothing
    assert client.patch(url, json={"status": "WFH", "reason": "again!", "regularized_by": "x"}).status_code == 422
    # LEAVE clears the punch times
    r3 = client.patch(url, json={"status": "LEAVE", "reason": "approved leave", "regularized_by": "x"})
    j3 = r3.json()
    assert (j3["punch_in"], j3["punch_out"], j3["work_hours"], j3["half_day"]) == (None, None, None, False)
    assert len(j3["history"]) == 3


def test_regularize_validation(client):
    make_emp(client, "EMP0001")
    insert_log("EMP0001", "2026-07-08", "ABSENT")
    ok = {"reason": "valid reason", "regularized_by": "hr"}
    t = ist_ms("2026-07-08 09:30:00")
    u = "/attendance/EMP0001/2026-07-08"
    assert client.patch("/attendance/EMP9999/2026-07-08", json={**ok, "status": "LEAVE"}).status_code == 404
    assert client.patch("/attendance/EMP0001/2026-07-09", json={**ok, "status": "LEAVE"}).status_code == 404
    assert client.patch(u, json={**ok, "status": "LEAVE", "punch_in": t}).status_code == 422   # LEAVE + punch time
    assert client.patch(u, json={**ok, "status": "PRESENT"}).status_code == 422                 # presence needs punch_in
    assert client.patch(u, json={**ok, "status": "PRESENT", "punch_in": ist_ms("2026-07-09 09:30:00")}).status_code == 422
    assert client.patch(u, json={**ok, "status": "PRESENT", "punch_in": t, "punch_out": t}).status_code == 422
    assert client.patch(u, json={**ok, "status": "PRESENT", "punch_in": t, "punch_out": t + 25 * 3600000}).status_code == 422
    assert client.patch(u, json={"status": "LEAVE", "reason": "abc", "regularized_by": "hr"}).status_code == 422
    assert client.patch(u, json={**ok, "status": "ABSENT"}).status_code == 422                  # nothing changes
    r = client.patch(u, json={**ok, "status": "ON_DUTY", "punch_in": t, "punch_out": t + 3600000})
    assert r.status_code == 200 and r.json()["work_hours"] == 1.0 and r.json()["half_day"] is True


def test_concurrent_regularize_keeps_every_history_entry(client):
    make_emp(client, "EMP0001")
    insert_log("EMP0001", "2026-07-08", "PRESENT", utc("2026-07-08 04:00:00"), None)
    base = ist_ms("2026-07-08 09:30:00")

    def patch(i):
        return client.patch(
            "/attendance/EMP0001/2026-07-08",
            json={"punch_in": base + (i + 1) * 60000, "reason": f"change {i}", "regularized_by": "hr"},
        ).status_code

    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(patch, range(8)))
    assert set(codes) <= {200, 409}
    hist = main.db.attendance_logs.find_one({"emp_code": "EMP0001"})["history"]
    assert len(hist) == codes.count(200) >= 1  # no 200 without its history entry, none overwritten


# --------------------------------------------------------------- attendance list
def test_attendance_list(client):
    make_emp(client, "EMP0001")
    make_emp(client, "EMP0002")
    insert_log("EMP0002", "2026-07-06")
    insert_log("EMP0001", "2026-07-06")
    insert_log("EMP0001", "2026-07-07", "WFH")
    insert_log("EMP0001", "2026-07-08", "LEAVE")
    r = client.get("/attendance").json()
    assert r["total"] == 4
    assert [(i["date"], i["emp_code"]) for i in r["items"]] == [
        ("2026-07-08", "EMP0001"), ("2026-07-07", "EMP0001"), ("2026-07-06", "EMP0001"), ("2026-07-06", "EMP0002"),
    ]
    r = client.get("/attendance", params={"emp_code": "EMP0001", "date_from": "2026-07-07", "date_to": "2026-07-07"}).json()
    assert r["total"] == 1 and r["items"][0]["status"] == "WFH"
    assert client.get("/attendance", params={"status": "LEAVE"}).json()["total"] == 1
    assert client.get("/attendance", params={"page": 2, "page_size": 3}).json()["items"][0]["emp_code"] == "EMP0002"
    assert client.get("/attendance", params={"date_from": "2026-07-09", "date_to": "2026-07-01"}).status_code == 422
    assert client.get("/attendance", params={"date_from": "2026-13-01"}).status_code == 422
    assert client.get("/attendance", params={"status": "SICK"}).status_code == 422


def test_legacy_record_defaults(client):
    make_emp(client, "EMP0002")
    main.db.attendance_logs.insert_one({
        "emp_code": "EMP0002", "date": "2026-07-14", "status": "PRESENT",
        "punch_in": utc("2026-07-14 04:02:00"), "punch_out": utc("2026-07-14 13:01:00"),
        "work_hours": 8.98, "late_minutes": 0, "overtime_minutes": 0,
    })
    j = client.get("/attendance").json()["items"][0]
    assert j["half_day"] is False and j["history"] == []
    assert j["punch_in"] == ist_ms("2026-07-14 09:32:00")


# --------------------------------------------------------------------- analytics
@pytest.fixture
def july(client):
    make_emp(client, "EMP0001")
    make_emp(client, "EMP0002")
    make_emp(client, "EMP0003", joined="2026-07-15")  # mid-month joiner
    make_emp(client, "EMP0004")                       # no logs at all
    make_emp(client, "EMP0005", "Ops", "22:00", "06:00")
    make_emp(client, "EMP0006", joined="2026-08-05")  # joins after July
    P = utc("2026-07-06 04:00:00")
    insert_log("EMP0001", "2026-07-06", pin=P, pout=P, wh=9.08, late=35, ot=40)
    insert_log("EMP0001", "2026-07-07", pin=P, pout=P, wh=9.03)
    insert_log("EMP0001", "2026-07-08", "WFH", pin=P, pout=P, wh=3.5, half=True)
    insert_log("EMP0001", "2026-07-09", "LEAVE")
    insert_log("EMP0001", "2026-07-11", pin=P, pout=P, wh=8.0, late=20, ot=30)  # Saturday
    insert_log("EMP0002", "2026-07-06", pin=P, pout=P, wh=8.0, late=35)
    insert_log("EMP0002", "2026-07-07", pin=P, pout=P, wh=7.0, late=20)
    insert_log("EMP0002", "2026-07-08", "ON_DUTY", pin=P)
    insert_log("EMP0003", "2026-07-15", pin=P, late=10)
    insert_log("EMP0005", "2026-07-06", pin=P, pout=P, wh=8.75, late=50, ot=40)
    insert_log("EMP9999", "2026-07-06", pin=P, late=100)  # orphan log: must be ignored


def test_employee_monthly(client, july):
    j = client.get("/analytics/employees/EMP0001/monthly", params={"month": "2026-07"}).json()
    assert j == {
        "emp_code": "EMP0001", "month": "2026-07", "working_days": 23, "present_days": 2.5, "leave_days": 1,
        "late_count": 2, "total_late_minutes": 55, "total_overtime_minutes": 70, "attendance_pct": 10.87,
    }
    j = client.get("/analytics/employees/EMP0003/monthly", params={"month": "2026-07"}).json()
    assert (j["working_days"], j["present_days"], j["attendance_pct"]) == (13, 1.0, 7.69)  # joined 15th, counted from there
    j = client.get("/analytics/employees/EMP0004/monthly", params={"month": "2026-07"}).json()
    assert (j["present_days"], j["attendance_pct"], j["late_count"]) == (0.0, 0.0, 0)
    j = client.get("/analytics/employees/EMP0006/monthly", params={"month": "2026-07"}).json()
    assert (j["working_days"], j["attendance_pct"]) == (0, None)
    assert client.get("/analytics/employees/EMP7777/monthly", params={"month": "2026-07"}).status_code == 404
    assert client.get("/analytics/employees/EMP0001/monthly", params={"month": "2026-13"}).status_code == 422
    assert client.get("/analytics/employees/EMP0001/monthly").status_code == 422


def test_department_summary(client, july):
    r = client.get("/analytics/departments/summary", params={"month": "2026-07"}).json()
    assert [i["department"] for i in r["items"]] == ["Eng", "Ops"]
    eng, ops = r["items"]
    assert eng == {
        "department": "Eng", "headcount": 4,  # EMP0001-4; EMP0006 joins in August; EMP0004 has no logs
        "present_days": 6.5, "avg_work_hours": 7.44,  # 44.61/6 = 7.435 -> half-up 7.44
        "late_count": 5, "total_late_minutes": 120, "leave_count": 1, "on_duty_count": 1,
    }
    assert ops["headcount"] == 1 and ops["avg_work_hours"] == 8.75
    r = client.get("/analytics/departments/summary", params={"month": "2026-07", "department": "Ops"}).json()
    assert [i["department"] for i in r["items"]] == ["Ops"]
    r = client.get("/analytics/departments/summary", params={"month": "2026-06"}).json()
    assert r["items"] == []  # nobody had joined yet: departments with headcount 0 are omitted
    j = client.get("/analytics/departments/summary", params={"month": "2026-07", "department": "Eng"}).json()["items"][0]
    assert j["headcount"] == 4


def test_late_leaderboard_ties_and_limit(client, july):
    def board(**p):
        return client.get("/analytics/leaderboard/late", params={"month": "2026-07", **p}).json()["items"]

    rows = board()
    assert [(r["rank"], r["emp_code"], r["total_late_minutes"]) for r in rows] == [
        (1, "EMP0001", 55), (1, "EMP0002", 55), (3, "EMP0005", 50), (4, "EMP0003", 10),
    ]
    assert rows[0]["name"] == "Name EMP0001" and rows[0]["department"] == "Eng" and rows[0]["late_count"] == 2
    assert [r["emp_code"] for r in board(limit=1)] == ["EMP0001", "EMP0002"]  # tie at the cutoff: both kept
    assert [r["emp_code"] for r in board(limit=2)] == ["EMP0001", "EMP0002"]
    assert [r["rank"] for r in board(limit=3)] == [1, 1, 3]
    assert [(r["rank"], r["emp_code"]) for r in board(department="Eng")] == [(1, "EMP0001"), (1, "EMP0002"), (3, "EMP0003")]
    assert client.get("/analytics/leaderboard/late", params={"month": "2026-07", "limit": 51}).status_code == 422
    assert board(month="2026-05") == []


def test_department_trend_exact(client, july):
    """Hand calculation (headcount 3 every day; EMP0003 joins on the 15th):
    Mon 6: present 2.0 -> .6667 | Tue 7: 2.0 -> .6667 | Wed 8: 0.5 + 1 = 1.5 -> .5 | Thu 9: 0 -> 0 | Fri 10: no records -> 0
    Sat/Sun rates are null. 7-day average of non-null rates inside the range, half-up at 4 dp:
      Wed (.6667+.6667+.5)/3 = .6111 | Thu 18334/4 = .45835 -> .4584 | Fri 18334/5 = .3667 | Sat, Sun the same.
    """
    items = client.get("/analytics/departments/Eng/trend", params={"from": "2026-07-04", "to": "2026-07-12"}).json()["items"]
    assert len(items) == 9
    by = {i["date"]: i for i in items}
    assert [i["date"] for i in items] == sorted(by)
    assert [by[d]["attendance_rate"] for d in sorted(by)] == [None, None, 0.6667, 0.6667, 0.5, 0.0, 0.0, None, None]
    assert [by[d]["moving_avg_7d"] for d in sorted(by)] == [None, None, 0.6667, 0.6667, 0.6111, 0.4584, 0.3667, 0.3667, 0.3667]
    assert [by[d]["present_count"] for d in sorted(by)] == [0.0, 0.0, 2.0, 2.0, 1.5, 0.0, 0.0, 1.0, 0.0]
    assert [by[d]["late_count"] for d in sorted(by)] == [0, 0, 2, 1, 0, 0, 0, 1, 0]
    assert [by[d]["is_working_day"] for d in sorted(by)] == [False, False, True, True, True, True, True, False, False]
    assert {i["headcount"] for i in items} == {3}


def test_department_trend_edges(client, july):
    # EMP0003 joins 07-15: headcount steps from 3 to 4 on that day
    items = client.get("/analytics/departments/Eng/trend", params={"from": "2026-07-14", "to": "2026-07-15"}).json()["items"]
    assert [i["headcount"] for i in items] == [3, 4]
    # gaps are filled even when the range has no records at all
    items = client.get("/analytics/departments/Eng/trend", params={"from": "2026-01-05", "to": "2026-01-06"}).json()["items"]
    assert len(items) == 2 and items[0]["headcount"] == 0 and items[0]["attendance_rate"] is None
    assert client.get("/analytics/departments/Nope/trend", params={"from": "2026-07-01", "to": "2026-07-02"}).status_code == 404
    assert client.get("/analytics/departments/Eng/trend", params={"from": "2026-07-05", "to": "2026-07-01"}).status_code == 422
    assert client.get("/analytics/departments/Eng/trend", params={"from": "2026-01-01", "to": "2026-04-03"}).status_code == 422  # 93 days
    assert client.get("/analytics/departments/Eng/trend", params={"from": "2026-01-01", "to": "2026-04-02"}).status_code == 200  # 92
    assert client.get("/analytics/departments/Eng/trend", params={"from": "2026-01-01"}).status_code == 422


# --------------------------------------------------------------- explain / system
@pytest.mark.parametrize(
    "endpoint,params",
    [
        ("attendance_list", {}),
        ("attendance_list", {"emp_code": "EMP0001", "date_from": "2026-07-01", "date_to": "2026-07-31"}),
        ("attendance_list", {"status": "LEAVE"}),
        ("employee_monthly", {"emp_code": "EMP0001", "month": "2026-07"}),
        ("department_summary", {"month": "2026-07"}),
        ("department_summary", {"month": "2026-07", "department": "Eng"}),
        ("late_leaderboard", {"month": "2026-07"}),
        ("late_leaderboard", {"month": "2026-07", "department": "Eng", "limit": 5}),
        ("department_trend", {"department": "Eng", "from": "2026-07-01", "to": "2026-07-31"}),
    ],
)
def test_explain_uses_indexes(client, july, endpoint, params):
    r = client.get(f"/admin/explain/{endpoint}", params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["endpoint"] == endpoint and body["collection"] in ("attendance_logs", "employees")
    text = r.text
    assert "IXSCAN" in text and "COLLSCAN" not in text


def test_explain_validation(client):
    assert client.get("/admin/explain/employee_monthly", params={"month": "2026-07"}).status_code == 422
    assert client.get("/admin/explain/department_trend", params={"department": "Eng"}).status_code == 422
    assert client.get("/admin/explain/nonsense").status_code == 422


def test_health_and_indexes(client):
    assert client.get("/health").json() == {"status": "ok"}
    main.create_indexes()  # idempotent
    emp = {i["name"] for i in main.db.employees.list_indexes()}
    logs = {i["name"] for i in main.db.attendance_logs.list_indexes()}
    assert {"uq_emp_code", "dept_emp_code", "joined_on"} <= emp
    assert {"uq_emp_date", "date_desc_emp", "status_date_emp"} <= logs
