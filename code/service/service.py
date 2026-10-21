"""과제 8 상시 추론 서비스 v1 (team2-1).

매분(:15초) 공간마다 60분 예측·이상·상태를 계산해
- MQTT(localhost:1885, retain)로 발행하고
- 조회 API(:8091)로 제공하고
- 결과·처리 시간·경보 이력을 InfluxDB `service` 버킷에 남긴다.

입력: 합성 재생 데이터(InfluxDB `sim_live`, 강의실 A·B·회의실·라운지) — 학습 모델이 이 4개 공간용이다.
      실데이터(`sensors`, 실습실·사무실·사무실 밖)는 모델 없이 '데이터 끊김' 감시만 한다.
장애 대응
- 한 주기 계산이 실패해도 예외를 잡아 기록하고 다음 주기에 다시 계산한다.
- 데이터가 끊겨도 서비스는 계속 돌고, 'data_gap' 경보를 낸다.
- 같은 경보는 처음 한 번(raised), 이어지면 30분마다(reminder), 풀리면(cleared) 보낸다.
- 시험용: /srv/team/service/FAIL_NEXT 파일이 있으면 다음 주기 한 번만 일부러 실패한다.
- 재부팅 시 자동 시작: systemd (team2-svc.service)
"""
import json
import logging
import sys
import threading
import time
import traceback
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

import warnings

import pandas as pd
import paho.mqtt.client as mqtt
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.warnings import MissingPivotFunction
from influxdb_client.client.write_api import SYNCHRONOUS

warnings.simplefilter("ignore", MissingPivotFunction)

sys.path.insert(0, "/srv/team/service_kit")
import kit as K  # noqa: E402

ROOT = Path("/srv/team/service")
TOPIC = "team2/svc"
API_PORT = 8091
REPEAT_MIN = 30
REAL_GAP_MIN = 20
KST = "Asia/Seoul"
VERSION = "v1"

log = logging.getLogger("svc")
log.setLevel(logging.INFO)
for h in (logging.StreamHandler(sys.stdout), RotatingFileHandler(ROOT / "service.log", maxBytes=5_000_000, backupCount=3)):
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)


def load_env(path=ROOT / ".env"):
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = load_env()
STATE = {"latest": {}, "real": {}, "events": deque(maxlen=500),
         "status": {"service": "team2-1 추론 서비스", "version": VERSION, "online": True, "started": None,
                    "cycles": 0, "failures": 0, "last_cycle": None, "last_elapsed_s": None, "last_error": None,
                    "last_error_at": None, "backends": None}}
LOCK = threading.Lock()


# ------------------------------------------------------------------ 경보 상태 (중복 방지)
class Alerts:
    def __init__(self, path=ROOT / "alert_state.json"):
        self.path = path
        self.state = json.loads(path.read_text()) if path.exists() else {}

    def update(self, key, active, details, now):
        events = []
        cur = self.state.setdefault(key, {})
        for name, on in active.items():
            prev = cur.get(name)
            ev = None
            if on and not prev:
                cur[name] = {"since": now.isoformat(), "last_sent": now.isoformat()}
                ev = "raised"
            elif on and prev and now - pd.Timestamp(prev["last_sent"]) >= pd.Timedelta(minutes=REPEAT_MIN):
                prev["last_sent"] = now.isoformat()
                ev = "reminder"
            elif not on and prev:
                cur.pop(name)
                ev = "cleared"
            if ev:
                since = cur.get(name, prev or {}).get("since") if (on or prev) else None
                events.append({"key": key, "type": name, "kind": ev, "at": now.isoformat(), "since": since,
                               "detail": details.get(name)})
        self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))
        return events

    def active(self):
        return {k: v for k, v in self.state.items() if v}


# ------------------------------------------------------------------ 데이터 읽기
def _df(q, flux):
    d = q.query_data_frame(flux)
    if isinstance(d, list):
        d = pd.concat(d, ignore_index=True) if d else pd.DataFrame()
    return d


def kst(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert(KST).dt.tz_localize(None)


def read_sim(q):
    flux = f'''from(bucket: "sim_live") |> range(start: -{K.WINDOW_H * 60 + 15}m)
  |> filter(fn: (r) => r._measurement == "env" and (r._field == "co2" or r._field == "temperature" or r._field == "humidity"))
  |> pivot(rowKey: ["_time", "location"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "location", "co2", "temperature", "humidity"])'''
    d = _df(q, flux)
    if d.empty:
        return pd.DataFrame(columns=["time", "space", "co2", "temperature", "humidity"])
    for c in ("co2", "temperature", "humidity"):
        if c not in d:
            d[c] = float("nan")
    return pd.DataFrame({"time": kst(d["_time"]), "space": d["location"], "co2": d["co2"],
                         "temperature": d["temperature"], "humidity": d["humidity"]})


def read_real(q, now):
    """실데이터 공간별 마지막 수신(센서별) + 두 수집기의 최근 상태."""
    d = _df(q, '''from(bucket: "sensors") |> range(start: -6h)
  |> filter(fn: (r) => r._measurement == "env" and r._field == "temperature")
  |> group(columns: ["location", "device"]) |> last() |> keep(columns: ["_time", "location", "device"])''')
    out = {}
    if not d.empty:
        d["age"] = (now - kst(d["_time"])).dt.total_seconds() / 60
        for loc, g in d.groupby("location"):
            late = g[g.age > REAL_GAP_MIN]
            out[loc] = {"devices": len(g), "newest_age_min": round(float(g.age.min()), 1),
                        "late_devices": sorted(late.device.astype(str)), "data_gap": len(late) == len(g)}
    coll = {}
    for bucket, name in (("sensors", "team2-collector"), ("sim_live", "team2-collector-sim")):
        h = _df(q, f'''from(bucket: "{bucket}") |> range(start: -30m)
  |> filter(fn: (r) => r._measurement == "collector" and r._field == "mqtt_connected") |> last()''')
        if h.empty:
            coll[name] = {"heartbeat_age_min": None, "mqtt_connected": None}
        else:
            coll[name] = {"heartbeat_age_min": round(float((now - kst(h["_time"]).iloc[0]).total_seconds() / 60), 1),
                          "mqtt_connected": bool(h["_value"].iloc[0])}
    return out, coll


# ------------------------------------------------------------------ 한 주기
def cycle(kit, q, w, mq, alerts, schedule):
    t0 = time.time()
    now = pd.Timestamp.now(tz=KST).tz_localize(None).floor("s")
    fail = ROOT / "FAIL_NEXT"
    if fail.exists():
        fail.unlink()
        raise RuntimeError("의도된 실패 시험(FAIL_NEXT)")
    raw = read_sim(q)
    res = kit.run(raw, K.shifted_schedule(schedule, now), now)
    real, coll = read_real(q, now)
    sim_coll = coll["team2-collector-sim"]

    events = []
    for s, it in res.items():
        a, f, st = it["anomaly"], it["forecast"], it["state"]
        r = a["rules"]
        if r["data_gap"]["active"]:
            hb = sim_coll["heartbeat_age_min"]
            r["data_gap"]["likely_cause"] = ("수집기 정지 또는 연결 끊김" if hb is None or hb > 3 or not sim_coll["mqtt_connected"]
                                             else "센서·게이트웨이 무수신")
        active = {k: v["active"] for k, v in r.items()}
        active.update(anomaly_model=bool(a.get("is_anomaly")), co2_over_1000=bool(f.get("already_over_1000")),
                      vent_soon=bool(st.get("vent_alert")))
        details = {k: v for k, v in r.items()}
        details.update(anomaly_model={"score": a.get("score"), "cause": a.get("cause")},
                       co2_over_1000={"co2": it["now"]["co2"]},
                       vent_soon={"exceed_1000_in_min": f.get("exceed_1000_in_min"), "co2": it["now"]["co2"]})
        events += alerts.update(s, active, details, now)
        it["alerts_active"] = sorted(alerts.state.get(s, {}).keys())
    for loc, r in real.items():
        events += alerts.update(f"real:{loc}", {"data_gap": r["data_gap"]}, {"data_gap": r}, now)
        r["alerts_active"] = sorted(alerts.state.get(f"real:{loc}", {}).keys())

    elapsed = round(time.time() - t0, 3)
    with LOCK:
        STATE["latest"], STATE["real"] = res, {"spaces": real, "collectors": coll, "at": now.isoformat()}
        STATE["events"].extend(events)
        stt = STATE["status"]
        stt.update(cycles=stt["cycles"] + 1, last_cycle=now.isoformat(), last_elapsed_s=elapsed,
                   backends={k: v["backend"] for k, v in kit.backends().items()}, online=True)
        status = dict(stt)

    for s, it in res.items():
        mq.publish(f"{TOPIC}/state/{s}", json.dumps(it, ensure_ascii=False), qos=1, retain=True)
        mq.publish(f"{TOPIC}/alerts/{s}", json.dumps({"space": s, "at": now.isoformat(), "active": it["alerts_active"]},
                                                     ensure_ascii=False), qos=1, retain=True)
    for loc, r in real.items():
        mq.publish(f"{TOPIC}/real/{loc}", json.dumps(r | {"at": now.isoformat()}, ensure_ascii=False), qos=1, retain=True)
    for e in events:
        mq.publish(f"{TOPIC}/events", json.dumps(e, ensure_ascii=False, default=str), qos=1, retain=False)
        log.info("경보 %s %s %s", e["kind"], e["key"], e["type"])
    mq.publish(f"{TOPIC}/status", json.dumps(status, ensure_ascii=False), qos=1, retain=True)

    pts = [Point("svc_cycle").field("elapsed_s", float(elapsed)).field("ok", 1).field("events", len(events))
           .field("backend_tensorrt", int(all(v == "tensorrt" for v in status["backends"].values()))).time(pd.Timestamp(now, tz=KST))]
    for s, it in res.items():
        p = Point("svc_pred").tag("space", s).time(pd.Timestamp(now, tz=KST))
        if it["now"]["co2"] is not None:
            p.field("co2_now", float(it["now"]["co2"]))
        f = it["forecast"]
        if f["available"]:
            p.field("co2_30", float(f["co2"][5])).field("co2_60", float(f["co2"][11])).field("temp_60", float(f["temperature"][11]))
            p.field("exceed_1000_in_min", float(f["exceed_1000_in_min"]) if f["exceed_1000_in_min"] is not None else -1.0)
        if it["anomaly"].get("score") is not None:
            p.field("ae_score", float(it["anomaly"]["score"]))
        if it["state"]["available"]:
            p.field("occupied_prob", float(it["state"]["occupied_prob"])).field("vent_alert", int(it["state"]["vent_alert"]))
        p.field("alerts_active", len(it["alerts_active"]))
        pts.append(p)
    for e in events:
        pts.append(Point("svc_event").tag("key", e["key"]).tag("type", e["type"]).tag("kind", e["kind"])
                   .field("n", 1).time(pd.Timestamp(e["at"], tz=KST)))
    w.write(bucket="service", record=pts)
    log.info("주기 완료 %.3fs, 경보 이벤트 %d건", elapsed, len(events))


# ------------------------------------------------------------------ 조회 API
app = FastAPI(title="team2 교육장 추론 서비스 (team2-1)", version=VERSION,
              description="합성 재생 데이터 4개 공간의 60분 예측·이상·상태. 형식: /srv/team/service_kit/INTERFACE.md")


def _space(space):
    with LOCK:
        it = STATE["latest"].get(space)
    if it is None:
        raise HTTPException(404, f"알 수 없는 공간: {space} (가능: {', '.join(K.SPACES)})")
    return it


@app.get("/health")
def health():
    with LOCK:
        return dict(STATE["status"])


@app.get("/spaces")
def spaces():
    with LOCK:
        latest = dict(STATE["latest"])
    return [{"space": s, "name": it["name"], "time": it["time"], "co2_now": it["now"]["co2"],
             "co2_60": it["forecast"]["co2"][11] if it["forecast"]["available"] else None,
             "exceed_1000_in_min": it["forecast"].get("exceed_1000_in_min"),
             "occupied": it["state"].get("occupied"), "vent_alert": it["state"].get("vent_alert"),
             "anomaly": it["anomaly"].get("is_anomaly"), "alerts_active": it.get("alerts_active", [])}
            for s, it in latest.items()]


@app.get("/spaces/{space}")
def space_detail(space: str):
    return _space(space)


@app.get("/spaces/{space}/forecast")
def forecast(space: str, minutes: int = Query(30, ge=5, le=60, description="5~60분, 5분 단위"),
             var: str = Query("co2", pattern="^(co2|temperature)$")):
    it = _space(space)
    f = it["forecast"]
    if not f["available"]:
        raise HTTPException(503, f.get("reason", "예측 불가"))
    m = int(round(minutes / 5) * 5)
    return {"space": space, "name": it["name"], "base_time": it["time"], "minutes_ahead": m, "var": var,
            "value": f[var][f["minutes_ahead"].index(m)], "unit": "ppm" if var == "co2" else "℃"}


@app.get("/alerts")
def alerts_now():
    with LOCK:
        return {s: it.get("alerts_active", []) for s, it in STATE["latest"].items()} | \
               {f"real:{k}": v.get("alerts_active", []) for k, v in STATE["real"].get("spaces", {}).items()}


@app.get("/events")
def events(limit: int = Query(50, ge=1, le=500)):
    with LOCK:
        return list(STATE["events"])[-limit:]


@app.get("/real")
def real():
    with LOCK:
        return STATE["real"]


def serve_api():
    uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=API_PORT, log_level="warning")).run()


# ------------------------------------------------------------------ 시작
def main():
    STATE["status"]["started"] = pd.Timestamp.now(tz=KST).isoformat()
    # 기본은 ONNX Runtime(CPU): 이 크기 모델은 가장 빠르고(과제 7), CUDA 런타임을 안 띄워 메모리 약 0.5GB 절약.
    # 메모리 여유가 있으면 .env 에 SVC_BACKEND=tensorrt
    kit = K.Kit(prefer=ENV.get("SVC_BACKEND", "onnxruntime"))
    log.info("모델 실행기: %s", kit.backends())
    schedule = pd.read_csv("/srv/team/data/synthetic/schedule.csv", parse_dates=["start", "end"])
    db = InfluxDBClient(url=ENV["INFLUX_URL"], token=ENV["INFLUX_TOKEN"], org=ENV["INFLUX_ORG"], timeout=30_000)
    q, w = db.query_api(), db.write_api(write_options=SYNCHRONOUS)
    mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="team2-svc")
    mq.username_pw_set(ENV["MQTT_USER"], ENV["MQTT_PASS"])
    mq.will_set(f"{TOPIC}/status", json.dumps({"online": False, "service": "team2-1 추론 서비스"}, ensure_ascii=False),
                qos=1, retain=True)
    mq.reconnect_delay_set(1, 30)
    mq.connect_async(ENV.get("MQTT_HOST", "localhost"), int(ENV.get("MQTT_PORT", 1885)))
    mq.loop_start()
    threading.Thread(target=serve_api, daemon=True).start()
    alerts = Alerts()
    log.info("서비스 시작: API :%d, MQTT %s/#", API_PORT, TOPIC)
    while True:
        now = time.time()
        time.sleep(60 - (now % 60) + 15 if now % 60 > 15 else 15 - now % 60)  # 매분 15초
        try:
            cycle(kit, q, w, mq, alerts, schedule)
        except Exception as e:
            with LOCK:
                st = STATE["status"]
                st.update(failures=st["failures"] + 1, last_error=f"{type(e).__name__}: {e}",
                          last_error_at=pd.Timestamp.now(tz=KST).isoformat())
                status = dict(st)
            log.error("주기 실패 — 다음 주기에 다시 시도\n%s", traceback.format_exc())
            try:
                mq.publish(f"{TOPIC}/status", json.dumps(status, ensure_ascii=False), qos=1, retain=True)
                w.write(bucket="service", record=Point("svc_cycle").field("ok", 0).field("elapsed_s", 0.0)
                        .time(pd.Timestamp.now(tz=KST)))
            except Exception:
                pass


if __name__ == "__main__":
    main()
