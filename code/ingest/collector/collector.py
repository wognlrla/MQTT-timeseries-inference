"""MQTT → InfluxDB 상시 수집기.

메시지 해석 규칙(토픽, 시각, 태그, 필드)은 모두 config.yaml에 있다.
- MQTT: 자동 재연결 + 영속 세션(QoS 1)으로 브로커 재시작·네트워크 끊김 후 자동 복구
- InfluxDB: 쓰기 실패 시 메모리에 보관했다가 재시도
- 중복 메시지(deduplicationId) 제거
- 1분마다 자체 상태를 measurement `collector` 로 기록
"""
import json
import logging
import os
import signal
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
import yaml
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collector")

CFG = yaml.safe_load(open(os.getenv("CONFIG", "config.yaml"), encoding="utf-8"))
MQ, IX = CFG.get("mqtt", {}), CFG.get("influx", {})
BUCKET, ORG = os.environ["INFLUX_BUCKET"], os.environ["INFLUX_ORG"]

stats = dict(received=0, written=0, duplicate=0, unmatched=0, parse_error=0, write_error=0)
pending = deque(maxlen=IX.get("max_pending", 200_000))
lock = threading.Lock()
seen = OrderedDict()
stop = threading.Event()


def get_path(obj, path):
    if path in (None, ""):
        return obj
    for key in str(path).split("."):
        if isinstance(obj, list) and key.isdigit():
            obj = obj[int(key)] if int(key) < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
        if obj is None:
            return None
    return obj


def as_float(v):
    # int/float 혼용 시 InfluxDB field type conflict가 나므로 숫자는 모두 float로 저장
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def parse_time(v):
    if isinstance(v, (int, float)):  # epoch 초/밀리초
        return datetime.fromtimestamp(v / 1000 if v > 1e12 else v, tz=timezone.utc)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            pass
    return None


def is_duplicate(key):
    if key is None:
        return False
    if key in seen:
        return True
    seen[key] = None
    if len(seen) > 20_000:
        seen.popitem(last=False)
    return False


def to_points(src, payload, received_at):
    ts = parse_time(get_path(payload, src.get("time"))) or received_at
    tags = {k: str(v) for k, p in (src.get("tags") or {}).items()
            if (v := get_path(payload, p)) not in (None, "")}
    points = []
    for out in src.get("outputs", []):
        fields = {}
        if "fields_from" in out:
            body = get_path(payload, out["fields_from"]) or {}
            rename, exclude = out.get("rename") or {}, set(out.get("exclude") or [])
            for k, v in body.items():
                if k not in exclude and (f := as_float(v)) is not None:
                    fields[rename.get(k, k)] = f
        for name, path in (out.get("fields") or {}).items():
            if (f := as_float(get_path(payload, path))) is not None:
                fields[name] = f
        if fields:
            p = Point(out["measurement"]).time(ts, WritePrecision.MS)
            for k, v in tags.items():
                p.tag(k, v)
            for k, v in fields.items():
                p.field(k, v)
            points.append(p)
    return points


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        log.error("MQTT 연결 실패: %s", reason_code)
        return
    log.info("MQTT 연결됨 (session_present=%s)", flags.session_present)
    for src in CFG["sources"]:
        client.subscribe(src["topic"], qos=MQ.get("qos", 1))


def on_disconnect(client, userdata, flags, reason_code, properties):
    log.warning("MQTT 연결 끊김: %s — 자동 재연결 시도", reason_code)


def on_message(client, userdata, msg):
    received_at = datetime.now(timezone.utc)
    src = next((s for s in CFG["sources"] if mqtt.topic_matches_sub(s["topic"], msg.topic)), None)
    with lock:
        stats["received"] += 1
        if src is None:
            stats["unmatched"] += 1
            return
        try:
            payload = json.loads(msg.payload)
        except Exception:
            stats["parse_error"] += 1
            return
        if src.get("dedup_key") and is_duplicate(get_path(payload, src["dedup_key"])):
            stats["duplicate"] += 1
            return
        pending.extend(to_points(src, payload, received_at))


def writer_loop(write_api):
    batch_size, flush_s = IX.get("batch_size", 500), IX.get("flush_seconds", 2)
    backoff = 1
    while not stop.is_set() or pending:
        with lock:
            batch = [pending.popleft() for _ in range(min(batch_size, len(pending)))]
        if not batch:
            stop.wait(flush_s)
            continue
        try:
            write_api.write(bucket=BUCKET, org=ORG, record=batch)
            with lock:
                stats["written"] += len(batch)
            backoff = 1
        except Exception as e:
            with lock:
                stats["write_error"] += 1
                pending.extendleft(reversed(batch))  # 순서 유지하며 되돌려 놓기
            log.error("InfluxDB 쓰기 실패(%s), %ds 후 재시도. 보관 중 %d건", e, backoff, len(pending))
            stop.wait(backoff)
            backoff = min(backoff * 2, 60)


def heartbeat_loop(client):
    while not stop.wait(60):
        with lock:
            snap = dict(stats, pending=len(pending), mqtt_connected=int(client.is_connected()))
        log.info("상태 %s", snap)
        p = Point("collector").tag("name", MQ.get("client_id", "collector"))
        for k, v in snap.items():
            p.field(k, float(v))
        with lock:
            pending.append(p.time(datetime.now(timezone.utc), WritePrecision.S))


def main():
    influx = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"], org=ORG)
    write_api = influx.write_api(write_options=SYNCHRONOUS)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=MQ.get("client_id", "collector"), clean_session=False)
    client.username_pw_set(os.environ["MQTT_USER"], os.environ["MQTT_PASS"])
    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    threading.Thread(target=writer_loop, args=(write_api,), daemon=True).start()
    threading.Thread(target=heartbeat_loop, args=(client,), daemon=True).start()

    def shutdown(*_):
        log.info("종료 요청 — 남은 데이터 기록 후 종료")
        client.disconnect()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    client.connect_async(os.environ["MQTT_HOST"], int(os.getenv("MQTT_PORT", 1883)),
                         keepalive=MQ.get("keepalive", 30))
    client.loop_forever(retry_first_connection=True)

    stop.set()
    deadline = time.time() + 10
    while pending and time.time() < deadline:
        time.sleep(0.2)
    influx.close()


if __name__ == "__main__":
    main()
