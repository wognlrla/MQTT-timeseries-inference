"""MQTT로 들어오는 센서들을 N분간 구독해 기종·공간·측정항목·주기를 요약한다.

사용: python survey_mqtt.py [분, 기본 5]
결과: 화면 요약 + mqtt_samples.jsonl (기종별 원본 메시지 1개씩)
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 5

devices = defaultdict(lambda: {"times": [], "keys": set(), "fcnt": []})
topics = set()
samples = {}


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"연결 결과: {reason_code}")
    client.subscribe(os.getenv("MQTT_TOPIC", "#"), qos=1)


def on_message(client, userdata, msg):
    topics.add(msg.topic)
    try:
        p = json.loads(msg.payload)
    except Exception:
        print(f"[JSON 아님] {msg.topic}: {msg.payload[:80]!r}")
        return
    info = p.get("deviceInfo", {})
    name = info.get("deviceName", msg.topic)
    d = devices[name]
    d["profile"] = info.get("deviceProfileName")
    d["location"] = (info.get("tags") or {}).get("location")
    d["times"].append(p.get("time"))
    d["fcnt"].append(p.get("fCnt"))
    d["keys"].update((p.get("object") or {}).keys())
    d["last"] = p.get("object")
    samples.setdefault(d["profile"], {"topic": msg.topic, "payload": p})
    print(f"{datetime.now():%H:%M:%S} {d['location']} {name} {p.get('object')}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(os.getenv("MQTT_USER"), os.getenv("MQTT_PASS"))
client.on_connect = on_connect
client.on_message = on_message
client.connect(os.getenv("MQTT_HOST"), int(os.getenv("MQTT_PORT", 1883)))
client.loop_start()
print(f"{MINUTES}분간 수집합니다...")
time.sleep(MINUTES * 60)
client.loop_stop()


def interval(ts):
    ts = sorted(datetime.fromisoformat(t[:19]) for t in ts if t)  # 초 단위면 충분
    if len(ts) < 2:
        return "-"
    gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    return f"{sum(gaps) / len(gaps):.0f}s"


print("\n=== 장치 요약 ===")
print(f"{'location':<12}{'device':<24}{'profile':<14}{'msgs':>5}{'주기':>7}  측정항목")
for name, d in sorted(devices.items(), key=lambda x: (str(x[1]["location"]), x[0])):
    print(f"{str(d['location']):<12}{name:<24}{str(d['profile']):<14}"
          f"{len(d['times']):>5}{interval(d['times']):>7}  {sorted(d['keys'])}")
print(f"\n토픽 예시: {sorted(topics)[:3]}")

with open("mqtt_samples.jsonl", "w") as f:
    for s in samples.values():
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print("기종별 샘플 메시지 -> mqtt_samples.jsonl")
