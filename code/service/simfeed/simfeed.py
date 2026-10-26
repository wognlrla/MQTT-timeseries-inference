"""합성 센서 재생기: 시험 주간(2026-10-05 월 ~ 10-11 일)의 sensors.csv를 '지금 요일·시각'에 맞춰 MQTT로 발행한다.

- 매핑: 현재 시각 t → 같은 요일·같은 시각의 시험 주간 시각. 매주 반복한다.
  (서비스의 일정 매핑과 같은 규칙. service_kit/kit.py의 SIM_WEEK_START와 같아야 한다)
- 시작 시 최근 BACKFILL_H 시간을 먼저 한꺼번에 발행해, 서비스가 바로 2시간 이상의 과거를 볼 수 있게 한다.
- 원본에 있는 결측·깨진 값·고장(시험 주간 17건)이 그대로 재생된다.
- 토픽 sim/sensors/<공간>, 메시지 {"ts": ISO(+09:00), "room": 공간, "co2": .., "temp": .., "humidity": ..}
"""
import json
import os
import time

import pandas as pd
import paho.mqtt.client as mqtt

SIM_WEEK_START = pd.Timestamp("2026-10-05")
BACKFILL_H = float(os.getenv("BACKFILL_H", "6"))
TICK_S = 5

df = pd.read_csv("/data/sensors.csv", parse_dates=["time"])
df = df[(df.time >= SIM_WEEK_START) & (df.time < SIM_WEEK_START + pd.Timedelta(days=7))]
df["sim_off"] = df.time - SIM_WEEK_START  # 시험 주간 시작부터의 경과
df = df.sort_values("sim_off").reset_index(drop=True)


def week_start(t):
    return (t - pd.Timedelta(days=t.weekday())).normalize()


def rows_between(a, b):
    """벽시계 (a, b] 에 해당하는 재생 행과 매핑된 시각."""
    out = []
    ws = week_start(a)
    while ws <= b:
        lo, hi = a - ws, b - ws
        m = df[(df.sim_off > lo) & (df.sim_off <= hi)]
        if len(m):
            out.append(m.assign(wall=ws + m.sim_off))
        ws += pd.Timedelta(days=7)
    return pd.concat(out) if out else df.iloc[0:0].assign(wall=pd.NaT)


def publish(client, rows):
    for r in rows.itertuples():
        msg = {"ts": r.wall.isoformat() + "+09:00", "room": r.space}
        for k, v in (("co2", r.co2), ("temp", r.temperature), ("humidity", r.humidity)):
            if pd.notna(v):
                msg[k] = float(v)
        client.publish(f"sim/sensors/{r.space}", json.dumps(msg), qos=1)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="team2-simfeed")
client.username_pw_set(os.environ["MQTT_USER"], os.environ["MQTT_PASS"])
client.reconnect_delay_set(1, 30)
client.connect_async(os.environ["MQTT_HOST"], int(os.getenv("MQTT_PORT", 1883)))
client.loop_start()
while not client.is_connected():
    time.sleep(0.5)

# 부팅 직후 시계가 아직 맞춰지지 않았으면(1970년 등) 맞춰질 때까지 기다린다
while pd.Timestamp.now().year < 2025:
    time.sleep(5)
now = pd.Timestamp.now().floor("s")
back = rows_between(now - pd.Timedelta(hours=BACKFILL_H), now)
publish(client, back)
print(f"{now} 재생 시작: 최근 {BACKFILL_H}시간 {len(back)}건 먼저 발행 (현재 = 시험 주간 {SIM_WEEK_START + (now - week_start(now))})", flush=True)
last, sent, t_log = now, 0, time.time()
while True:
    time.sleep(TICK_S)
    now = pd.Timestamp.now().floor("s")
    # 시계가 크게 바뀌면(NTP 동기화 등) 그 사이를 몰아서 재생하지 않고 건너뛴다
    if not (pd.Timedelta(0) <= now - last <= pd.Timedelta(minutes=10)):
        print(f"{now} 시계 변화 감지({last} → {now}), 밀린 구간은 건너뜀", flush=True)
        last = now
        continue
    rows = rows_between(last, now)
    publish(client, rows)
    sent += len(rows)
    last = now
    if time.time() - t_log > 600:
        print(f"{now} 10분 동안 {sent}건 발행, 연결={client.is_connected()}", flush=True)
        sent, t_log = 0, time.time()
