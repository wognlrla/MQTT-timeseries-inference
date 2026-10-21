"""과제 1: 24시간 수집 현황 보고 (보드에서 호스트 venv로 실행).

1) 공간·항목별 커버율 = 그 공간 센서 중 하나라도 값이 있는 1분 칸 / 전체 1분 칸 (과제 정의)
2) 센서별 수신률 = 받은 건수 / (24시간 ÷ 그 센서의 보고 주기 중앙값)
3) 공간·항목별로 5분 넘게 비어 있는 구간과, 같은 시각 수집기 상태(1분 상태 기록)의 끊김을 대조

사용: /srv/team/venv/bin/python coverage24.py "2026-10-20 10:21" 24
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction

warnings.simplefilter("ignore", MissingPivotFunction)
env = dict(l.split("=", 1) for l in Path("/srv/team/service/.env").read_text().splitlines() if "=" in l and not l.startswith("#"))
start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2026-10-20 10:21")
hours = float(sys.argv[2]) if len(sys.argv) > 2 else 24
end = start + pd.Timedelta(hours=hours)
q = InfluxDBClient(url=env["INFLUX_URL"], token=env["INFLUX_TOKEN"], org=env["INFLUX_ORG"], timeout=120_000).query_api()
rng = f'range(start: {start.tz_localize("Asia/Seoul").isoformat()}, stop: {end.tz_localize("Asia/Seoul").isoformat()})'


def frame(flux):
    d = q.query_data_frame(flux)
    d = pd.concat(d) if isinstance(d, list) else d
    if len(d):
        d["t"] = pd.to_datetime(d["_time"]).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return d


d = frame(f'from(bucket:"sensors") |> {rng} |> filter(fn:(r)=> r._measurement=="env" and '
          f'(r._field=="co2" or r._field=="temperature" or r._field=="humidity")) '
          f'|> keep(columns:["_time","_field","location","device"])')
minutes = int(hours * 60)
d["m"] = d.t.dt.floor("min")
cov = d.groupby(["location", "_field"]).m.nunique().unstack("_field").div(minutes).mul(100).round(2)

dev = d[d._field == "temperature"].sort_values("t")
rows = []
for (loc, name), g in dev.groupby(["location", "device"]):
    iv = g.t.diff().dt.total_seconds().div(60)
    med = iv.median()
    rows.append({"공간": loc, "센서": name, "받은 건수": len(g), "보고 주기(분, 중앙값)": round(med, 1),
                 "수신률(%)": round(100 * len(g) / (minutes / med), 1) if med else None,
                 "최장 공백(분)": round(iv.max(), 0)})
devtab = pd.DataFrame(rows).sort_values(["공간", "센서"])

gaps = []
for (loc, f), g in d.groupby(["location", "_field"]):
    ms = pd.Series(sorted(g.m.unique()))
    for a, b in zip(ms[:-1], ms[1:]):
        if (b - a) > pd.Timedelta(minutes=5):
            gaps.append({"공간": loc, "항목": f, "시작": a, "끝": b, "길이(분)": int((b - a).total_seconds() / 60)})
gaps = pd.DataFrame(gaps)

hb = frame(f'from(bucket:"sensors") |> {rng} |> filter(fn:(r)=> r._measurement=="collector" and r._field=="mqtt_connected")')
hb_gaps = []
if len(hb):
    hs = hb.sort_values("t")
    for a, b, v in zip(hs.t[:-1], hs.t[1:], hs._value[1:]):
        if (b - a) > pd.Timedelta(minutes=2):
            hb_gaps.append({"수집기 상태 기록 끊김 시작": a, "끝": b, "길이(분)": round((b - a).total_seconds() / 60, 1)})
    disc = hs[hs._value == 0]
    for t in disc.t:
        hb_gaps.append({"수집기 상태 기록 끊김 시작": t, "끝": t, "길이(분)": 0.0, "비고": "MQTT 연결 끊김 기록"})
hb_gaps = pd.DataFrame(hb_gaps)

pd.set_option("display.width", 220)
print(f"## 기간 {start} ~ {end} ({hours:g}시간, 1분 칸 {minutes}개)\n")
print("## 공간·항목별 1분 커버율(%)\n", cov.to_string())
print("\n## 센서별 수신률\n", devtab.to_string(index=False))
print("\n## 5분 넘게 빈 구간 (공간·항목)\n", gaps.sort_values("길이(분)", ascending=False).head(20).to_string(index=False) if len(gaps) else "없음")
print("\n## 수집기 상태 기록 끊김\n", hb_gaps.to_string(index=False) if len(hb_gaps) else "없음")
out = Path("/srv/team/ingest/out")
out.mkdir(exist_ok=True)
cov.to_csv(out / "coverage24_space.csv")
devtab.to_csv(out / "coverage24_device.csv", index=False)
gaps.to_csv(out / "coverage24_gaps.csv", index=False)
hb_gaps.to_csv(out / "coverage24_collector_gaps.csv", index=False)
