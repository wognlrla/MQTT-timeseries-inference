"""과제 8: 가동 기록 요약 (주기별 처리 시간, 경보 발생 내역, 실패·누락 주기).

사용: /srv/team/venv/bin/python summary_12h.py [시작 KST, 기본=서비스 시작 시각] [시간, 기본 12]
결과: 화면 + out/summary_<시작>.md
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction

warnings.simplefilter("ignore", MissingPivotFunction)
ROOT = Path("/srv/team/service")
env = dict(l.split("=", 1) for l in (ROOT / ".env").read_text().splitlines() if "=" in l and not l.startswith("#"))


def sysd(prop):
    return subprocess.run(["systemctl", "show", "team2-svc", "-p", prop, "--value"], capture_output=True, text=True).stdout.strip()


start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp(sysd("ActiveEnterTimestamp").rsplit(" ", 1)[0].split(" ", 1)[1])
hours = float(sys.argv[2]) if len(sys.argv) > 2 else 12
end = start + pd.Timedelta(hours=hours)
q = InfluxDBClient(url=env["INFLUX_URL"], token=env["INFLUX_TOKEN"], org=env["INFLUX_ORG"]).query_api()
rng = f'range(start: {start.tz_localize("Asia/Seoul").isoformat()}, stop: {end.tz_localize("Asia/Seoul").isoformat()})'


def df(flux):
    d = q.query_data_frame(flux)
    return pd.concat(d) if isinstance(d, list) else d


cyc = df(f'from(bucket:"service") |> {rng} |> filter(fn:(r)=> r._measurement=="svc_cycle") '
         f'|> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")')
ev = df(f'from(bucket:"service") |> {rng} |> filter(fn:(r)=> r._measurement=="svc_event")')
cyc["t"] = pd.to_datetime(cyc["_time"]).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
ok = cyc[cyc.ok == 1]
fail = cyc[cyc.ok == 0]
expected = int(hours * 60)
minutes = set(ok.t.dt.floor("min"))
missing = [m for m in pd.date_range(start.ceil("min"), end.floor("min") - pd.Timedelta(minutes=1), freq="min")
           if m not in minutes and m not in set(fail.t.dt.floor("min"))]
e = ev[["_time", "key", "type", "kind"]].copy() if len(ev) else pd.DataFrame(columns=["_time", "key", "type", "kind"])
if len(e):
    e["at"] = pd.to_datetime(e["_time"]).dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.strftime("%m-%d %H:%M")
lines = [
    f"# 가동 기록 요약 ({start:%Y-%m-%d %H:%M} ~ {end:%H:%M} KST, {hours:g}시간)", "",
    "| 항목 | 값 |", "|---|---|",
    f"| 성공 주기 | {len(ok)} / 예상 {expected} |",
    f"| 실패 주기 (예외를 잡고 다음 주기에 회복) | {len(fail)} |",
    f"| 기록이 없는 분 (서비스가 돌지 않은 분) | {len(missing)} |",
    f"| 처리 시간 평균 / p95 / 최대 | {ok.elapsed_s.mean():.3f} / {ok.elapsed_s.quantile(.95):.3f} / {ok.elapsed_s.max():.3f} 초 |",
    f"| systemd 재시작 횟수 | {sysd('NRestarts')} |",
    f"| 메모리 (현재 / 최대) | {int(sysd('MemoryCurrent') or 0) / 2**20:.0f} / {int(sysd('MemoryPeak') or 0) / 2**20:.0f} MB |",
    "", "## 경보 이벤트 (raised / reminder / cleared)", "",
]
if len(e):
    t = e.groupby(["key", "type", "kind"]).size().unstack("kind", fill_value=0)
    lines += ["| 공간 | 경보 | " + " | ".join(t.columns) + " |", "|---|---" + "|---" * len(t.columns) + "|"]
    lines += [f"| {k} | {ty} | " + " | ".join(str(v) for v in r) + " |" for (k, ty), r in zip(t.index, t.values)]
    lines += ["", "## 경보 발생 시각 (raised·cleared)", ""]
    lines += [f"- {r.at} {r.key} {r.type} **{r.kind}**" for r in e[e.kind != "reminder"].sort_values("at").itertuples()]
else:
    lines.append("경보 없음")
if len(missing):
    lines += ["", f"## 기록이 없는 분 ({len(missing)}개)", "", ", ".join(m.strftime("%H:%M") for m in missing[:60])]
txt = "\n".join(lines)
print(txt)
(ROOT / "out").mkdir(exist_ok=True)
(ROOT / "out" / f"summary_{start:%Y%m%d_%H%M}.md").write_text(txt)
