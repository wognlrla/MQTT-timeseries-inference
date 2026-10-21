"""수집 현황 보고: 공간·센서·항목별 건수, 마지막 수신, 커버율, 최장 공백.

사용: docker compose exec collector python coverage.py [시간, 기본 24] [항목,쉼표구분 기본 co2,temperature,humidity]
커버율 = (데이터가 1건 이상 있는 1분 칸 수) / (조회 구간의 1분 칸 수)
"""
import os
import sys
from datetime import datetime, timezone

from influxdb_client import InfluxDBClient

HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
FIELDS = (sys.argv[2] if len(sys.argv) > 2 else "co2,temperature,humidity").split(",")
BUCKET = os.environ["INFLUX_BUCKET"]

q = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"],
                   org=os.environ["INFLUX_ORG"]).query_api()
field_filter = " or ".join(f'r._field == "{f}"' for f in FIELDS)
base = (f'from(bucket: "{BUCKET}") |> range(start: -{HOURS}h) '
        f'|> filter(fn: (r) => r._measurement == "env" and ({field_filter})) '
        f'|> keep(columns: ["_time", "_value", "_field", "location", "device"])')


def collect(flux, col="_value"):
    out = {}
    for t in q.query(flux):
        for r in t.records:
            out[(r.values.get("location"), r.values.get("device"), r.get_field())] = r.values.get(col)
    return out


counts = collect(base + " |> count()")
minutes = collect(base + ' |> aggregateWindow(every: 1m, fn: count, createEmpty: false) |> count()')
last = collect(base + " |> last()", "_time")
gaps = collect(base + ' |> elapsed(unit: 1s) |> max(column: "elapsed")', "elapsed")

now = datetime.now(timezone.utc)
total_min = HOURS * 60
print(f"조회 구간: 최근 {HOURS}시간 ({total_min}개 1분 칸), 기준 {now.astimezone():%Y-%m-%d %H:%M %Z}\n")
print(f"{'공간':<8}{'센서':<20}{'항목':<12}{'건수':>6}{'커버율':>8}{'최장공백':>9}  마지막 수신(전)")
for key in sorted(counts, key=lambda k: tuple(map(str, k))):
    loc, dev, field = key
    cov = 100 * (minutes.get(key) or 0) / total_min
    gap = gaps.get(key)
    ago = (now - last[key]).total_seconds() / 60 if key in last else None
    print(f"{str(loc):<8}{str(dev):<20}{field:<12}{counts[key]:>6}{cov:>7.1f}%"
          f"{(f'{gap / 60:.0f}분' if gap else '-'):>9}  {f'{ago:.1f}분' if ago is not None else '-'}")
