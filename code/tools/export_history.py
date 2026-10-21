"""운영진 InfluxDB(iot-bucket)의 과거 원본을 월별 Parquet로 내려받는다.

- 정제하지 않은 원본 그대로 저장한다(정제는 과제 2의 공통 전처리에서).
- 공용 서버 부담을 줄이려고 '월 × 측정항목' 단위로 나눠 조회하고, 사이사이 쉰다.
- 이미 받은 달은 건너뛴다(이어받기). 진행 중인 달은 --refresh-current 로 다시 받는다.

사용: python tools/export_history.py [--from 2025-08] [--to 2026-10] [--out data/raw/influx_history] [--refresh-current]
결과: <out>/history_YYYY-MM.parquet  (long 형식: time, metric, value, device_id, location, point, spot, gateway_id)
      <out>/manifest.json            (내려받은 시각, 기간, 월·항목별 건수)
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

# 숫자 측정값만. 펌웨어 버전·시리얼 같은 문자열 메타데이터는 제외한다.
METRICS = ["co2", "temperature", "humidity", "pressure", "tvoc", "illumination",
           "infrared", "infrared_and_visible", "activity", "LAeq", "LAI", "LAImax",
           "magnet_status", "battery", "rssi", "snr"]
COLUMNS = ["time", "metric", "value", "device_id", "location", "point", "spot", "gateway_id"]


def month_range(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


TAGS = ["device_id", "location", "point", "spot", "gateway_id"]
# 2025년 데이터 일부는 위치·센서 ID를 태그가 아니라 같은 시각의 문자열 필드로 기록했다
META_FIELDS = ["device_id", "location", "point", "spot"]


def _frame(q, flux):
    df = q.query_data_frame(flux)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
    return df


def fetch(q, bucket, metric, start, stop):
    base = f'from(bucket: "{bucket}") |> range(start: {start}, stop: {stop}) |> filter(fn: (r) => r._measurement == "{metric}")'
    keep = '|> group() |> keep(columns: ["_time", "_field", "_value", ' + ", ".join(f'"{t}"' for t in TAGS) + "])"
    val = _frame(q, f'''import "types"
{base}
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => types.isType(v: r._value, type: "float") or types.isType(v: r._value, type: "int"))
  |> map(fn: (r) => ({{r with _value: float(v: r._value)}}))
  {keep}''')
    if val.empty:
        return pd.DataFrame(columns=COLUMNS)
    for t in TAGS:
        if t not in val.columns:
            val[t] = None

    field_filter = " or ".join(f'r._field == "{f}"' for f in META_FIELDS)
    meta = _frame(q, f'{base} |> filter(fn: (r) => {field_filter}) {keep}')
    if not meta.empty:
        for t in TAGS:
            if t not in meta.columns:
                meta[t] = None
        # 태그가 비어 있는 행끼리 시각으로 짝지어, 필드에 적힌 위치·ID로 채운다
        meta = meta[meta[META_FIELDS].isna().all(axis=1)]
        meta = (meta.pivot_table(index="_time", columns="_field", values="_value", aggfunc="first")
                .add_prefix("f_").reset_index())
        val = val.merge(meta, on="_time", how="left")
        untagged = val[META_FIELDS].isna().all(axis=1)
        for f in META_FIELDS:
            if f"f_{f}" in val.columns:
                val.loc[untagged, f] = val.loc[untagged, f"f_{f}"]

    df = val.rename(columns={"_time": "time", "_value": "value"})
    df["metric"] = metric
    return df[COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2025-08")
    ap.add_argument("--to", dest="end", default=datetime.now().strftime("%Y-%m"))
    ap.add_argument("--out", default="data/raw/influx_history")
    ap.add_argument("--refresh-current", action="store_true")
    ap.add_argument("--pause", type=float, default=1.0, help="쿼리 사이 대기(초)")
    args = ap.parse_args()

    load_dotenv(os.path.join(os.getcwd(), ".env"))
    bucket = os.environ["INFLUX_BUCKET"]
    client = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"],
                            org=os.environ["INFLUX_ORG"], timeout=600_000)
    q = client.query_api()
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"months": {}}
    manifest.update(source=f"{os.environ['INFLUX_URL']} bucket={bucket}", metrics=METRICS)

    now = datetime.now(timezone.utc)
    to_ym = tuple(map(int, args.end.split("-")))
    for y, m in month_range(tuple(map(int, args.start.split("-"))), to_ym):
        key = f"{y:04d}-{m:02d}"
        path = os.path.join(args.out, f"history_{key}.parquet")
        is_current = (y, m) == (now.year, now.month)
        if os.path.exists(path) and not (is_current and args.refresh_current):
            print(f"{key} 이미 있음 — 건너뜀")
            continue
        start = f"{key}-01T00:00:00Z"
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        stop = f"{ny:04d}-{nm:02d}-01T00:00:00Z"
        parts, counts = [], {}
        t0 = time.time()
        for metric in METRICS:
            for attempt in range(3):
                try:
                    df = fetch(q, bucket, metric, start, stop)
                    break
                except Exception as e:
                    print(f"  {key} {metric} 실패({str(e)[:120]}), 재시도 {attempt + 1}/3")
                    time.sleep(10 * (attempt + 1))
            else:
                raise SystemExit(f"{key} {metric} 3회 실패 — 중단(다시 실행하면 이 달부터 이어받음)")
            counts[metric] = len(df)
            if len(df):
                parts.append(df)
            time.sleep(args.pause)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
        out["time"] = pd.to_datetime(out["time"], utc=True)
        for c in ("metric", "device_id", "location", "point", "spot", "gateway_id"):
            out[c] = out[c].astype("string").astype("category")
        out = out.sort_values(["time", "metric", "device_id"]).reset_index(drop=True)
        out.to_parquet(path, index=False, compression="zstd")
        manifest["months"][key] = {"rows": int(len(out)), "per_metric": counts,
                                   "exported_at": now.isoformat(), "partial": is_current}
        json.dump(manifest, open(manifest_path, "w"), ensure_ascii=False, indent=1)
        print(f"{key}: {len(out):,}행, {os.path.getsize(path) / 1e6:.1f}MB, {time.time() - t0:.0f}초")
    client.close()


if __name__ == "__main__":
    main()
