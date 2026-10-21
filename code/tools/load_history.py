"""과거 원본 Parquet(export_history.py 결과)를 Jetson InfluxDB의 history 버킷에 적재한다.

저장 구조는 실시간 수집기(sensors 버킷)와 맞춘다.
  센서값   measurement=env     tags: location, point, spot, dev_eui   field: <측정항목>
  통신품질 measurement=uplink  tags: location, point, spot, dev_eui, gateway_id   field: rssi, snr
같은 시리즈·같은 시각의 중복 행은 InfluxDB가 하나로 합친다(마지막 값 유지).

사용(보드에서, 수집 스택 네트워크 안의 컨테이너로 실행):
  docker run --rm --network ingest_default --env-file /srv/team/ingest/.env \
    -e INFLUX_URL=http://influxdb:8086 -v /srv/team/data:/data:ro -v /srv/team/tools:/tools:ro \
    python:3.12-slim sh -c "pip install -q pandas pyarrow influxdb-client && \
    python /tools/load_history.py /data/raw/influx_history"
"""
import glob
import json
import os
import sys
import time

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

SRC = sys.argv[1] if len(sys.argv) > 1 else "/data/raw/influx_history"
BUCKET = os.getenv("HISTORY_BUCKET", "history")
UPLINK = {"rssi", "snr"}
CHUNK = 50_000


def esc_tag(s: pd.Series) -> pd.Series:
    # 라인 프로토콜: 태그 값의 공백·쉼표·등호는 백슬래시로 이스케이프
    return (s.astype("string").str.replace("\\", "\\\\", regex=False)
            .str.replace(" ", "\\ ", regex=False).str.replace(",", "\\,", regex=False)
            .str.replace("=", "\\=", regex=False))


def to_lines(df: pd.DataFrame) -> pd.Series:
    meas = pd.Series("env", index=df.index).where(~df["metric"].isin(UPLINK), "uplink")
    tags = meas
    for col, name in (("location", "location"), ("point", "point"), ("spot", "spot"),
                      ("device_id", "dev_eui"), ("gateway_id", "gateway_id")):
        v = esc_tag(df[col])
        tags = tags + ("," + name + "=" + v).fillna("")
    ns = df["time"].astype("int64").astype(str)
    return tags + " " + df["metric"].astype(str) + "=" + df["value"].map(repr) + " " + ns


def main():
    client = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"],
                            org=os.environ["INFLUX_ORG"], timeout=120_000)
    write = client.write_api(write_options=SYNCHRONOUS)
    report = {}
    start_month = os.getenv("FROM_MONTH", "0000-00")  # 이어서 넣을 때: FROM_MONTH=2026-08
    for path in sorted(glob.glob(os.path.join(SRC, "history_*.parquet"))):
        month = os.path.basename(path)[8:15]
        if month < start_month:
            continue
        t0 = time.time()
        df = pd.read_parquet(path)
        df = df[df["value"].notna()].reset_index(drop=True)
        total = 0
        # 보드 메모리가 작아서(스왑 없음) 한 달치를 한 번에 문자열로 만들지 않고 나눠서 처리한다
        for i in range(0, len(df), CHUNK):
            lines = to_lines(df.iloc[i:i + CHUNK]).tolist()
            for attempt in range(5):
                try:
                    write.write(bucket=BUCKET, record=lines)
                    break
                except Exception as e:
                    print(f"  {month} 쓰기 실패({str(e)[:100]}), 재시도 {attempt + 1}/5", flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                raise SystemExit(f"{month} 적재 실패 — FROM_MONTH={month} 로 다시 실행(덮어써도 중복 없음)")
            total += len(lines)
        del df
        report[month] = total
        print(f"{month}: {total:,}건, {time.time() - t0:.0f}초", flush=True)
    print(json.dumps({"bucket": BUCKET, "total": sum(report.values()), "months": report}, ensure_ascii=False))
    client.close()


if __name__ == "__main__":
    main()
