"""InfluxDB 2.x에 이미 무엇이 어떤 구조로 쌓여 있는지 확인한다.

사용: python inspect_influx.py
"""
import os

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()
client = InfluxDBClient(url=os.getenv("INFLUX_URL"), token=os.getenv("INFLUX_TOKEN"),
                        org=os.getenv("INFLUX_ORG") or None, timeout=30_000)

print("서버 상태:", client.ping(), "/ 버전:", client.version())

# 버킷 하나만 읽을 수 있는 토큰은 org 목록을 못 본다 → 버킷의 org_id로 대신한다
buckets = [b for b in client.buckets_api().find_buckets().buckets if not b.name.startswith("_")]
print("\n=== bucket (보관기간) ===")
for b in buckets:
    rules = b.retention_rules[0].every_seconds if b.retention_rules else 0
    print(f"  {b.name}  org_id={b.org_id}  ({'무기한' if not rules else f'{rules // 86400}일'})")
org = os.getenv("INFLUX_ORG") or buckets[0].org_id

q = client.query_api()
targets = [os.getenv("INFLUX_BUCKET")] if os.getenv("INFLUX_BUCKET") else [b.name for b in buckets]
for bucket in targets:
    print(f"\n######## bucket: {bucket} ########")
    rng = "start: " + os.getenv("RANGE", "-365d")
    meas = []
    for t in q.query(f'import "influxdata/influxdb/schema"\n'
                     f'schema.measurements(bucket: "{bucket}", {rng})', org=org):
        meas = [r.get_value() for r in t.records]
    print("measurements:", meas)
    for m in meas[:30]:
        tags = [r.get_value() for t in q.query(
            f'import "influxdata/influxdb/schema"\n'
            f'schema.measurementTagKeys(bucket: "{bucket}", measurement: "{m}", {rng})', org=org)
            for r in t.records if not r.get_value().startswith("_")]
        fields = [r.get_value() for t in q.query(
            f'import "influxdata/influxdb/schema"\n'
            f'schema.measurementFieldKeys(bucket: "{bucket}", measurement: "{m}", {rng})', org=org)
            for r in t.records]
        print(f"\n  [{m}]\n    tags:   {tags}\n    fields: {fields}")
        for tag in ("location", "device_name", "deviceName", "device_profile_name"):
            if tag in tags:
                vals = [r.get_value() for t in q.query(
                    f'import "influxdata/influxdb/schema"\n'
                    f'schema.measurementTagValues(bucket: "{bucket}", measurement: "{m}", '
                    f'tag: "{tag}", {rng})', org=org) for r in t.records]
                print(f"    {tag} = {vals}")
        first = q.query(f'from(bucket: "{bucket}") |> range({rng}) '
                        f'|> filter(fn: (r) => r._measurement == "{m}") |> group() '
                        f'|> first(column: "_time")', org=org)
        last = q.query(f'from(bucket: "{bucket}") |> range({rng}) '
                       f'|> filter(fn: (r) => r._measurement == "{m}") |> group() '
                       f'|> last(column: "_time")', org=org)
        for name, res in (("최초", first), ("최신", last)):
            for t in res:
                for r in t.records:
                    print(f"    {name}: {r.get_time()}  {r.get_field()}={r.get_value()}  "
                          f"{ {k: v for k, v in r.values.items() if not k.startswith('_') and k not in ('result', 'table')} }")
