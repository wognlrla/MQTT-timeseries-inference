"""과제 6: 사전학습 시계열 모델(Chronos-Bolt) 추가 학습 없이 CO₂ 60분 예측 → 기준선·GRU와 비교.

- 입력: 공간별 5분 격자 CO₂ 과거값만(과제 2 전처리, 긴 결측은 NaN 그대로 — Chronos가 결측을 처리한다).
- 평가 시점·정답: 과제 2 eval_points / 과제 3 all_predictions와 같다.
- 과거 길이(24칸=2시간, 288칸=하루, 2016칸=일주일)는 **검증 기간** 60분 MAE로 고르고, 시험 기간은 고른 설정만 보고한다.
- 예측 구간: 10%·90% 분위 → 실제값이 그 사이에 들어온 비율(이상적으로 80%).
- 비용: 보드에서 1회 추론 지연(배치 1·4, 평균·p95), GPU 메모리, 프로세스 메모리 증가량.

사용: /srv/team/venv/bin/python run.py
"""
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

OUT = Path("/srv/team/task6/out")
MODELS = ["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small", "amazon/chronos-bolt-base"]
CTX = [24, 288, 2016]
H = 12
HSTEP = {5: 0, 15: 2, 30: 5, 60: 11}
Q = [0.1, 0.5, 0.9]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024


def series():
    g = pd.read_csv("/srv/team/task2/out/grid_5min.csv.gz", parse_dates=["time"])
    return {s: d.set_index("time").co2.sort_index() for s, d in g.groupby("space")}


def contexts(ser, pts, L):
    out = np.full((len(pts), L), np.nan, np.float32)
    for i, (t, s) in enumerate(zip(pts.time, pts.space)):
        x = ser[s]
        j = x.index.get_loc(t)
        seg = x.values[max(0, j - L + 1): j + 1]
        out[i, L - len(seg):] = seg
    return torch.tensor(out)


def forecast(pipe, ctx, bs=256):
    qs = []
    for i in range(0, len(ctx), bs):
        q, _ = pipe.predict_quantiles(ctx[i:i + bs], prediction_length=H, quantile_levels=Q)
        qs.append(q.float().cpu().numpy())
    return np.concatenate(qs)  # [N, H, 3]


def latency(fn, n=100, warm=10):
    for _ in range(warm):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(ts)), 2), round(float(np.percentile(ts, 95)), 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    part_dir = OUT / "parts"
    part_dir.mkdir(exist_ok=True)
    ser = series()
    pts = pd.read_csv("/srv/team/task2/out/eval_points.csv", parse_dates=["time"])
    allp = pd.read_csv("/srv/team/task3/out/all_predictions.csv.gz", parse_dates=["time"])
    truth = allp[(allp.model == "persistence") & (allp["var"] == "co2")][["time", "space", "h", "split", "fault_window", "y"]]

    for name in MODELS:
        short = name.split("/")[-1]
        todo = [(L, sp) for L in CTX for sp in ["val", "test"] if not (part_dir / f"{short}_{L}_{sp}.csv.gz").exists()]
        if not todo:
            continue
        pipe = BaseChronosPipeline.from_pretrained(name, device_map=DEV, torch_dtype=torch.float32)
        for L, split in todo:
            p = pts[pts.split == split].reset_index(drop=True)
            # 보드 메모리(CPU·GPU 공유 8GB)에 맞춰 긴 문맥·큰 모델은 배치를 줄인다
            bs = max(16, int(256 * 288 / max(L, 288) / (4 if "base" in short else 1)))
            t0 = time.time()
            q = forecast(pipe, contexts(ser, p, L), bs=bs)
            rows = [pd.DataFrame({"time": p.time, "space": p.space, "h": h, "model": short, "ctx": L,
                                  "q10": q[:, k, 0], "q50": q[:, k, 1], "q90": q[:, k, 2]}) for h, k in HSTEP.items()]
            pd.concat(rows).to_csv(part_dir / f"{short}_{L}_{split}.csv.gz", index=False)
            print(f"{short} ctx={L} {split}: {len(p)}개 {time.time() - t0:.1f}s (batch {bs})", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    pred = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in sorted(part_dir.glob("*.csv.gz"))])
    pred = pred.merge(truth, on=["time", "space", "h"], how="left")
    assert pred.y.notna().all()
    pred.to_csv(OUT / "chronos_predictions.csv.gz", index=False)
    print("saved", len(pred))


if __name__ == "__main__":
    main()
