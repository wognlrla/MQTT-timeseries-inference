"""공통 전처리 (과제 2 정제 규칙).

학습·평가(과제 2~7)와 상시 서비스(과제 8)가 모두 이 모듈을 쓴다.
학습과 서비스의 전처리가 다르면 모델이 학습 때와 다른 분포를 보게 되므로, 규칙은 여기 한 곳에만 둔다.

정제 규칙
1. 범위 밖 값 제거: RANGES 밖의 값은 결측으로 바꾼다(0ppm, 65535ppm, -40℃, 60℃ 같은 깨진 값).
2. 중복 제거: 같은 공간·같은 시각의 기록은 첫 값만 남긴다(재전송).
3. 5분 간격 정렬: [t-5분, t) 구간의 평균을 시각 t의 값으로 쓴다.
   t 시점에 이미 도착한 값만 쓰므로 상시 서비스에서도 똑같이 재현된다.
   센서마다 다른 기록 주기(1·2·5분)가 이 단계에서 하나로 맞춰진다.
4. 짧은 결측(15분 = 3칸 이하): 직전 값으로 채운다(forward fill).
   선형 보간은 '다음 값'을 써야 해서 서비스 시점에는 쓸 수 없으므로 쓰지 않는다.
5. 긴 결측(15분 초과): 채우지 않고 결측으로 둔다. 학습·평가에서 해당 시점을 제외한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RANGES = {"co2": (300.0, 5000.0), "temperature": (5.0, 45.0), "humidity": (1.0, 100.0)}
VARS = list(RANGES)
FREQ = "5min"
STEP_MIN = 5
FILL_LIMIT = 3  # 5분 × 3칸 = 15분


# ------------------------------------------------------------------ 불러오기
def load_synthetic(root="/srv/team/data/synthetic"):
    """합성 데이터 묶음을 읽는다. 시각은 KST(naive)."""
    root = Path(root)
    sensors = pd.read_csv(root / "sensors.csv", parse_dates=["time"])
    schedule = pd.read_csv(root / "schedule.csv", parse_dates=["start", "end"])
    faults = pd.read_csv(root / "faults.csv", parse_dates=["start", "end"])
    meta = json.loads((root / "meta.json").read_text())
    return sensors, schedule, faults, meta


def split_bounds(meta):
    return {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in meta["split"].items()}


def split_of(times, meta):
    """각 시각이 train / val / test 중 어디에 속하는지. 시험 기간은 끝 시각을 포함한다."""
    times = pd.Series(times)
    out = pd.Series("none", index=times.index, dtype=object)
    for name, (a, b) in split_bounds(meta).items():
        out[(times >= a) & ((times <= b) if name == "test" else (times < b))] = name
    return out


# ------------------------------------------------------------------ 1·2단계: 원본 정리
def clean_raw(df):
    """범위 밖 값을 결측으로 바꾸고 중복을 지운다. (정리된 원본, 항목별 제거 건수) 반환."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["space", "time"], keep="first")
    stats = {"duplicates": before - len(df), "out_of_range": {}}
    for v, (lo, hi) in RANGES.items():
        bad = df[v].notna() & ((df[v] < lo) | (df[v] > hi))
        stats["out_of_range"][v] = int(bad.sum())
        df.loc[bad, v] = np.nan
    return df.sort_values(["space", "time"]).reset_index(drop=True), stats


# ------------------------------------------------------------------ 3~5단계: 5분 격자
def to_grid(raw, start, end):
    """정리된 원본을 공간별 5분 격자로 만든다.

    반환 열: time, space, co2, temperature, humidity, n_obs(칸 안 원본 수),
            <var>_filled(직전 값으로 채움), <var>_gap(긴 결측이라 비어 있음)
    """
    start, end = pd.Timestamp(start).ceil(FREQ), pd.Timestamp(end).floor(FREQ) + pd.Timedelta(FREQ)
    idx = pd.date_range(start, end, freq=FREQ)
    frames = []
    for space, g in raw.groupby("space"):
        g = g.set_index("time")
        r = g[VARS].resample(FREQ, label="right", closed="left").mean()
        n = g[VARS[0]].resample(FREQ, label="right", closed="left").size()
        r = r.reindex(idx)
        r["n_obs"] = n.reindex(idx).fillna(0).astype(int)
        for v in VARS:
            filled = r[v].ffill(limit=FILL_LIMIT)
            r[f"{v}_filled"] = r[v].isna() & filled.notna()
            r[v] = filled
            r[f"{v}_gap"] = r[v].isna()
        r.insert(0, "space", space)
        frames.append(r.rename_axis("time").reset_index())
    return pd.concat(frames, ignore_index=True)


def fault_mask(grid, faults, pad_min=0):
    """격자 각 칸이 고장 구간과 겹치는지(fault)와 고장 종류(fault_type)를 붙인다."""
    grid = grid.copy()
    grid["fault"] = False
    grid["fault_type"] = ""
    pad = pd.Timedelta(minutes=pad_min)
    for f in faults.itertuples():
        # 칸 t는 [t-5분, t) 구간 → 고장 [start, end]와 겹치면 표시
        m = ((grid["space"] == f.space) & (grid["time"] > f.start - pad)
             & (grid["time"] - pd.Timedelta(FREQ) <= f.end + pad))
        grid.loc[m, "fault"] = True
        grid.loc[m, "fault_type"] = f.type
    return grid


# ------------------------------------------------------------------ 정규화 (학습 기간만)
def fit_norm(grid, meta):
    """학습 기간·고장 아님·결측 아님인 값으로만 항목별 평균·표준편차를 구한다."""
    a, b = split_bounds(meta)["train"]
    tr = grid[(grid["time"] >= a) & (grid["time"] < b) & ~grid.get("fault", False)]
    return {v: {"mean": float(tr[v].mean()), "std": float(tr[v].std())} for v in VARS}


def apply_norm(df, stats):
    df = df.copy()
    for v in VARS:
        df[v] = (df[v] - stats[v]["mean"]) / stats[v]["std"]
    return df


def build(root="/srv/team/data/synthetic"):
    """합성 데이터 → (5분 격자 + 고장 표시, 원본 정리 통계, 일정, 고장, 메타)."""
    sensors, schedule, faults, meta = load_synthetic(root)
    raw, stats = clean_raw(sensors)
    grid = to_grid(raw, meta["start"], meta["end"])
    grid = fault_mask(grid, faults)
    grid["split"] = split_of(grid["time"], meta).values
    return grid, raw, stats, schedule, faults, meta
