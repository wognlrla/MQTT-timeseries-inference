"""과제 3: 모델 입력·출력 만들기 (과제 8 서비스도 같은 함수를 쓴다).

입력 (시각 t 기준)
- 과거 2시간(24칸): CO₂·온도·습도(학습 기간 정규화), 각 항목의 결측 표시, [일정 사용 시] 일정 진행 여부·예상 인원
- 정적 정보: 시각(sin/cos), 요일(sin/cos), 주말 여부, 공간(원-핫 4)
- [일정 사용 시] 앞으로 60분(12칸)의 일정 진행 여부·예상 인원 → 미래에도 미리 알 수 있는 정보
출력
- 앞으로 5~60분(12칸)의 CO₂·온도. 기본은 '지금 값에서의 변화량'(delta)을 학습 기간 표준편차로 나눈 값
"""
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, "/srv/team/common")
import preprocess as pp  # noqa: E402

SPACES = ["classroom_a", "classroom_b", "meeting", "lounge"]
LOOKBACK = 24   # 2시간
HORIZON = 12    # 60분
TARGETS = ["co2", "temperature"]
T_IDX = [pp.VARS.index(v) for v in TARGETS]


def schedule_grid(schedule, idx, meta):
    """5분 칸별 일정 진행 비율(sched_on)과 예상 인원/정원(sched_exp). 칸 t = [t-5분, t)."""
    out = {}
    mins = pd.date_range(idx[0] - pd.Timedelta(pp.FREQ), idx[-1], freq="1min")
    for s in SPACES:
        cap = meta["spaces"][s]["capacity"]
        on = np.zeros(len(mins))
        ex = np.zeros(len(mins))
        for r in schedule[schedule.space == s].itertuples():
            m = (mins >= r.start) & (mins < r.end)
            on[m] = 1.0
            ex[m] = np.maximum(ex[m], r.expected / cap)
        df = pd.DataFrame({"sched_on": on, "sched_exp": ex}, index=mins)
        out[s] = df.resample(pp.FREQ, label="right", closed="left").mean().reindex(idx).fillna(0.0)
    return out


def static_feats(times, space):
    h = times.hour + times.minute / 60
    d = times.weekday
    onehot = np.zeros((len(times), len(SPACES)))
    onehot[:, SPACES.index(space)] = 1
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7),
                            (d >= 5).astype(float), onehot]).astype(np.float32)


def build_arrays(grid, schedule, meta, norm, use_schedule=True):
    """모든 공간·모든 가능한 시각 t에 대해 입력·정답 배열을 만든다."""
    idx = pd.DatetimeIndex(sorted(grid["time"].unique()))
    sched = schedule_grid(schedule, idx, meta) if use_schedule else None
    mean = np.array([norm[v]["mean"] for v in pp.VARS], dtype=np.float32)
    std = np.array([norm[v]["std"] for v in pp.VARS], dtype=np.float32)
    parts = []
    for s in SPACES:
        g = grid[grid.space == s].set_index("time").reindex(idx)
        raw = g[pp.VARS].to_numpy(np.float32)
        miss = np.isnan(raw)
        z = np.where(miss, 0.0, (raw - mean) / std).astype(np.float32)
        observed = ~miss & ~g[[f"{v}_filled" for v in pp.VARS]].fillna(False).to_numpy(bool)
        past = [z, miss.astype(np.float32)]
        if use_schedule:
            sc = sched[s].to_numpy(np.float32)
            past.append(sc)
        past = np.concatenate(past, axis=1)
        T = len(idx)
        origins = np.arange(LOOKBACK - 1, T - HORIZON)
        Xp = sliding_window_view(past, LOOKBACK, axis=0)[origins - LOOKBACK + 1].transpose(0, 2, 1)
        fut_rows = origins[:, None] + np.arange(1, HORIZON + 1)[None, :]
        Xf = sc[fut_rows] if use_schedule else np.zeros((len(origins), HORIZON, 0), np.float32)
        cur = raw[origins][:, T_IDX]
        fut = raw[fut_rows][:, :, T_IDX]
        y_mask = observed[fut_rows][:, :, T_IDX]
        fault = g["fault"].fillna(False).to_numpy(bool)
        fwin = sliding_window_view(np.concatenate([fault, np.zeros(HORIZON, bool)]), LOOKBACK + HORIZON)[
            origins - LOOKBACK + 1].any(axis=1)
        times = idx[origins]
        parts.append(dict(
            past=np.ascontiguousarray(Xp, dtype=np.float32), future=np.ascontiguousarray(Xf, dtype=np.float32),
            static=static_feats(times, s), cur=cur, fut=fut, y_mask=y_mask,
            time=times.values, space=np.array([s] * len(origins)),
            split=pp.split_of(pd.Series(times), meta).to_numpy(), fault_in_window=fwin))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["cur_ok"] = ~np.isnan(out["cur"]).any(axis=1)
    out["std_t"] = std[T_IDX]
    out["mean_t"] = mean[T_IDX]
    return out


def targets(arr, kind="delta"):
    """학습용 정답(정규화). delta: (미래-지금)/std, abs: (미래-평균)/std"""
    if kind == "delta":
        y = (arr["fut"] - arr["cur"][:, None, :]) / arr["std_t"]
    elif kind == "abs":
        y = (arr["fut"] - arr["mean_t"]) / arr["std_t"]
    else:
        raise ValueError(kind)
    return np.nan_to_num(y.astype(np.float32)), (arr["y_mask"] & arr["cur_ok"][:, None, None]).astype(np.float32)
