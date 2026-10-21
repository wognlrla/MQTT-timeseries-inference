"""과제 4: 이상 탐지기 (규칙 2종 + 조건부 오토인코더). 과제 8 서비스도 이 모듈을 쓴다.

규칙 (1분 원본, 범위 밖 값은 이미 제거된 상태)
- spike : 직전 5개 기록의 중앙값에서 기준 이상 벗어났다가, 바로 다음 기록에서 기준의 절반 안으로 돌아오면 경보.
          기준 = 학습 기간 정상 데이터의 최대 이탈 × 1.5 (공간·항목별). 다음 기록을 봐야 해서 기록 주기만큼 늦게 울린다.
- stuck : CO₂와 온도가 모두 같은 값으로 STUCK_MIN분 이상 이어지면 경보. 평소 같은 값은 최대 10분까지만 이어진다.

오토인코더 (5분 격자)
- 입력: 최근 2시간(24칸) CO₂·온도·습도(학습 기간 정규화) + 조건(시각·요일·공간, 같은 2시간의 일정)
- 정상(학습 기간·고장 없음)만으로 학습 → '이 시각·이 공간·이 일정에서의 평소 모습'을 복원한다.
- 점수: 최근 30분(6칸)의 항목별 복원 오차를 검증 기간 오차 분포로 나눈 값의 최댓값. 원인 = 가장 큰 항목.
"""
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, "/srv/team/common")
sys.path.insert(0, "/srv/team/task3")
import features as F  # noqa: E402
import preprocess as pp  # noqa: E402

RULE_VARS = ["co2", "temperature"]
SPIKE_FACTOR = 1.5
STUCK_MIN = 30
WIN = 24        # 2시간
SCORE_LAST = 6  # 최근 30분


# ------------------------------------------------------------------ 규칙
def calibrate_spike(raw, meta):
    a, b = pp.split_bounds(meta)["train"]
    tr = raw[(raw.time >= a) & (raw.time < b)]
    thr = {}
    for (space, ), g in tr.groupby(["space"]):
        for v in RULE_VARS:
            x = g.set_index("time")[v].dropna()
            dev = (x - x.rolling(5).median().shift(1)).abs()
            thr[(space, v)] = float(dev.max() * SPIKE_FACTOR)
    return thr


def spike_alarms(raw, thr):
    out = []
    for space, g in raw.groupby("space"):
        for v in RULE_VARS:
            x = g.set_index("time")[v].dropna()
            med = x.rolling(5).median().shift(1)
            t = thr[(space, v)]
            nxt = x.shift(-1)
            hit = ((x - med).abs() > t) & ((nxt - med).abs() < t / 2)
            # 다음 기록이 와야 확정되므로 경보 시각 = 다음 기록 시각
            when = x.index.to_series().shift(-1)[hit]
            out.append(pd.DataFrame({"time": when.values, "space": space, "detector": "rule_spike", "var": v,
                                     "score": ((x - med).abs() / t)[hit].values}))
    return pd.concat(out, ignore_index=True).dropna(subset=["time"])


def stuck_alarms(raw, minutes=STUCK_MIN):
    out = []
    for space, g in raw.groupby("space"):
        x = g.set_index("time")[RULE_VARS].dropna()
        same = (x.diff() == 0).all(axis=1)
        run_id = (~same).cumsum()
        start = x.index.to_series().groupby(run_id.values).transform("first")
        dur = (x.index.to_series() - start).dt.total_seconds() / 60
        hit = dur >= minutes
        out.append(pd.DataFrame({"time": x.index[hit], "space": space, "detector": "rule_stuck", "var": "all",
                                 "score": (dur[hit] / minutes).values}))
    return pd.concat(out, ignore_index=True)


LEVEL_FACTOR = 1.1     # 평소 최고치(학습 99.9%) × 1.1
NIGHT_HOURS = range(0, 5)
NIGHT_MARGIN = 100     # 학습 기간 새벽 최고치 + 100ppm
NIGHT_MIN_CELLS = 6    # 30분 이상 유지


def calibrate_level(grid, meta):
    """CO₂ 5분 값의 공간별 '평소 최고치'와 '새벽 최고치' (학습 기간, 고장 없음)."""
    a, b = pp.split_bounds(meta)["train"]
    tr = grid[(grid.time >= a) & (grid.time < b) & ~grid.fault]
    level = (tr.groupby("space").co2.quantile(0.999) * LEVEL_FACTOR).to_dict()
    night = (tr[tr.time.dt.hour.isin(NIGHT_HOURS)].groupby("space").co2.max() + NIGHT_MARGIN).to_dict()
    return {"level": level, "night": night}


def level_alarms(grid, cal):
    """환기 정지 대응: 5분 CO₂가 그 공간의 평소 최고치를 넘음."""
    g = grid[grid.co2 > grid.space.map(cal["level"])]
    return pd.DataFrame({"time": g.time.values, "space": g.space.values, "detector": "rule_level", "var": "co2",
                         "score": (g.co2 / g.space.map(cal["level"])).values})


def night_alarms(grid, cal):
    """영점 틀어짐 대응: 사람이 없는 새벽 CO₂는 외기 근처여야 한다. 30분 넘게 새벽 최고치+100ppm 위면 경보."""
    out = []
    for space, g in grid.sort_values("time").groupby("space"):
        hi = (g.co2 > cal["night"][space]) & g.time.dt.hour.isin(NIGHT_HOURS)
        run = hi.groupby((~hi).cumsum()).cumsum()
        h = g[run >= NIGHT_MIN_CELLS]
        out.append(pd.DataFrame({"time": h.time.values, "space": space, "detector": "rule_night", "var": "co2",
                                 "score": (h.co2 / cal["night"][space]).values}))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ 오토인코더
class CondAE(nn.Module):
    def __init__(self, n_x, n_c, hidden=128, bottleneck=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_x + n_c, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(),
                                 nn.Linear(32, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck + n_c, 32), nn.ReLU(), nn.Linear(32, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_x))

    def forward(self, x, c):
        return self.dec(torch.cat([self.enc(torch.cat([x, c], 1)), c], 1))


def ae_windows(grid, schedule, meta, norm):
    """공간별 5분 격자에서 2시간 창(끝 시각 t)마다 입력 x, 조건 c, 사용 가능 여부를 만든다."""
    idx = pd.DatetimeIndex(sorted(grid["time"].unique()))
    sched = F.schedule_grid(schedule, idx, meta)
    mean = np.array([norm[v]["mean"] for v in pp.VARS], np.float32)
    std = np.array([norm[v]["std"] for v in pp.VARS], np.float32)
    parts = []
    for s in F.SPACES:
        g = grid[grid.space == s].set_index("time").reindex(idx)
        z = ((g[pp.VARS].to_numpy(np.float32) - mean) / std)
        ok_cell = ~np.isnan(z).any(axis=1)
        z = np.nan_to_num(z)
        sc = sched[s].to_numpy(np.float32)
        ends = np.arange(WIN - 1, len(idx))
        rows = ends[:, None] + np.arange(-WIN + 1, 1)[None, :]
        x = z[rows].reshape(len(ends), -1)
        c = np.concatenate([F.static_feats(idx[ends], s), sc[rows].reshape(len(ends), -1)], axis=1)
        fault = g["fault"].fillna(False).to_numpy(bool)
        parts.append(dict(x=x.astype(np.float32), c=c.astype(np.float32), ok=ok_cell[rows].all(axis=1),
                          fault=fault[rows].any(axis=1), time=idx[ends].values, space=np.array([s] * len(ends)),
                          split=pp.split_of(pd.Series(idx[ends]), meta).to_numpy()))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def train_ae(w, seed=0, epochs=80, patience=8, dev="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = np.where((w["split"] == "train") & w["ok"] & ~w["fault"])[0]
    va = np.where((w["split"] == "val") & w["ok"] & ~w["fault"])[0]
    m = CondAE(w["x"].shape[1], w["c"].shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    X, C = torch.from_numpy(w["x"]).to(dev), torch.from_numpy(w["c"]).to(dev)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 256):
            b = torch.from_numpy(perm[i:i + 256]).to(dev)
            loss = ((m(X[b], C[b]) - X[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vb = torch.from_numpy(va).to(dev)
            vl = ((m(X[vb], C[vb]) - X[vb]) ** 2).mean().item()
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return m, {"val_mse": best, "epochs": ep + 1, "train_n": len(tr)}


def ae_errors(m, w, dev="cuda"):
    """창 끝 30분의 항목별 평균 제곱 오차 [N, 3]."""
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(w["x"]), 8192):
            x = torch.from_numpy(w["x"][i:i + 8192]).to(dev)
            c = torch.from_numpy(w["c"][i:i + 8192]).to(dev)
            e = ((m(x, c) - x) ** 2).view(len(x), WIN, len(pp.VARS))[:, -SCORE_LAST:, :].mean(1)
            out.append(e.cpu().numpy())
    return np.concatenate(out)


def ae_scale(err, w):
    """검증 기간(고장 없음) 오차의 99.9% 값으로 항목별 크기를 맞춘다 → 점수 1 ≈ 평소의 최상단."""
    va = (w["split"] == "val") & w["ok"]
    return np.quantile(err[va], 0.999, axis=0)
