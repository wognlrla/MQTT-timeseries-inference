"""전기수(2025-12 ~ 2026-09) 실데이터 60분 예측 — 분기별 비교.

분기: 1분기 2025-12 ~ 2026-03 / 2분기 2026-04 ~ 06 / 3분기 2026-07 ~ 09 (KST)
공간: 실습실·사무실·회의실 (운영진 DB에서 이관한 과거 원본 Parquet)

전처리: 과제 2와 같은 규칙(범위 밖 제거 → 5분 격자 → 15분 이하 직전 값 채움). 같은 공간의 여러 센서는 평균.
       원본이 커서(월 수백만 행) 센서 기록을 먼저 1분 평균으로 줄인 뒤 5분 격자로 만든다.
모델: 과제 3 GRU 구조, 실데이터에는 수업 시간표가 없어 **일정 입력 없이** 과거 2시간 + 시각·요일 + 공간.
실험
  A. 분기별: 분기 안에서 시간순 70% 학습 / 10% 검증 / 마지막 20% 시험
  B. 다음 분기 예측: 1분기로 학습 → 2분기 전체 시험, 1·2분기로 학습 → 3분기 전체 시험 (학습 기간의 마지막 10%는 검증)
평가: 시험 구간에서 현재값·정답·기준선 입력이 모두 있는 5분 시점. 기준선 3종과 같은 시점에서 비교. 시드 2개 평균.

사용: /srv/team/venv/bin/python cohort.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

sys.path[:0] = ["/srv/team/common", "/srv/team/task3"]
import preprocess as pp  # noqa: E402
from train import GRUForecaster  # noqa: E402

OUT = Path("/srv/team/cohort/out")
SRC = Path("/srv/team/data/raw/influx_history")
SPACES = ["실습실", "사무실", "회의실"]
COHORTS = {"1분기": ("2025-12-01", "2026-04-01"), "2분기": ("2026-04-01", "2026-07-01"), "3분기": ("2026-07-01", "2026-10-01")}
MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
L, H = 24, 12
HSTEP = {5: 0, 15: 2, 30: 5, 60: 11}
T_IDX = [0, 1]  # co2, temperature
SEEDS = [0, 1]
# 보드 메모리 보호: CUDA 런타임(약 1GB+)을 띄우지 않고 CPU로 학습한다(모델이 작아 충분히 빠름)
DEV = "cpu"
torch.set_num_threads(4)
plt.rcParams["font.family"] = "Noto Sans CJK JP"


# ------------------------------------------------------------------ 데이터
def load_raw():
    parts = []
    for m in MONTHS:
        f = SRC / f"history_{m}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["time", "metric", "value", "location", "device_id", "point"],
                            filters=[("metric", "in", pp.VARS), ("location", "in", SPACES)])
        d["metric"] = d["metric"].astype(str)
        d["location"] = d["location"].astype(str)
        d = d.drop_duplicates(["time", "metric", "location", "device_id", "point"])  # 같은 센서·같은 시각 재전송 제거
        for v, (lo, hi) in pp.RANGES.items():                                        # 범위 밖 값 제거
            bad = (d.metric == v) & ((d.value < lo) | (d.value > hi))
            d = d[~bad]
        d["time"] = d["time"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.floor("1min")
        a = d.groupby(["location", "metric", "time"]).value.mean().unstack("metric").reset_index()
        parts.append(a)
        print(f"{m}: 원본 {len(d):,}행 → 1분 {len(a):,}행", flush=True)
    raw = pd.concat(parts, ignore_index=True).rename(columns={"location": "space"})
    for v in pp.VARS:
        if v not in raw:
            raw[v] = np.nan
    return raw[["time", "space"] + pp.VARS]


def static_feats(times, space):
    h = times.hour + times.minute / 60
    d = times.weekday
    oh = np.zeros((len(times), len(SPACES)))
    oh[:, SPACES.index(space)] = 1
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * d / 7),
                            np.cos(2 * np.pi * d / 7), (d >= 5).astype(float), oh]).astype(np.float32)


def build(grid, norm):
    """공간별 시점마다 입력(과거 2시간)·정답(변화량)·기준선."""
    idx = pd.DatetimeIndex(sorted(grid.time.unique()))
    mean = np.array([norm[v][0] for v in pp.VARS], np.float32)
    std = np.array([norm[v][1] for v in pp.VARS], np.float32)
    parts = []
    for s in SPACES:
        g = grid[grid.space == s].set_index("time").reindex(idx)
        raw = g[pp.VARS].to_numpy(np.float32)
        miss = np.isnan(raw)
        z = np.where(miss, 0.0, (raw - mean) / std).astype(np.float32)
        obs = ~miss & ~g[[f"{v}_filled" for v in pp.VARS]].fillna(False).to_numpy(bool)
        past_all = np.concatenate([z, miss.astype(np.float32)], axis=1)
        T = len(idx)
        o = np.arange(max(L - 1, 288 - 1), T - H)                 # 어제 기준선을 위해 하루 이후부터
        rows = o[:, None] + np.arange(1, H + 1)[None, :]
        cur = raw[o][:, T_IDX]
        fut = raw[rows][:, :, T_IDX]
        fut_obs = obs[rows][:, :, T_IDX]
        x = pd.DataFrame(raw[:, T_IDX])
        mean1h = x.rolling(12, min_periods=6).mean().to_numpy(np.float32)[o]
        yday = raw[rows - 288][:, :, T_IDX]
        parts.append(dict(past=np.ascontiguousarray(sliding_window_view(past_all, L, axis=0)[o - L + 1].transpose(0, 2, 1)),
                          static=static_feats(idx[o], s), cur=cur, fut=fut, fut_obs=fut_obs, mean1h=mean1h, yday=yday,
                          time=idx[o].values, space=np.array([s] * len(o))))
    a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    a["std_t"], a["cur_ok"] = std[T_IDX], ~np.isnan(a["cur"]).any(1)
    return a


def norm_from(grid, a, b):
    g = grid[(grid.time >= a) & (grid.time < b)]
    return {v: (float(g[v].mean()), float(g[v].std())) for v in pp.VARS}


# ------------------------------------------------------------------ 학습·평가
def train(a, tr, va, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    y = np.nan_to_num((a["fut"] - a["cur"][:, None, :]) / a["std_t"]).astype(np.float32)
    m = (a["fut_obs"] & a["cur_ok"][:, None, None]).astype(np.float32)
    model = GRUForecaster(a["past"].shape[2], a["static"].shape[1], 0).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = lambda k, i: torch.from_numpy(a[k][i]).to(DEV)  # noqa: E731
    fz = lambda n: torch.zeros(n, H, 0, device=DEV)      # noqa: E731
    yv, mv = torch.from_numpy(y[va]).to(DEV), torch.from_numpy(m[va]).to(DEV)
    best, state, bad = 1e9, None, 0
    for ep in range(40):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            p = model(T("past", b), T("static", b), fz(len(b)))
            mb = torch.from_numpy(m[b]).to(DEV)
            loss = (torch.abs(p - torch.from_numpy(y[b]).to(DEV)) * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(((torch.abs(model(T("past", va[i:i + 8192]), T("static", va[i:i + 8192]), fz(len(va[i:i + 8192])))
                                 - yv[i:i + 8192]) * mv[i:i + 8192]).sum().item()) for i in range(0, len(va), 8192)) / mv.sum().item()
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(state)
    return model, ep + 1


def predict(model, a, sel):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sel), 8192):
            b = sel[i:i + 8192]
            z = model(torch.from_numpy(a["past"][b]).to(DEV), torch.from_numpy(a["static"][b]).to(DEV),
                      torch.zeros(len(b), H, 0, device=DEV)).cpu().numpy()
            out.append(a["cur"][b][:, None, :] + z * a["std_t"])
    return np.concatenate(out)


def eval_points(a, mask):
    ok = mask & a["cur_ok"] & ~np.isnan(a["mean1h"]).any(1)
    for k in HSTEP.values():
        ok &= a["fut_obs"][:, k, :].all(1) & ~np.isnan(a["yday"][:, k, :]).any(1)
    return np.where(ok)[0]


# ------------------------------------------------------------------ 분기별 사용 패턴
def profiles(grid):
    rows, prof = [], {}
    for c, (a, b) in COHORTS.items():
        g = grid[(grid.time >= a) & (grid.time < b)].copy()
        g["wd"] = g.time.dt.weekday < 5
        g["hour"] = g.time.dt.hour + g.time.dt.minute / 60
        for s in SPACES:
            x = g[g.space == s]
            day = x[x.wd & x.hour.between(9, 18)]
            daily_max = x[x.wd].set_index("time").co2.resample("D").max().dropna()
            rows.append({"분기": c, "공간": s, "5분칸 관측률(%)": round(100 * x.co2.notna().mean(), 1),
                         "평일 9~18시 평균 CO2": round(day.co2.mean(), 0),
                         "평일 9~18시 1,000ppm 초과(%)": round(100 * (day.co2 > 1000).mean(), 1),
                         "평일 하루 최고 CO2(중앙값)": round(daily_max.median(), 0),
                         "평일 9~18시 평균 온도": round(day.temperature.mean(), 1)})
            prof[(c, s)] = x[x.wd].groupby(x.hour).co2.mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for i, s in enumerate(["실습실", "사무실"]):
        for c in COHORTS:
            p = prof[(c, s)]
            ax[i].plot(p.index, p.values, label=c)
        ax[i].axhline(1000, color="red", lw=0.7)
        ax[i].set_title(f"{s} — 평일 시간대별 평균 CO2")
        ax[i].set_xlabel("시각")
        ax[i].set_xticks(range(0, 25, 3))
    ax[0].set_ylabel("CO2 (ppm)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_cohort_profile.png", dpi=130)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    grid = pp.to_grid(raw, "2025-11-01", "2026-09-30 23:55")
    del raw
    prof = profiles(grid)
    prof.to_csv(OUT / "cohort_profile.csv", index=False)
    print(prof.to_string(index=False), flush=True)

    allrows, meta = [], {}
    for c, (a, b) in COHORTS.items():           # A. 분기별 (70/10/20)
        A, B = pd.Timestamp(a), pd.Timestamp(b)
        tr_end, te_start = A + (B - A) * 0.7, A + (B - A) * 0.8
        name = f"A {c} 자체"
        rows, info = experiment_inner(name, grid, A, tr_end, te_start, te_start, B)
        allrows += rows
        meta[name] = info
    for name, (a, b), (sa, sb) in [("B 1분기→2분기", COHORTS["1분기"], COHORTS["2분기"]),
                                   ("B 1·2분기→3분기", (COHORTS["1분기"][0], COHORTS["2분기"][1]), COHORTS["3분기"])]:
        A, B = pd.Timestamp(a), pd.Timestamp(b)
        rows, info = experiment_inner(name, grid, A, A + (B - A) * 0.9, B, pd.Timestamp(sa), pd.Timestamp(sb))
        allrows += rows
        meta[name] = info
    res = pd.DataFrame(allrows)
    res.to_csv(OUT / "cohort_results.csv", index=False)
    (OUT / "cohort_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    pd.set_option("display.width", 250)
    t = res[res.공간 == "전체"].pivot_table(index=["실험", "방법"], columns=["var", "h"], values="MAE").round(2)
    print(t.to_string())
    t2 = res[(res.h == 60) & (res["var"] == "co2")].pivot_table(index=["실험", "방법"], columns="공간", values="MAE").round(1)
    print(t2.to_string())


def experiment_inner(name, grid, ta, tr_end, tb, sa, sb):
    """학습 [ta, tr_end), 검증 [tr_end, tb), 시험 [sa, sb)."""
    norm = norm_from(grid, ta, tr_end)
    a = build(grid, norm)
    t = pd.to_datetime(a["time"])
    tr = np.where((t >= ta) & (t < tr_end) & a["cur_ok"])[0]
    va = np.where((t >= tr_end) & (t < tb) & a["cur_ok"])[0]
    te = eval_points(a, (t >= sa) & (t < sb))
    preds = []
    for seed in SEEDS:
        t0 = time.time()
        model, ep = train(a, tr, va, seed)
        preds.append(predict(model, a, te))
        print(f"  {name} seed{seed}: 학습 {len(tr):,} / 검증 {len(va):,} / 시험 {len(te):,}, {ep}에폭, {time.time() - t0:.0f}초", flush=True)
    errs = {"현재값 유지": np.abs(a["cur"][te][:, None, :] - a["fut"][te]),
            "최근 1시간 평균": np.abs(a["mean1h"][te][:, None, :] - a["fut"][te]),
            "어제 같은 시각": np.abs(a["yday"][te] - a["fut"][te]),
            "GRU (실데이터 학습)": np.mean([np.abs(p - a["fut"][te]) for p in preds], axis=0)}
    sp = a["space"][te]
    rows = []
    for meth, err in errs.items():
        for h, k in HSTEP.items():
            for j, v in enumerate(["co2", "temperature"]):
                rows.append({"실험": name, "방법": meth, "h": h, "var": v, "공간": "전체", "MAE": float(err[:, k, j].mean()), "n": len(te)})
                for s in SPACES:
                    msk = sp == s
                    if msk.any():
                        rows.append({"실험": name, "방법": meth, "h": h, "var": v, "공간": s, "MAE": float(err[msk, k, j].mean()), "n": int(msk.sum())})
    return rows, {"train": [str(ta), str(tr_end)], "val": [str(tr_end), str(tb)], "test": [str(sa), str(sb)],
                  "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                  "n_test_by_space": {s: int((sp == s).sum()) for s in SPACES}}


if __name__ == "__main__":
    main()
