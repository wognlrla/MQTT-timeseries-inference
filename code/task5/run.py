"""과제 5: 상태 추정 — 사용 여부 추정 + 환기 선제 경보.

[사용 여부]
- 학습 정답: 일정(시간표·예약·라운지 혼잡 예상)이 진행 중인 5분 칸 = 사용중.
- 평가 정답: '직접 관찰' 기록. 시험 주간에 공간마다 하루 3번, 1시간씩 무작위로 관찰한 것으로 가정하고
  합성 데이터의 실제 인원(truth)으로 기록을 만든다(관찰 84시간 = 1,008칸).
- 입력: 센서 변화만(CO₂ 수준·변화량, 온도·습도 변화량, 공간). 일정과 시각은 넣지 않는다.
- 규칙: 최근 15분 CO₂가 30ppm 이상 올랐으면 사용중.

[환기 선제 경보]
- 정답(양성): 지금 CO₂ < 1,000ppm 이고, 앞으로 30분(6칸) 안에 1,000ppm 이상이 된다.
  이미 1,000ppm 이상인 시점은 '선제' 경보 대상이 아니므로 데이터에서 뺀다.
- 입력: 센서 수준·변화량 + 시각·요일 + 공간 + 지금과 앞으로 30분의 일정.
- 결정 기준: 검증 기간에서 재현율 88% 이상을 만족하는 가장 높은 확률 기준(시험 기간은 쓰지 않음).
- 규칙: 지금 CO₂ ≥ 850ppm 이면 경보. 과제 3 예측 모델: 예측한 5·15·30분 뒤 CO₂ 중 하나라도 1,000 이상이면 경보.

공통: 고장 구간(앞 60분~뒤 30분)은 학습·평가에서 뺀다. 결과는 out/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, "/srv/team/common")
sys.path.insert(0, "/srv/team/task3")
import features as F  # noqa: E402
import preprocess as pp  # noqa: E402

OUT = Path("/srv/team/task5/out")
NAMES = {"classroom_a": "강의실 A", "classroom_b": "강의실 B", "meeting": "회의실", "lounge": "라운지"}
CO2_OUT = 420.0
LIMIT = 1000.0
OBS_PER_DAY, OBS_HOURS, OBS_SEED = 3, (7, 20), 7


# ------------------------------------------------------------------ 데이터
def build_table():
    grid, raw, _, schedule, faults, meta = pp.build()
    idx = pd.DatetimeIndex(sorted(grid.time.unique()))
    sched = F.schedule_grid(schedule, idx, meta)
    truth = pd.read_csv("/srv/team/data/synthetic/truth.csv.gz", parse_dates=["time"])
    rows = []
    for s in F.SPACES:
        g = grid[grid.space == s].set_index("time").reindex(idx)
        t = truth[truth.space == s].set_index("time")[["occupied", "occupants"]]
        t5 = t.resample(pp.FREQ, label="right", closed="left").mean().reindex(idx)
        d = pd.DataFrame(index=idx)
        d["space"] = s
        d["co2"], d["temp"], d["hum"] = g.co2, g.temperature, g.humidity
        for k, n in [("5", 1), ("15", 3), ("30", 6)]:
            d[f"dco2_{k}"] = g.co2.diff(n)
            d[f"dtemp_{k}"] = g.temperature.diff(n)
            d[f"dhum_{k}"] = g.humidity.diff(n)
        d["co2_rng30"] = g.co2.rolling(7).max() - g.co2.rolling(7).min()
        d["co2_max60"] = g.co2.rolling(13).max()
        d["sched_on"] = sched[s].sched_on.values
        d["sched_exp"] = sched[s].sched_exp.values
        fut = pd.concat([sched[s].sched_on.shift(-k) for k in range(1, 7)], axis=1)
        d["sched_on_next30"] = fut.max(axis=1).values
        d["sched_exp_next30"] = pd.concat([sched[s].sched_exp.shift(-k) for k in range(1, 7)], axis=1).max(axis=1).values
        d["occ_true"] = (t5.occupied >= 0.5).astype(float).where(t5.occupied.notna())
        d["occupants"] = t5.occupants
        d["occ_sched"] = (d.sched_on >= 0.5).astype(float)
        fut_co2 = pd.concat([g.co2.shift(-k) for k in range(1, 7)], axis=1)
        d["fut_max"] = fut_co2.max(axis=1)
        d["fut_n"] = fut_co2.notna().sum(axis=1)
        flt = g.fault.fillna(False).astype(int)
        d["fault_near"] = (flt.rolling(13, min_periods=1).max().astype(bool)
                           | flt[::-1].rolling(7, min_periods=1).max()[::-1].astype(bool)).values
        rows.append(d.rename_axis("time").reset_index())
    df = pd.concat(rows, ignore_index=True)
    df["split"] = pp.split_of(df.time, meta).values
    h = df.time.dt.hour + df.time.dt.minute / 60
    df["h_sin"], df["h_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * df.time.dt.weekday / 7), np.cos(2 * np.pi * df.time.dt.weekday / 7)
    df["weekend"] = (df.time.dt.weekday >= 5).astype(float)
    for s in F.SPACES:
        df[f"sp_{s}"] = (df.space == s).astype(float)
    df["co2_ex"] = df.co2 - CO2_OUT
    return df, schedule, meta


def observation_slots(df, meta):
    """시험 주간 '직접 관찰' 시간대: 공간·날짜마다 1시간짜리 3번 (07~20시 중 무작위, 겹치지 않게)."""
    rng = np.random.default_rng(OBS_SEED)
    a, b = pp.split_bounds(meta)["test"]
    days = pd.date_range(a.normalize(), b.normalize(), freq="D")
    obs = pd.Series(False, index=df.index)
    slots = []
    for s in F.SPACES:
        for day in days:
            starts = rng.choice(np.arange(OBS_HOURS[0], OBS_HOURS[1]), OBS_PER_DAY, replace=False)
            for h in starts:
                st = day + pd.Timedelta(hours=int(h))
                m = (df.space == s) & (df.time > st) & (df.time <= st + pd.Timedelta(hours=1))
                obs |= m
                slots.append({"space": s, "start": st})
    return obs, pd.DataFrame(slots)


# ------------------------------------------------------------------ 모델
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(X_tr, y_tr, X_va, y_va, pos_weight=1.0, seed=0, epochs=60, patience=6):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = MLP(X_tr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
    Xv, yv = torch.tensor(X_va), torch.tensor(y_va)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = lossf(m(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = lossf(m(Xv), yv).item()
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return m


def proba(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def standardize(df, cols, ref):
    mu, sd = ref[cols].mean(), ref[cols].std().replace(0, 1)
    return ((df[cols] - mu) / sd).fillna(0).to_numpy(np.float32), {"mean": mu.to_dict(), "std": sd.to_dict()}


def metrics(y, p, pos="1", neg="0"):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tp, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    fp, fn = int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())

    def f1(tp_, fp_, fn_):
        return 2 * tp_ / max(2 * tp_ + fp_ + fn_, 1)
    f_pos, f_neg = f1(tp, fp, fn), f1(tn, fn, fp)
    return {"n": len(y), "정확도": round((tp + tn) / max(len(y), 1), 3),
            "정밀도": round(tp / max(tp + fp, 1), 3), "재현율": round(tp / max(tp + fn, 1), 3),
            "F1(양성)": round(f_pos, 3), "macro-F1": round((f_pos + f_neg) / 2, 3),
            "빈방→사용중 오류율": round(fp / max(fp + tn, 1), 3), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


# ------------------------------------------------------------------ 실행
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, schedule, meta = build_table()
    sp_cols = [f"sp_{s}" for s in F.SPACES]
    ok = df.co2.notna() & df.dco2_30.notna() & ~df.fault_near
    res = {}

    # ===== 1. 사용 여부 =====
    occ_cols = ["co2_ex", "dco2_5", "dco2_15", "dco2_30", "co2_rng30", "co2_max60", "dtemp_15", "dtemp_30",
                "dhum_15", "dhum_30"] + sp_cols
    tr = df[ok & (df.split == "train")]
    va = df[ok & (df.split == "val")]
    Xtr, occ_norm = standardize(tr, occ_cols, tr)
    Xva, _ = standardize(va, occ_cols, tr)
    m_occ = fit(Xtr, tr.occ_sched.to_numpy(np.float32), Xva, va.occ_sched.to_numpy(np.float32))
    obs, slots = observation_slots(df, meta)
    te = df[ok & obs].copy()
    Xte, _ = standardize(te, occ_cols, tr)
    te["p_model"] = proba(m_occ, Xte)
    te["pred_model"] = (te.p_model >= 0.5).astype(int)
    te["pred_rule"] = (te.dco2_15 >= 30).astype(int)
    te["pred_sched"] = te.occ_sched.astype(int)
    occ_tab = pd.DataFrame({k: metrics(te.occ_true, te[c]) for k, c in
                            [("일정 그대로(참고)", "pred_sched"), ("규칙: 15분 +30ppm", "pred_rule"),
                             ("모델 (센서만)", "pred_model")]}).T
    # 일정 기반 정답 vs 관찰
    agree = pd.crosstab(te.occ_sched.map({1.0: "일정: 사용", 0.0: "일정: 없음"}),
                        te.occ_true.map({1.0: "관찰: 사용중", 0.0: "관찰: 비어있음"}))
    # 전체 시험 기간(정답 전체)으로도 확인
    te_all = df[ok & (df.split == "test")].copy()
    te_all["pred_model"] = (proba(m_occ, standardize(te_all, occ_cols, tr)[0]) >= 0.5).astype(int)
    te_all["pred_rule"] = (te_all.dco2_15 >= 30).astype(int)
    occ_all = pd.DataFrame({"규칙": metrics(te_all.occ_true, te_all.pred_rule),
                            "모델": metrics(te_all.occ_true, te_all.pred_model)}).T
    # 오류 사례: 상태가 바뀐 뒤 몇 분 지났나
    te_all = te_all.sort_values(["space", "time"])
    chg = te_all.groupby("space").occ_true.diff().fillna(0) != 0
    last = te_all.time.where(chg).groupby(te_all.space).ffill()
    te_all["since_min"] = (te_all.time - last).dt.total_seconds() / 60
    te_all["상황"] = np.select(
        [(te_all.occ_true == 0) & (te_all.since_min <= 30), (te_all.occ_true == 1) & (te_all.since_min <= 15),
         (te_all.occ_true == 1), (te_all.occ_true == 0)],
        ["나간 뒤 30분 이내(빈 방)", "들어온 뒤 15분 이내", "사용 중(그 외)", "빈 방(그 외)"], "?")
    case = te_all.groupby("상황").apply(lambda g: pd.Series({
        "칸 수": len(g), "모델 오류율": round((g.pred_model != g.occ_true).mean(), 3),
        "규칙 오류율": round((g.pred_rule != g.occ_true).mean(), 3)}))
    small = te_all[(te_all.occ_true == 1) & te_all.space.isin(["classroom_a", "classroom_b"])].copy()
    small["인원"] = pd.cut(small.occupants, [0, 5, 10, 20, 100], labels=["1~5명", "6~10명", "11~20명", "21명+"])
    size_tab = small.groupby("인원", observed=True).apply(lambda g: pd.Series({
        "칸 수": len(g), "모델 재현율": round(g.pred_model.mean(), 3), "규칙 재현율": round(g.pred_rule.mean(), 3)}))
    by_space_occ = te_all.groupby("space").apply(lambda g: pd.Series(metrics(g.occ_true, g.pred_model))[
        ["정확도", "F1(양성)", "빈방→사용중 오류율"]])
    by_space_occ.index = by_space_occ.index.map(NAMES)

    # ===== 2. 환기 선제 경보 =====
    vok = ok & (df.co2 < LIMIT) & (df.fut_n >= 4)
    df["vent_y"] = (df.fut_max >= LIMIT).astype(float)
    vent_cols = ["co2", "dco2_5", "dco2_15", "dco2_30", "co2_rng30", "co2_max60", "temp", "dtemp_15", "dhum_15",
                 "h_sin", "h_cos", "dow_sin", "dow_cos", "weekend", "sched_on", "sched_exp", "sched_on_next30",
                 "sched_exp_next30"] + sp_cols
    vtr, vva, vte = (df[vok & (df.split == s)].copy() for s in ["train", "val", "test"])
    Xtr, vent_norm = standardize(vtr, vent_cols, vtr)
    Xva, _ = standardize(vva, vent_cols, vtr)
    Xte, _ = standardize(vte, vent_cols, vtr)
    pw = float((1 - vtr.vent_y.mean()) / vtr.vent_y.mean())
    m_vent = fit(Xtr, vtr.vent_y.to_numpy(np.float32), Xva, vva.vent_y.to_numpy(np.float32), pos_weight=pw)
    pva, pte = proba(m_vent, Xva), proba(m_vent, Xte)
    cand = np.linspace(0.01, 0.99, 99)
    rec = np.array([((pva >= c) & (vva.vent_y == 1)).sum() / max((vva.vent_y == 1).sum(), 1) for c in cand])
    thr = float(cand[rec >= 0.88].max()) if (rec >= 0.88).any() else 0.5
    vte["pred_model"] = (pte >= thr).astype(int)
    vte["pred_rule"] = (vte.co2 >= 850).astype(int)
    vte["pred_none"] = 0
    # 과제 3 예측 모델로 판단
    f3 = pd.read_csv("/srv/team/task3/out/all_predictions.csv.gz", parse_dates=["time"])
    f3 = f3[(f3.model == "gru_sched") & (f3["var"] == "co2") & (f3.h <= 30)]
    f3 = f3.groupby(["time", "space", "seed"]).yhat.max().groupby(["time", "space"]).mean().rename("f3_max").reset_index()
    vte = vte.merge(f3, on=["time", "space"], how="left")
    vte["pred_forecast"] = (vte.f3_max >= LIMIT).astype(int)
    covered = vte.f3_max.notna()
    vent_tab = pd.DataFrame({
        "전부 '경보 없음'": metrics(vte.vent_y, vte.pred_none),
        "규칙: 현재 ≥850ppm": metrics(vte.vent_y, vte.pred_rule),
        f"분류 모델 (기준 {thr:.2f})": metrics(vte.vent_y, vte.pred_model)}).T
    vent_fc = pd.DataFrame({
        "규칙": metrics(vte[covered].vent_y, vte[covered].pred_rule),
        "분류 모델": metrics(vte[covered].vent_y, vte[covered].pred_model),
        "과제3 예측 모델": metrics(vte[covered].vent_y, vte[covered].pred_forecast)}).T
    by_space_vent = vte.groupby("space").apply(lambda g: pd.Series(metrics(g.vent_y, g.pred_model))[
        ["n", "TP", "FN", "FP", "재현율", "정밀도"]])
    by_space_vent.index = by_space_vent.index.map(NAMES)
    # 놓친 경보의 상황
    vte["p_model"] = pte  # merge(how="left")는 왼쪽 순서를 유지하므로 pte와 행 순서가 같다
    fn = vte[(vte.vent_y == 1) & (vte.pred_model == 0)]
    fn_case = pd.DataFrame({"space": fn.space.map(NAMES), "time": fn.time, "co2": fn.co2.round(0),
                            "30분내최대": fn.fut_max.round(0), "일정중": fn.sched_on, "확률": fn.p_model.round(3)})

    # ===== 저장·출력 =====
    pd.set_option("display.width", 250)
    show = ["n", "정확도", "정밀도", "재현율", "F1(양성)", "macro-F1", "빈방→사용중 오류율", "TP", "FP", "FN", "TN"]
    print(f"관찰 시간대: {len(slots)}개 × 1시간, 평가 칸 {len(te)}")
    print("\n## 사용 여부 (직접 관찰 기준)\n", occ_tab[show].to_string())
    print("\n## 일정 기반 정답 vs 직접 관찰\n", agree.to_string())
    print("\n## 사용 여부 (시험 기간 전체, 참고)\n", occ_all[show].to_string())
    print("\n## 공간별 (시험 전체, 모델)\n", by_space_occ.to_string())
    print("\n## 상황별 오류율 (시험 전체)\n", case.to_string())
    print("\n## 강의실 인원별 재현율 (사용 중인 칸)\n", size_tab.to_string())
    print(f"\n## 환기 선제 경보 (시험, 양성 비율 {100 * vte.vent_y.mean():.1f}%, 확률 기준 {thr:.2f} — 검증 재현율 88% 이상)\n",
          vent_tab[show].to_string())
    print(f"\n## 과제3 예측 모델 비교 (평가 시점 {covered.sum()}개)\n", vent_fc[show].to_string())
    print("\n## 공간별 환기 경보 (분류 모델)\n", by_space_vent.to_string())
    print("\n## 놓친 경보 (분류 모델)\n", fn_case.head(20).to_string(index=False))

    occ_tab.to_csv(OUT / "occupancy_observed.csv")
    agree.to_csv(OUT / "occupancy_schedule_vs_observed.csv")
    occ_all.to_csv(OUT / "occupancy_test_all.csv")
    case.to_csv(OUT / "occupancy_error_cases.csv")
    size_tab.to_csv(OUT / "occupancy_by_occupants.csv")
    vent_tab.to_csv(OUT / "vent_scorecard.csv")
    vent_fc.to_csv(OUT / "vent_vs_forecast.csv")
    fn_case.to_csv(OUT / "vent_missed.csv", index=False)
    slots.to_csv(OUT / "observation_slots.csv", index=False)
    torch.save({"state_dict": m_occ.state_dict(), "cols": occ_cols, "norm": occ_norm}, OUT.parent / "occ_mlp.pt")
    torch.save({"state_dict": m_vent.state_dict(), "cols": vent_cols, "norm": vent_norm, "threshold": thr},
               OUT.parent / "vent_mlp.pt")
    (OUT / "summary.json").write_text(json.dumps({
        "vent_threshold": thr, "vent_pos_rate_test": float(vte.vent_y.mean()),
        "occ_observed": occ_tab.to_dict("index"), "vent": vent_tab.to_dict("index")}, ensure_ascii=False, indent=1,
        default=str))


if __name__ == "__main__":
    main()
