"""과제 3: 학습 모델을 과제 2와 같은 평가 시점에서 기준선과 비교한다.

- 정답 y와 평가 시점은 과제 2 결과(baseline_predictions.csv.gz)를 그대로 쓴다.
- 모델 성적은 시드별 MAE의 평균(± 표준편차).
- 주 성적: 시험 기간, 고장 앞뒤 60분 제외.

사용: /srv/team/venv/bin/python evaluate.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/srv/team/common")
import preprocess as pp  # noqa: E402

ROOT = Path("/srv/team/task3")
OUT = ROOT / "out"
NAMES = {"classroom_a": "강의실 A", "classroom_b": "강의실 B", "meeting": "회의실", "lounge": "라운지"}
LABEL = {"persistence": "현재값 유지", "mean_1h": "최근 1시간 평균", "yesterday": "어제 같은 시각",
         "gru_nosched": "GRU (일정 제외)", "gru_sched": "GRU (일정 포함)", "gru_sched_abs": "GRU (일정 포함, 절댓값 예측)"}
ORDER = list(LABEL)


def phase_table(schedule):
    """평가 시각 t의 상황: 일정 시작 1시간 전 / 일정 중 / 일정 끝난 뒤 1시간 / 그 외"""
    def phase(space, t):
        ev = schedule[schedule.space == space]
        if ((ev.start > t) & (ev.start <= t + pd.Timedelta(minutes=60))).any():
            return "시작 1시간 전"
        if ((ev.start <= t) & (ev.end > t)).any():
            return "일정 중"
        if ((ev.end <= t) & (ev.end > t - pd.Timedelta(minutes=60))).any():
            return "끝난 뒤 1시간"
        return "일정 없음"
    return phase


def main():
    OUT.mkdir(exist_ok=True)
    base = pd.read_csv("/srv/team/task2/out/baseline_predictions.csv.gz", parse_dates=["time"])
    keys = ["time", "space", "var", "h"]
    truth = base[base.baseline == "persistence"][keys + ["split", "fault_window", "y"]]
    rows = [base.rename(columns={"baseline": "model"}).assign(seed=0)[keys + ["split", "fault_window", "model", "seed", "yhat", "y"]]]
    for f in sorted(glob.glob(str(ROOT / "preds" / "*.csv.gz"))):
        name, seed = re.match(r"(.+)_s(\d+)\.csv\.gz", Path(f).name).groups()
        p = pd.read_csv(f, parse_dates=["time"])
        p = truth.merge(p, on=keys, how="left")
        assert p.yhat.notna().all(), f"{f}: 평가 시점 중 예측이 빠진 곳이 있음"
        rows.append(p.assign(model=name, seed=int(seed)))
    df = pd.concat(rows, ignore_index=True)
    df["err"] = (df.y - df.yhat).abs()
    models = [m for m in ORDER if m in df.model.unique()]

    def table(d, by):
        per_seed = d.groupby(["model", "seed"] + by).err.mean()
        m = per_seed.groupby(["model"] + by).mean()
        s = per_seed.groupby(["model"] + by).std().fillna(0)
        return m, s

    res = {}
    for split in ["val", "test"]:
        d = df[(df.split == split) & ~df.fault_window]
        m, s = table(d, ["var", "h"])
        res[split] = (m, s)
    m_all, _ = table(df[df.split == "test"], ["var", "h"])
    m_sp, s_sp = table(df[(df.split == "test") & ~df.fault_window & (df.h == 60)], ["var", "space"])

    # 상황별(일정 단계) 60분 CO₂ 오차
    _, _, _, schedule, _, meta = pp.build()
    ph = phase_table(schedule)
    d60 = df[(df.split == "test") & ~df.fault_window & (df.h == 60) & (df["var"] == "co2")].copy()
    uniq = d60[["space", "time"]].drop_duplicates()
    uniq["phase"] = [ph(s, t) for s, t in zip(uniq.space, uniq.time)]
    d60 = d60.merge(uniq, on=["space", "time"])
    m_ph, _ = table(d60, ["space", "phase"])
    n_ph = uniq.groupby(["space", "phase"]).size()

    def fmt(m, s, var, idx_name):
        t = m.xs(var, level="var").unstack(idx_name)
        sd = s.xs(var, level="var").unstack(idx_name)
        t = t.reindex([x for x in models])
        sd = sd.reindex(t.index)
        nd = 1 if var == "co2" else 2
        cell = t.round(nd).astype(str) + np.where(sd.round(nd) > 0, " ±" + sd.round(nd).astype(str), "")
        cell.index = [LABEL[i] for i in cell.index]
        return cell

    lines = []

    def md(tab):
        head = "| | " + " | ".join(map(str, tab.columns)) + " |"
        sep = "|---" * (len(tab.columns) + 1) + "|"
        body = ["| " + str(i) + " | " + " | ".join(map(str, r)) + " |" for i, r in zip(tab.index, tab.values)]
        return "\n".join([head, sep] + body)

    def add(title, tab):
        lines.append(f"\n### {title}\n\n" + md(tab) + "\n")
        print(f"\n## {title}\n{tab.to_string()}")

    for var, unit in [("co2", "ppm"), ("temperature", "℃")]:
        add(f"시험 기간 {var} MAE ({unit}, 고장 제외) — 예측 시점별", fmt(*res["test"], var, "h"))
    for var, unit in [("co2", "ppm"), ("temperature", "℃")]:
        add(f"검증 기간 {var} MAE ({unit}, 고장 제외)", fmt(*res["val"], var, "h"))
    add("시험 기간 CO2 MAE (ppm, 고장 포함)", fmt(m_all, m_all * 0, "co2", "h"))
    for var in ["co2", "temperature"]:
        t = fmt(m_sp, s_sp, var, "space")
        t.columns = [NAMES[c] for c in t.columns]
        add(f"공간별 60분 {var} MAE (시험, 고장 제외)", t)
    for space in ["meeting", "classroom_a", "classroom_b", "lounge"]:
        t = m_ph.xs(space, level="space").unstack("phase").reindex(models).round(1)
        t.index = [LABEL[i] for i in t.index]
        t.columns = [f"{c} (n={n_ph.loc[(space, c)]})" for c in t.columns]
        add(f"{NAMES[space]}: 상황별 60분 CO2 MAE (ppm, 시험)", t)

    # 확인 기준
    mt = res["test"][0]
    p60 = mt.loc[("persistence", "co2", 60)]
    checks = {}
    for mdl in [x for x in models if x.startswith("gru")]:
        g60 = mt.loc[(mdl, "co2", 60)]
        checks[mdl] = {
            "co2_60_mae": round(g60, 2), "persistence_co2_60": round(p60, 2),
            "co2_60_improvement_pct": round(100 * (1 - g60 / p60), 1), "co2_pass(>=30%)": bool(g60 <= 0.7 * p60),
            "temp_60_mae": round(mt.loc[(mdl, "temperature", 60)], 3),
            "persistence_temp_60": round(mt.loc[("persistence", "temperature", 60)], 3),
            "temp_pass": bool(mt.loc[(mdl, "temperature", 60)] < mt.loc[("persistence", "temperature", 60)])}
    if {"gru_sched", "gru_nosched"} <= set(models):
        checks["schedule_value_co2_60_ppm"] = round(mt.loc[("gru_nosched", "co2", 60)] - mt.loc[("gru_sched", "co2", 60)], 2)
        sp = m_sp.xs("co2", level="var")
        checks["schedule_value_by_space_co2_60_ppm"] = {
            NAMES[s]: round(sp.loc[("gru_nosched", s)] - sp.loc[("gru_sched", s)], 1) for s in NAMES}
    print("\n## 확인 기준\n" + json.dumps(checks, ensure_ascii=False, indent=1))
    (OUT / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1))
    (OUT / "tables.md").write_text("\n".join(lines))
    df.to_csv(OUT / "all_predictions.csv.gz", index=False)


if __name__ == "__main__":
    main()
