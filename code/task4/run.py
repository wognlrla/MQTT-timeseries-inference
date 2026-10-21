"""과제 4: 탐지기 학습·임계값 결정·평가.

1. 규칙 기준값을 학습 기간으로 정한다.
2. 오토인코더를 정상(학습 기간, 고장 없음)만으로 학습한다.
3. 오토인코더 임계값은 검증 기간(고장 없음)에서 '헛경보 하루 0.5회 이하'를 만족하는 가장 낮은 값으로 정한다.
   (규칙의 헛경보 몫을 남겨 두고, 합쳐서 하루 1회 이하가 되도록)
4. 시험 기간 고장 17건으로 평가한다. 시험 기간은 임계값 결정에 쓰지 않는다.

경보 묶기: 같은 공간·같은 탐지기의 경보가 30분 안에 이어지면 하나의 '경보 사건'으로 본다.
탐지 판정: 고장 [시작, 끝+30분] 안에 그 공간의 경보가 있으면 탐지. 지연 = 첫 경보 - 고장 시작.
헛경보: 어떤 고장 [시작-30분, 끝+60분]과도 겹치지 않는 경보 사건.

사용: /srv/team/venv/bin/python run.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/srv/team/common")
sys.path.insert(0, "/srv/team/task4")
import detectors as D  # noqa: E402
import preprocess as pp  # noqa: E402

ROOT = Path("/srv/team/task4")
OUT = ROOT / "out"
NAMES = {"classroom_a": "강의실 A", "classroom_b": "강의실 B", "meeting": "회의실", "lounge": "라운지"}
TYPES = {"spike": "튄 값", "stuck": "값 고정", "drift": "영점 틀어짐", "ac_failure": "냉방 정지", "vent_failure": "환기 정지"}
GAP = pd.Timedelta(minutes=30)
FA_BUDGET_AE = 0.5
THR_GRID = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]


def episodes(al):
    """경보 → 경보 사건(공간·탐지기별로 30분 안에 이어지면 하나)."""
    if al.empty:
        return pd.DataFrame(columns=["space", "detector", "start", "end", "var"])
    al = al.sort_values(["space", "detector", "time"])
    new = (al.groupby(["space", "detector"]).time.diff() > GAP) | al.groupby(["space", "detector"]).time.diff().isna()
    al = al.assign(ep=new.cumsum())
    return al.groupby("ep").agg(space=("space", "first"), detector=("detector", "first"), start=("time", "min"),
                                end=("time", "max"), var=("var", "first")).reset_index(drop=True)


def evaluate(al, faults, period, days):
    a, b = period
    al = al[(al.time >= a) & (al.time <= b)]
    rows = []
    for f in faults.itertuples():
        hit = al[(al.space == f.space) & (al.time >= f.start) & (al.time <= f.end + GAP)].sort_values("time")
        rows.append({"space": f.space, "type": f.type, "start": f.start, "sensor": f.sensor,
                     "detected": len(hit) > 0,
                     "delay_min": (hit.time.iloc[0] - f.start).total_seconds() / 60 if len(hit) else np.nan,
                     "first_var": hit["var"].iloc[0] if len(hit) else ""})
    ev = pd.DataFrame(rows)
    ep = episodes(al)
    fa = []
    for e in ep.itertuples():
        f = faults[faults.space == e.space]
        overlap = ((f.start - GAP <= e.end) & (f.end + 2 * GAP >= e.start)).any()
        if not overlap:
            fa.append(e)
    fa = pd.DataFrame(fa)
    return ev, fa, len(fa) / days


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid, raw, _, schedule, faults, meta = pp.build()
    norm = json.loads(Path("/srv/team/task2/out/norm_stats.json").read_text())["stats"]
    sp = pp.split_bounds(meta)
    val_days = (sp["val"][1] - sp["val"][0]).days
    test_days = (sp["test"][1] - sp["test"][0]).days + 1
    test_faults = faults[(faults.start >= sp["test"][0])]

    # 1. 규칙
    thr = D.calibrate_spike(raw, meta)
    cal = D.calibrate_level(grid, meta)
    rule_sets = {"rule_spike": D.spike_alarms(raw, thr), "rule_stuck": D.stuck_alarms(raw),
                 "rule_level": D.level_alarms(grid, cal), "rule_night": D.night_alarms(grid, cal)}
    rules = pd.concat(rule_sets.values(), ignore_index=True)

    # 2. 오토인코더
    w = D.ae_windows(grid, schedule, meta, norm)
    model, info = D.train_ae(w, dev=dev)
    err = D.ae_errors(model, w, dev=dev)
    scale = D.ae_scale(err, w)
    z = err / scale
    score = np.where(w["ok"], z.max(axis=1), np.nan)
    cause = np.array(pp.VARS)[z.argmax(axis=1)]
    sc = pd.DataFrame({"time": w["time"], "space": w["space"], "split": w["split"], "score": score, "var": cause}
                      | {f"z_{v}": z[:, i] for i, v in enumerate(pp.VARS)})
    sc.to_csv(OUT / "ae_scores.csv.gz", index=False)
    torch.save({"state_dict": model.state_dict(), "scale": scale.tolist(), "norm": norm, "win": D.WIN,
                "score_last": D.SCORE_LAST, "dims": [w["x"].shape[1], w["c"].shape[1]], "info": info},
               ROOT / "ae.pt")

    def ae_alarms(t):
        a = sc[sc.score > t]
        return a.assign(detector="ae")[["time", "space", "detector", "var", "score"]]

    # 3. 임계값: 검증 기간 헛경보 예산
    sweep = []
    for t in THR_GRID:
        _, _, fa_val = evaluate(ae_alarms(t), faults, sp["val"], val_days)
        ev_t, _, fa_test = evaluate(ae_alarms(t), test_faults, sp["test"], test_days)
        ev_c, _, fa_c = evaluate(pd.concat([ae_alarms(t), rules]), test_faults, sp["test"], test_days)
        sweep.append({"threshold": t, "val_FA_per_day(AE)": round(fa_val, 2),
                      "test_detect(AE)": f"{int(ev_t.detected.sum())}/{len(ev_t)}", "test_FA_per_day(AE)": round(fa_test, 2),
                      "test_detect(combined)": f"{int(ev_c.detected.sum())}/{len(ev_c)}",
                      "test_FA_per_day(combined)": round(fa_c, 2)})
    sweep = pd.DataFrame(sweep)
    chosen = next(r["threshold"] for r in sweep.to_dict("records") if r["val_FA_per_day(AE)"] <= FA_BUDGET_AE)
    _, _, rule_fa_val = evaluate(rules, faults, sp["val"], val_days)

    # 4. 시험 평가
    sets = {"모델(AE)": ae_alarms(chosen), "규칙": rules, "결합": pd.concat([ae_alarms(chosen), rules])}
    results, fas = {}, {}
    for name, al in sets.items():
        ev, fa, fa_day = evaluate(al, test_faults, sp["test"], test_days)
        results[name] = (ev, fa_day)
        fas[name] = fa
    card = []
    for t, label in TYPES.items():
        row = {"고장": label, "건수": int((test_faults.type == t).sum())}
        for name, (ev, _) in results.items():
            e = ev[ev.type == t]
            row[f"{name} 탐지"] = f"{int(e.detected.sum())}/{len(e)}"
            row[f"{name} 지연(분)"] = round(e.delay_min.mean(), 0) if e.detected.any() else "-"
        card.append(row)
    total = {"고장": "합계", "건수": len(test_faults)}
    for name, (ev, fa_day) in results.items():
        total[f"{name} 탐지"] = f"{int(ev.detected.sum())}/{len(ev)} ({100 * ev.detected.mean():.0f}%)"
        total[f"{name} 지연(분)"] = round(ev.delay_min.mean(), 0)
    card.append(total)
    card = pd.DataFrame(card)
    fa_row = {name: round(fa_day, 2) for name, (_, fa_day) in results.items()}

    # 규칙별 탐지·헛경보
    by_rule = []
    for rname, al in rule_sets.items():
        ev_r, _, fa_t = evaluate(al, test_faults, sp["test"], test_days)
        _, _, fa_v = evaluate(al, faults, sp["val"], val_days)
        row = {"규칙": rname, "검증 헛경보/일": round(fa_v, 2), "시험 헛경보/일": round(fa_t, 2)}
        for t, label in TYPES.items():
            e = ev_r[ev_r.type == t]
            row[label] = f"{int(e.detected.sum())}/{len(e)}"
        by_rule.append(row)
    by_rule = pd.DataFrame(by_rule)

    # 오토인코더 원인 상위 2개 (첫 경보 시점)
    top2 = []
    for f in test_faults.itertuples():
        h = sc[(sc.space == f.space) & (sc.time >= f.start) & (sc.time <= f.end + GAP) & (sc.score > chosen)]
        if len(h):
            zz = h.iloc[0][[f"z_{v}" for v in pp.VARS]].astype(float).sort_values(ascending=False)
            top2.append(", ".join(f"{k[2:]} {v:.1f}" for k, v in zz.iloc[:2].items()))
        else:
            top2.append("")
    detail = results["결합"][0].copy()
    detail["AE 원인 상위2(z)"] = top2
    detail["모델"] = results["모델(AE)"][0].detected.values
    detail["규칙"] = results["규칙"][0].detected.values
    detail["AE 원인 항목"] = results["모델(AE)"][0].first_var.values
    detail["space"] = detail.space.map(NAMES)
    detail["type"] = detail.type.map(TYPES)

    # 공간별 검증 기간 AE 점수 분포 (운영 제안용)
    per_space = sc[(sc.split == "val")].groupby("space").score.quantile([0.99, 0.999]).unstack().round(2)
    per_space.index = per_space.index.map(NAMES)

    pd.set_option("display.width", 250)
    print("AE 학습:", info, "| 선택 임계값:", chosen, "| 규칙 검증 헛경보/일:", round(rule_fa_val, 2))
    print("CO2 평소 최고치 기준:", {NAMES[k]: round(v) for k, v in cal["level"].items()},
          "| 새벽 기준:", {NAMES[k]: round(v) for k, v in cal["night"].items()})
    print("spike 기준:", {f"{NAMES[s]}/{v}": round(t, 1) for (s, v), t in thr.items()})
    print("\n## 탐지 성적표 (시험)\n", card.to_string(index=False))
    print("\n## 하루당 헛경보 (시험, 전체 공간 합)\n", fa_row)
    print("\n## 고장별 상세\n", detail[["space", "type", "start", "sensor", "모델", "규칙", "delay_min", "AE 원인 상위2(z)"]].to_string(index=False))
    print("\n## 규칙별 탐지\n", by_rule.to_string(index=False))
    print("\n## 임계값 변화\n", sweep.to_string(index=False))
    print("\n## 헛경보 목록 (결합)\n", fas["결합"].to_string(index=False) if len(fas["결합"]) else "없음")
    print("\n## 검증 기간 공간별 AE 점수 분위\n", per_space.to_string())

    card.to_csv(OUT / "scorecard.csv", index=False)
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
    by_rule.to_csv(OUT / "by_rule.csv", index=False)
    detail.to_csv(OUT / "fault_detail.csv", index=False)
    fas["결합"].to_csv(OUT / "false_alarms.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "ae_threshold": chosen, "ae_train": info, "fa_per_day_test": fa_row, "rule_fa_per_day_val": rule_fa_val,
        "spike_thresholds": {f"{s}/{v}": t for (s, v), t in thr.items()}, "stuck_minutes": D.STUCK_MIN,
        "level_thresholds": cal["level"], "night_thresholds": cal["night"]},
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
