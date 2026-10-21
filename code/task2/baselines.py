"""과제 2: 5분 격자·평가 시점·정규화 값을 만들고 기준선 3종의 MAE를 잰다.

평가 시점(eval_points.csv)은 과제 3~6에서도 그대로 다시 쓴다(같은 시점·같은 조건 비교).

평가 시점 조건 (공간별 5분 격자의 시각 t, 검증·시험 기간)
- 지금 값 x(t)가 있다(긴 결측 아님)
- 모든 예측 시점 h ∈ {5,15,30,60}분의 정답 y(t+h)가 실제 관측이다(채운 값·결측 아님)
- 세 기준선의 입력(최근 1시간 값, 어제 t+h 값)이 모두 있다 → 세 기준선을 같은 시점에서 비교
- 고장 겹침(fault_window): [t-60분, t+60분] 안에 고장 칸이 있으면 표시. 주 성적은 고장 없는 시점으로 낸다.

기준선
- persistence : ŷ(t+h) = x(t)
- mean_1h     : ŷ(t+h) = 최근 1시간(12칸) 평균
- yesterday   : ŷ(t+h) = x(t+h-24시간)

사용: /srv/team/venv/bin/python /srv/team/task2/baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/srv/team/common")
import preprocess as pp  # noqa: E402

OUT = Path("/srv/team/task2/out")
HORIZONS = [5, 15, 30, 60]
TARGETS = ["co2", "temperature"]
BASELINES = ["persistence", "mean_1h", "yesterday"]


def predictions(g):
    """한 공간의 5분 격자(g, time 인덱스)에서 기준선 예측과 정답을 시점별로 만든다."""
    rows = {}
    for v in TARGETS:
        x = g[v]
        observed = x.notna() & ~g[f"{v}_filled"]
        rows[(v, "now")] = x
        rows[(v, "mean_1h")] = x.rolling(12, min_periods=6).mean()
        for h in HORIZONS:
            k = h // pp.STEP_MIN
            rows[(v, f"y_{h}")] = x.shift(-k).where(observed.shift(-k, fill_value=False))
            rows[(v, f"yday_{h}")] = x.shift(288 - k)  # 288칸 = 24시간
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid, raw, raw_stats, schedule, faults, meta = pp.build()
    grid.to_csv(OUT / "grid_5min.csv.gz", index=False)

    norm = pp.fit_norm(grid, meta)
    (OUT / "norm_stats.json").write_text(json.dumps(
        {"period": meta["split"]["train"], "rule": "학습 기간·고장 아님·결측 아님", "stats": norm},
        ensure_ascii=False, indent=1))

    # 고장 창: 시각 t 기준 앞뒤 60분 안에 고장 칸이 있는가
    grid = grid.sort_values(["space", "time"])
    grid["fault_window"] = (grid.groupby("space")["fault"]
                            .transform(lambda s: s.rolling(25, center=True, min_periods=1).max())
                            .astype(bool))

    points, preds = [], []
    for space, g in grid.groupby("space"):
        g = g.set_index("time")
        p = predictions(g)
        ok = pd.Series(True, index=g.index)
        for v in TARGETS:
            ok &= p[(v, "now")].notna() & p[(v, "mean_1h")].notna()
            for h in HORIZONS:
                ok &= p[(v, f"y_{h}")].notna() & p[(v, f"yday_{h}")].notna()
        ok &= g["split"].isin(["val", "test"])
        sel = g.index[ok]
        points.append(pd.DataFrame({"time": sel, "space": space, "split": g.loc[sel, "split"].values,
                                    "fault_window": g.loc[sel, "fault_window"].values}))
        for v in TARGETS:
            for h in HORIZONS:
                y = p.loc[sel, (v, f"y_{h}")].values
                base = {"persistence": p.loc[sel, (v, "now")].values,
                        "mean_1h": p.loc[sel, (v, "mean_1h")].values,
                        "yesterday": p.loc[sel, (v, f"yday_{h}")].values}
                for b, yhat in base.items():
                    preds.append(pd.DataFrame({
                        "time": sel, "space": space, "split": g.loc[sel, "split"].values,
                        "fault_window": g.loc[sel, "fault_window"].values,
                        "var": v, "h": h, "baseline": b, "y": y, "yhat": yhat}))
    points = pd.concat(points, ignore_index=True)
    points.to_csv(OUT / "eval_points.csv", index=False)
    preds = pd.concat(preds, ignore_index=True)
    preds["abs_err"] = (preds["y"] - preds["yhat"]).abs()
    preds.to_csv(OUT / "baseline_predictions.csv.gz", index=False)

    # 성적표
    def mae(df, by):
        return df.groupby(by)["abs_err"].mean().unstack("h").round(2)

    normal = preds[~preds["fault_window"]]
    tables = {
        "test_overall": mae(normal[normal.split == "test"], ["var", "baseline", "h"]),
        "val_overall": mae(normal[normal.split == "val"], ["var", "baseline", "h"]),
        "test_by_space": mae(normal[normal.split == "test"], ["var", "space", "baseline", "h"]),
        "test_with_faults": mae(preds[preds.split == "test"], ["var", "baseline", "h"]),
    }
    yd = normal[(normal.baseline == "yesterday") & (normal.h == 60)].copy()
    yd["weekday"] = yd["time"].dt.day_name()
    tables["yesterday_by_weekday_h60"] = yd.groupby(["var", "weekday"])["abs_err"].mean().unstack("weekday").round(2)
    for name, t in tables.items():
        t.to_csv(OUT / f"baseline_{name}.csv")

    summary = {
        "eval_points": {s: int(n) for s, n in points.groupby("split").size().items()},
        "eval_points_fault_window": int(points["fault_window"].sum()),
        "raw_cleaning": raw_stats,
    }
    (OUT / "baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 200)
    for name, t in tables.items():
        print(f"\n## {name}\n{t}")
    print("\n", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
