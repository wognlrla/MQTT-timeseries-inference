"""과제 6: Chronos 결과를 기준선·GRU와 같은 시점에서 비교한다.

- 과거 길이는 모델마다 검증 기간 60분 MAE가 가장 작은 것을 고른다(시험 기간은 선택에 쓰지 않음).
- 주 성적: 시험 기간, 고장 앞뒤 60분 제외. GRU는 시드 3개 평균.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/srv/team/task6/out")
NAMES = {"classroom_a": "강의실 A", "classroom_b": "강의실 B", "meeting": "회의실", "lounge": "라운지"}
LABEL = {"persistence": "현재값 유지", "mean_1h": "최근 1시간 평균", "gru_nosched": "GRU (일정 제외)",
         "gru_sched": "GRU (일정 포함)", "chronos-bolt-tiny": "Chronos-Bolt tiny", "chronos-bolt-small": "Chronos-Bolt small",
         "chronos-bolt-base": "Chronos-Bolt base"}


def md(t):
    head = "| | " + " | ".join(map(str, t.columns)) + " |"
    return "\n".join([head, "|---" * (len(t.columns) + 1) + "|"] +
                     ["| " + str(i) + " | " + " | ".join(map(str, r)) + " |" for i, r in zip(t.index, t.values)])


def main():
    c = pd.read_csv(OUT / "chronos_predictions.csv.gz", parse_dates=["time"])
    c["err"] = (c.y - c.q50).abs()
    c["inside"] = (c.y >= c.q10) & (c.y <= c.q90)
    c["width"] = c.q90 - c.q10
    normal = c[~c.fault_window]

    ctx_tab = normal[normal.h == 60].groupby(["model", "ctx", "split"]).err.mean().unstack("split").round(1)
    best = ctx_tab["val"].groupby(level="model").idxmin().map(lambda x: x[1])
    sel = normal.merge(best.rename("best").reset_index(), on="model")
    sel = sel[sel.ctx == sel.best]
    test = sel[sel.split == "test"]

    a = pd.read_csv("/srv/team/task3/out/all_predictions.csv.gz", parse_dates=["time"])
    a = a[(a["var"] == "co2") & (a.split == "test") & ~a.fault_window & a.model.isin(["persistence", "mean_1h", "gru_nosched", "gru_sched"])]
    a["err"] = (a.y - a.yhat).abs()
    base = a.groupby(["model", "seed", "h"]).err.mean().groupby(["model", "h"]).mean().unstack("h")
    chr_ = test.groupby(["model", "h"]).err.mean().unstack("h")
    comp = pd.concat([base, chr_]).reindex([k for k in LABEL if k in set(base.index) | set(chr_.index)]).round(1)
    comp.index = [LABEL[i] + (f" (과거 {best[i]}칸)" if i in best else "") for i in comp.index]

    cov = test.groupby(["model", "h"]).inside.mean().unstack("h").mul(100).round(1)
    wid = test.groupby(["model", "h"]).width.mean().unstack("h").round(0)
    cov_sp = test[test.h == 60].groupby(["model", "space"]).inside.mean().unstack("space").mul(100).round(1)
    cov_sp.columns = [NAMES[x] for x in cov_sp.columns]

    sp = pd.concat([
        a[a.h == 60].groupby(["model", "seed", "space"]).err.mean().groupby(["model", "space"]).mean().unstack("space"),
        test[test.h == 60].groupby(["model", "space"]).err.mean().unstack("space")]).round(1)
    sp = sp.reindex([k for k in LABEL if k in sp.index])
    sp.index = [LABEL[i] for i in sp.index]
    sp.columns = [NAMES[x] for x in sp.columns]

    allf = c[(c.split == "test") & (c.h == 60)].merge(best.rename("best").reset_index(), on="model")
    allf = allf[allf.ctx == allf.best].groupby("model").err.mean().round(1)

    cost = pd.read_csv(OUT / "cost_clean.csv") if (OUT / "cost_clean.csv").exists() else None

    pd.set_option("display.width", 220)
    parts = [("과거 길이별 60분 CO2 MAE (val로 선택, test는 참고)", ctx_tab),
             ("시험 기간 CO2 MAE (ppm, 고장 제외)", comp),
             ("10~90% 예측 구간 적중률 (%, 이상적 80%)", cov),
             ("예측 구간 평균 폭 (ppm)", wid),
             ("공간별 60분 적중률 (%)", cov_sp),
             ("공간별 60분 CO2 MAE (ppm)", sp),
             ("고장 포함 60분 MAE (Chronos)", allf.to_frame("MAE"))]
    if cost is not None:
        parts.append(("보드 비용", cost.set_index("model")))
    lines = []
    for title, t in parts:
        print(f"\n## {title}\n{t.to_string()}")
        lines.append(f"\n### {title}\n\n{md(t)}\n")
    (OUT / "tables.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
