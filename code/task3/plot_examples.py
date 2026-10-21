"""과제 3: 시험 기간 하루의 '60분 뒤' 예측과 실제를 겹쳐 그린다 (보고서용 예시 그래프)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/srv/team/common")
import preprocess as pp  # noqa: E402

OUT = Path("/srv/team/task3/out")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-10-07"
SPACES = [("classroom_a", "강의실 A"), ("meeting", "회의실")]
SHOW = {"persistence": ("현재값 유지", "grey", ":"), "gru_nosched": ("GRU 일정 제외", "tab:orange", "--"),
        "gru_sched": ("GRU 일정 포함", "tab:blue", "-")}

df = pd.read_csv(OUT / "all_predictions.csv.gz", parse_dates=["time"])
d = df[(df["var"] == "co2") & (df.h == 60) & (df.time.dt.strftime("%Y-%m-%d") == DAY)
       & (df.time.dt.hour.between(7, 20)) & (df.seed == 0)].copy()
d["target_time"] = d.time + pd.Timedelta(minutes=60)
_, _, _, schedule, _, _ = pp.build()

fig, axes = plt.subplots(len(SPACES), 1, figsize=(11, 7), sharex=True)
for ax, (space, name) in zip(axes, SPACES):
    s = d[d.space == space]
    truth = s[s.model == "persistence"].sort_values("target_time")
    ax.plot(truth.target_time, truth.y, color="black", lw=2, label="실제")
    for m, (lab, c, ls) in SHOW.items():
        p = s[s.model == m].sort_values("target_time")
        ax.plot(p.target_time, p.yhat, color=c, ls=ls, lw=1.4, label=f"{lab} (60분 전 예측)")
    for r in schedule[(schedule.space == space) & (schedule.start.dt.strftime("%Y-%m-%d") == DAY)].itertuples():
        ax.axvspan(r.start, r.end, color="tab:green", alpha=0.08)
    ax.axhline(1000, color="red", lw=0.7)
    ax.set_ylabel("CO2 (ppm)")
    ax.set_title(f"{name} — {DAY} (초록 = 일정)")
axes[0].legend(fontsize=8, ncol=4, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / f"fig_example_{DAY}.png", dpi=130)
print(OUT / f"fig_example_{DAY}.png")
